# dataset.py
import os
import json
import pandas as pd
import shelve
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import math  # at top if not already
# =========================
# Token ID layout (no SEP; duration tokens + EOW between stitched windows)
# =========================
# =========================
# Token ID layout (no SEP; duration tokens + EOW between stitched windows)
# =========================
PAD_ID = 0
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4
EOW_ID = 5

# --- NEW: Duration tokens are in DAYS via “coin” factorization ---
# Coins are ordered descending to keep token counts small and semantics clear
#  - <=30d kept at 1d precision
#  - 31–180d quantized weekly (7d)
#  - >180d quantized monthly-ish (30d)
# ===== Duration token config (DAYS coin system) =====
# ===== Duration token config (days) =====
DUR_BASE = 6
DURATION_UNITS_DAYS = [180, 60, 30, 14, 7, 3, 1]  # descending coins
NUM_DUR = len(DURATION_UNITS_DAYS)
CODES_START = DUR_BASE + NUM_DUR

# Optional: keep old name to avoid import crashes in other code
DURATION_UNITS = None

def _quantize_days(delta_days: int) -> int:
    """
    Simple policy:
      - Δd <= 180: exact days (no rounding)
      - Δd > 180 : floor to month multiple (30d)
    """
    d = max(0, int(delta_days))
    if d <= 180: return d
    return (d // 30) * 30  # floor to month (no overshoot)

def _emit_duration_tokens_from_days(delta_days: int) -> list[int]:
    """
    Quantize (per _quantize_days) then greedy-factorize with day coins.
    """
    qd = _quantize_days(delta_days)
    if qd <= 0: return []
    toks, rem = [], qd
    for i, u in enumerate(DURATION_UNITS_DAYS):
        if rem <= 0: break
        k = rem // u
        if k:
            toks.extend([DUR_BASE + i] * int(k))
            rem -= int(k) * u
    return toks

# =========================
# Collate
# =========================
def custom_collate(batch):
    flat = []
    for el in batch:
        if isinstance(el, list): flat.extend(el)
        else: flat.append(el)
    if not flat: return {}

    keys = flat[0].keys()
    out = {k: [] for k in keys}
    for item in flat:
        for k in keys: out[k].append(item[k])

    for k in keys:
        if isinstance(out[k][0], np.ndarray):
            out[k] = torch.from_numpy(np.stack(out[k], axis=0))
        elif isinstance(out[k][0], torch.Tensor):
            out[k] = torch.stack(out[k], dim=0)
    return out

# =========================
# Helpers
# =========================
def _pad_to_length_1d(arr: np.ndarray, target_len: int, pad_val: int) -> np.ndarray:
    if arr.shape[0] >= target_len: return arr[:target_len]
    out = np.full(target_len, pad_val, dtype=arr.dtype)
    out[:arr.shape[0]] = arr
    return out

def _months_from_days(delta_days: int) -> int:
    # rounded months by default; if you prefer floor+skip, change here consistently with your checks
    return int(round(max(0, delta_days) / 30.4375))

def _emit_duration_tokens(delta_months: int) -> list[int]:
    """Greedy factorization (descending) using DURATION_UNITS."""
    rem = max(0, int(delta_months))
    toks = []
    for u in DURATION_UNITS:
        k = rem // u
        if k > 0:
            toks.extend([DUR_BASE + DURATION_UNITS.index(u)] * k)
            rem -= k * u
    return toks

def build_ntp_targets(tokens: np.ndarray,
                      pad_id: int = PAD_ID,
                      eow_id: int = EOW_ID) -> np.ndarray:
    """
    NTP targets with ignore_index=-1 when CURRENT or NEXT is PAD/EOW.
    Duration tokens are learned normally.
    """
    T = tokens.shape[0]
    next_tok = np.roll(tokens, -1); next_tok[-1] = pad_id
    targets = next_tok.astype(np.int64)

    cur_block = (tokens == pad_id) | (tokens == eow_id)
    nxt_block = (next_tok == pad_id) | (next_tok == eow_id)
    targets[cur_block] = -1
    targets[nxt_block] = -1
    return targets

# =========================
# UnifiedSeqEHRDataset (duration-token version) + DEBUG hooks
# =========================
class UnifiedSeqEHRDataset(Dataset):
    """
    Emits dicts with:
      - token_arr: np.int64 [max_length]
      - targets:   np.int64 [max_length] (ignore_index=-1 as above)

    Per-visit serialization (masked space) with category ordering.
    After each visit (except last), inserts duration tokens encoding Δmonths to next visit.
    No day-based embeddings, no patient_ids.

    train=True, multiwin=True:
      - samples short windows from serialized patients and stitches multiple windows,
        inserting EOW_ID between stitched windows.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        max_length: int = 512,
        model: str = 'GPT',
        folder: str = "./data/",
        age=None,
        train: bool = False,
        agg_labels: bool = False,
        temporal_decay=None,
        multiwin: bool = False,
        header_file=None,
        shuffle_within_groups: bool = True,
        max_eval_windows_per_patient: int | None = 64,
        # ---- DEBUG knobs ----
        debug: bool = False,
        debug_limit: int = 20
    ):
        if header_file is None:
            header_file = os.path.join(folder, "data_files", "filtered_headers_token_merge.json")
        self.shuffle_within_groups = shuffle_within_groups

        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.folder = folder
        self.age = age
        self.model = model
        self.train = train
        self.agg_labels = agg_labels
        self.temporal_decay = temporal_decay
        self.multiwin = multiwin
        self.max_eval_windows_per_patient = max_eval_windows_per_patient

        # DEBUG
        self.debug = bool(debug)
        self.debug_limit = int(debug_limit)
        self._debug_hits = 0

        print("Open db files:")
        self.db_dict = {part: shelve.open(self.folder + '/processed_part_' + str(part) + '.shelve') for part in tqdm(range(40))}

        # Original space -> masked space
        remove_path = os.path.join(self.folder, "data_files", "delete_inds.npy")
        self.remove_inds = np.load(remove_path)
        total_prev_tokens = 57735
        self.keep_mask = np.ones(total_prev_tokens, dtype=bool)
        self.keep_mask[self.remove_inds] = False

        # Day-of-visit column in MASKED space
        self.day_col = 35845

        # Masked-space headers aligned with a[:, keep_mask]
        with open(header_file, "r") as f:
            headers = json.load(f)
        self.headers = headers

        # Category mapping
        _CAT_OTHER = 0; _CAT_AGE = 1; _CAT_DEMO = 2; _CAT_DIAG = 3; _CAT_LAB = 4; _CAT_MED = 5
        def _cat_from_header(txt: str) -> int:
            t = txt or ""
            if t.startswith("demographics_age"): return _CAT_AGE
            if t.startswith("demographics_"):    return _CAT_DEMO
            if t.startswith("Diagnosis:"):       return _CAT_DIAG
            if t.startswith("Lab:"):             return _CAT_LAB
            if t.startswith("Medication:"):      return _CAT_MED
            return _CAT_OTHER

        self.col_category_masked = np.fromiter(
            (_cat_from_header(t) for t in headers),
            dtype=np.int8,
            count=len(headers)
        )

        self.rng = np.random.default_rng()

    # ---- DEBUG helpers ----
    def _dbg(self, *msg):
        if self.debug and self._debug_hits < self.debug_limit:
            print(*msg)

    def _dbg_ctx(self, seq, pos, span=8):
        if not self.debug or self._debug_hits >= self.debug_limit: return
        L = max(0, pos - span); R = min(len(seq), pos + span + 1)
        window = seq[L:R].tolist() if hasattr(seq, "tolist") else list(seq[L:R])
        marker = " " * (pos - L) + "↑"
        print("   ctx:", window)
        print("   mark:", marker)

    # -------- Optional: randomly drop rows (visits) in CSR
    def _remove_random_rows_csr(self, a, percentage=0.1):
        n = a.shape[0]
        if n <= 1 or percentage <= 0.0: return a
        keep = np.random.rand(n) >= percentage
        if not keep.any(): keep[np.random.randint(n)] = True
        return a[keep]

    # -------- Vectorized category ordering for one visit
    def _order_visit_cols(self, masked_cols: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if masked_cols.size == 0: return masked_cols
        # exclude day column
        masked_cols = masked_cols[masked_cols != self.day_col]
        if masked_cols.size == 0: return masked_cols

        cats = self.col_category_masked[masked_cols]
        age  = masked_cols[cats == 1]
        demo = masked_cols[cats == 2]
        diag = masked_cols[cats == 3]
        lab  = masked_cols[cats == 4]
        med  = masked_cols[cats == 5]

        if self.shuffle_within_groups:
            if diag.size > 1: diag = rng.permutation(diag)
            if lab.size  > 1: lab  = rng.permutation(lab)
            if med.size  > 1: med  = rng.permutation(med)

        return np.concatenate([age, demo, diag, lab, med], axis=0)

    # -------- Per-patient serialization (masked space)
    def _serialize_patient(self, a_masked, days_data, rng: np.random.Generator):
        """
        Returns:
          token_list_final: np.int64 [N_tokens]
        """
        n_visits = a_masked.shape[0]
        token_list = []

        indptr = a_masked.indptr
        indices = a_masked.indices

        for r in range(n_visits):
            s, e = indptr[r], indptr[r + 1]
            cols = indices[s:e]

            # DEBUG: day col present
            if self.debug and np.any(cols == self.day_col):
                self._debug_hits += 1
                self._dbg(f"[DBG] visit {r}: day_col present (masked col={self.day_col})")

            # DEBUG: masked range sanity
            bad_neg = cols[cols < 0]
            bad_big = cols[cols >= len(self.headers)]
            if self.debug and (bad_neg.size or bad_big.size):
                self._debug_hits += 1
                self._dbg(f"[DBG] visit {r}: out-of-range masked cols:",
                          {"neg": bad_neg.tolist(), "ge_len": bad_big.tolist()})

            # (Note) masked-space indices can be small numbers [0..], that's fine;
            # the shift to token ids happens below via + CODES_START.

            # emit visit codes (if any non-day cols exist)
            if cols.size > 0 and not (cols.size == 1 and cols[0] == self.day_col):
                ordered_cols = self._order_visit_cols(cols, rng)
                if ordered_cols.size > 0:
                    ordered_tok_ids = ordered_cols + CODES_START  # shift to code space
                    # DEBUG: token id sanity post-shift
                    if self.debug and np.any(ordered_tok_ids < CODES_START):
                        self._debug_hits += 1
                        self._dbg(f"[DBG] visit {r}: shifted id < CODES_START, min={int(ordered_tok_ids.min())}")
                    token_list.extend(ordered_tok_ids.tolist())

            # emit duration tokens to next visit (day-based, quantize → factorize)
            if r + 1 < n_visits:
                delta_days = int(days_data[r+1]) - int(days_data[r])
                durs = _emit_duration_tokens_from_days(delta_days)
                if self.debug:
                    bad = [t for t in durs if not (DUR_BASE <= t < DUR_BASE + NUM_DUR)]
                    if bad:
                        self._debug_hits += 1
                        self._dbg(f"[DBG] visit {r}: bad duration ids {bad}, Δd={delta_days}, qΔd={_quantize_days(delta_days)}")
                token_list.extend(durs)



        token_list_final = np.asarray(token_list, dtype=np.int64)
        return token_list_final

    # -------- Sample a training window from serialized stream (≤ max_length//2)
    def _sample_train_window_from_serialized(self, token_list_final):
        if token_list_final.size == 0:
            tokens = np.full(self.max_length, PAD_ID, np.int64)
            tokens[:1] = [EOW_ID]  # ensure at least one non-PAD to build targets
            targets = build_ntp_targets(tokens, pad_id=PAD_ID, eow_id=EOW_ID)
            return {"token_arr": tokens, "targets": targets}

        start = np.random.randint(0, max(1, token_list_final.size - 1))
        max_tokens = self.max_length // 2
        end = min(token_list_final.size, start + max_tokens)
        span_tokens = token_list_final[start:end]

        tokens = _pad_to_length_1d(span_tokens.astype(np.int64), self.max_length, PAD_ID)
        targets = build_ntp_targets(tokens, pad_id=PAD_ID, eow_id=EOW_ID)
        return {"token_arr": tokens, "targets": targets}

    # -------- Combine multiple windows (multiwin=True) with EOW between
    def _combine_windows(self, windows):
        T = self.max_length
        tokens = np.full(T, PAD_ID, dtype=np.int64)
        pos = 0
        for i, w in enumerate(windows):
            t = w['token_arr']; real_len = int((t != PAD_ID).sum())
            t = t[:real_len]

            seg_len = min(real_len, max(0, T - pos))
            if seg_len <= 0: break

            tokens[pos:pos + seg_len] = t[:seg_len]
            pos += seg_len

            # insert EOW if room and not last window
            if i < len(windows) - 1 and pos < T:
                tokens[pos] = EOW_ID
                pos += 1
            if pos >= T: break

        targets = build_ntp_targets(tokens, pad_id=PAD_ID, eow_id=EOW_ID)

        # ---- DEBUG scan after stitching ----
        if self.debug and self._debug_hits < self.debug_limit:
            # find any tokens in {1..4} (legacy specials) within the stitched sample
            suspicious = np.where((tokens > 0) & (tokens < DUR_BASE) & (tokens != EOW_ID))[0]
            if suspicious.size:
                self._debug_hits += 1
                self._dbg(f"[DBG] _combine_windows: found {len(suspicious)} tokens in {{1..4}}")
                for pos_i in suspicious[:3]:
                    self._dbg(f"   pos={pos_i} id={int(tokens[pos_i])}")
                    self._dbg_ctx(tokens, int(pos_i), span=8)

            # EOW placement anomalies
            eows = np.where(tokens == EOW_ID)[0]
            if eows.size:
                if eows[0] == 0 or eows[-1] == (len(tokens) - 1) or np.any(np.diff(eows) == 1):
                    self._debug_hits += 1
                    self._dbg(f"[DBG] EOW anomaly: positions={eows.tolist()}")
                    for pos_i in eows[:3]:
                        self._dbg_ctx(tokens, int(pos_i), span=8)

        return {'token_arr': tokens, 'targets': targets}

    def __getitem__(self, idx):
        loc = self.df.iloc[idx]
        part = int(loc["part"])
        patient_id = str(loc["patient_id"])

        # Load masked-space matrix
        a = self.db_dict[part][patient_id]   # CSR in original space
        a = a[:, self.keep_mask]             # -> masked space (headers align with this)

        if self.model == 'GPT' and self.train:
            a = self._remove_random_rows_csr(a, percentage=0.1)

        # extract integer visit days from masked day column
        days_data = a[:, self.day_col].toarray().astype("int").ravel()

        # Serialize per-visit
        rng = self.rng if self.shuffle_within_groups else np.random.default_rng(0)
        token_list_final = self._serialize_patient(a, days_data, rng)

        if (not self.train) or (not self.multiwin):
            if self.train:
                window = self._sample_train_window_from_serialized(token_list_final)
                return [window]
            else:
                # evaluation: greedy fixed-size chunks up to max_length
                items = []
                N = token_list_final.size
                cur = 0
                while cur < N:
                    end = min(N, cur + self.max_length)
                    tokens_raw = token_list_final[cur:end]
                    tokens = _pad_to_length_1d(tokens_raw.astype(np.int64), self.max_length, PAD_ID)
                    targets = build_ntp_targets(tokens, pad_id=PAD_ID, eow_id=EOW_ID)
                    items.append({'token_arr': tokens, 'targets': targets})
                    cur = end
                    if (self.max_eval_windows_per_patient is not None and len(items) >= self.max_eval_windows_per_patient):
                        break
                return items
        else:
            # multiwin training: stitch multiple short windows (possibly different patients)
            windows = []
            total_len = 0
            # first window from current patient
            win = self._sample_train_window_from_serialized(token_list_final)
            real_len = int((win['token_arr'] != PAD_ID).sum())
            windows.append(win); total_len += real_len

            while total_len < self.max_length:
                loc_i = self.df.iloc[np.random.randint(len(self.df))]
                part_i = int(loc_i["part"])
                pid_i = str(loc_i["patient_id"])
                a_i = self.db_dict[part_i][pid_i][:, self.keep_mask]
                if self.model == 'GPT' and self.train:
                    a_i = self._remove_random_rows_csr(a_i, percentage=0.1)
                days_i = a_i[:, self.day_col].toarray().astype("int").ravel()
                tl_i = self._serialize_patient(a_i, days_i, self.rng)

                win_i = self._sample_train_window_from_serialized(tl_i)
                real_len_i = int((win_i['token_arr'] != PAD_ID).sum())
                need = real_len_i + 1  # +EOW slot
                if total_len + need > self.max_length: break
                windows.append(win_i)
                total_len += need

            return self._combine_windows(windows)

    def __len__(self):
        return len(self.df)

    def _cleanup(self):
        for k in list(self.db_dict.keys()):
            try: self.db_dict[k].close()
            except Exception: pass
        self.db_dict = {}

    # -------- Optional: one-shot patient debugger
    def debug_patient(self, part: int, patient_id: str, span: int = 12):
        a = self.db_dict[int(part)][str(patient_id)][:, self.keep_mask]
        days = a[:, self.day_col].toarray().astype("int").ravel()
        tl = self._serialize_patient(a, days, np.random.default_rng(0))
        tiny = np.where((tl > 0) & (tl < DUR_BASE) & (tl != EOW_ID))[0]
        print(f"[debug_patient] tokens={len(tl)}, suspicious(1..4) count={len(tiny)}")
        for pos in tiny[:5]:
            L = max(0, pos - span); R = min(len(tl), pos + span + 1)
            print(f"  pos={int(pos)} id={int(tl[pos])} window={tl[L:R].tolist()}")
