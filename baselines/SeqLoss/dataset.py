# dataset.py
import os
import json
import pandas as pd
import shelve
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# =========================
# Special token IDs
# =========================
PAD_ID = 0
SEP_ID = 1        # visit delimiter (we DO learn to predict SEP)
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4
EOW_ID = 5        # terminal token for intra-window segments; never learned

# Category tags (masked-space headers)
_CAT_OTHER = 0
_CAT_AGE   = 1
_CAT_DEMO  = 2
_CAT_DIAG  = 3
_CAT_LAB   = 4
_CAT_MED   = 5

# =========================
# Collate
#  - Flattens list-of-items into a single list of dicts
#  - Stacks numpy arrays / tensors
# =========================
def custom_collate(batch):
    # Flatten batch-of-lists into a single list of dicts
    flat = []
    for el in batch:
        if isinstance(el, list): flat.extend(el)
        else: flat.append(el)
    if not flat: return {}

    keys = flat[0].keys()
    out = {k: [] for k in keys}
    for item in flat:
        for k in keys:
            out[k].append(item[k])

    # Special-case padding for 'remaining_indices' whose K can vary per sample
    if "remaining_indices" in out and len(out["remaining_indices"]) > 0:
        arrs = out["remaining_indices"]
        # only pad numpy arrays; if already torch, convert to numpy first (should be numpy from dataset)
        np_arrs = []
        Kmax = 0
        for a in arrs:
            if isinstance(a, torch.Tensor): a = a.cpu().numpy()
            np_arrs.append(a)
            if a.ndim != 2:
                raise ValueError(f"remaining_indices must be 2D (T,K); got shape {a.shape}")
            Kmax = max(Kmax, a.shape[1])
        # right-pad with -1 to Kmax
        padded = []
        for a in np_arrs:
            T, K = a.shape
            if K < Kmax:
                pad = np.full((T, Kmax - K), -1, dtype=a.dtype)
                a = np.concatenate([a, pad], axis=1)
            padded.append(a)
        out["remaining_indices"] = [p for p in padded]

    # Now stack numpy → tensors (others unchanged)
    for k in keys:
        # all items under k should now have the same shape
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

def build_ntp_targets(tokens: np.ndarray,
                      pad_id: int,
                      sep_id: int = SEP_ID,
                      eow_id: int = EOW_ID) -> np.ndarray:
    """
    Next-token prediction with ignore_index = -1 where:
      - CURRENT token is PAD or EOW
      - NEXT token is PAD or EOW
    Importantly we DO learn to predict SEP when it is NEXT, and allow loss when CURRENT is SEP.
    """
    T = tokens.shape[0]
    next_tok = np.roll(tokens, -1); next_tok[-1] = pad_id
    targets = next_tok.astype(np.int64)

    cur_block = (tokens == pad_id) | (tokens == eow_id)
    nxt_block = (next_tok == pad_id) | (next_tok == eow_id)
    targets[cur_block] = -1
    targets[nxt_block] = -1
    return targets

def build_future_set_targets(tokens: np.ndarray, sep_id=SEP_ID, pad_id=PAD_ID, eow_id=EOW_ID, special_floor=6):
    """
    Build multiset supervision for the fast parallel loss.
    For each time t within a visit:
      - remaining_indices[t] lists the (unordered) codes still to come before the next SEP (padded with -1).
      - need_sep_mask[t] is True when no codes remain (so the correct next token is SEP).
    Assumes no duplicates per visit and that every visit ends with SEP.
    """
    T = tokens.shape[0]
    real = np.where(tokens != pad_id)[0]
    if real.size == 0:
        return np.full((T, 1), -1, np.int64), np.zeros(T, bool)

    last = real[-1]
    seps = np.where(tokens[:last+1] == sep_id)[0]
    starts = np.concatenate(([0], seps[:-1] + 1))
    ends   = seps + 1

    per_step = [None] * (last + 1)
    maxK = 0
    for s, e in zip(starts, ends):
        v = tokens[s:e]
        is_code = (v >= special_floor) & (v != eow_id)
        codes = v[is_code]
        if codes.size == 0:
            for t in range(s, e): per_step[t] = []
            continue
        rem = set(codes.tolist())
        for t in range(s, e):
            if t < e - 1:
                tok = v[t - s]
                if tok >= special_floor and tok != eow_id:
                    rem.discard(tok)
            per_step[t] = list(rem)
            if len(rem) > maxK: maxK = len(rem)

    K = maxK if maxK > 0 else 1
    remaining = np.full((T, K), -1, np.int64)
    need_sep = np.zeros(T, bool)
    for t in range(last + 1):
        s = per_step[t]
        if s is None: continue
        if len(s) == 0:
            need_sep[t] = True
        else:
            remaining[t, :len(s)] = np.asarray(s, np.int64)

    return remaining, need_sep

def build_future_set_targets_with_boundary(tokens: np.ndarray,
                                           sep_id=SEP_ID, pad_id=PAD_ID, eow_id=EOW_ID, special_floor=6):
    """
    Multiset targets with boundary supervision:

    - For code rows inside a visit: target = unordered set of codes *after* t up to the visit SEP.
      If empty -> need_sep[t] = True (meaning "next token should be SEP").
    - For SEP rows: if the next visit is fully visible in-window, target = FULL SET of the next visit's codes.
      If the next visit is not fully visible, we skip supervising this SEP row.

    Returns:
        remaining (T, Kmax) int64 with -1 padding,
        need_sep (T,) bool,
        supervise_mask (T,) bool  # True exactly where we supervise a loss (code rows + eligible SEP rows)
    """
    T = tokens.shape[0]
    real = np.where(tokens != pad_id)[0]
    if real.size == 0:
        return np.full((T, 1), -1, np.int64), np.zeros(T, bool), np.zeros(T, bool)

    last = real[-1]
    seps   = np.where(tokens[:last+1] == sep_id)[0]
    starts = np.concatenate(([0], seps[:-1] + 1))
    ends   = seps + 1  # visit spans [s, e) and SEP is at e-1

    per_step = [None] * (last + 1)
    supervise_mask = np.zeros(last + 1, dtype=bool)
    maxK = 0

    for s, e in zip(starts, ends):
        v = tokens[s:e]
        is_code = (v >= special_floor) & (v != eow_id) & (v != sep_id)
        codes = v[is_code]

        # ---- code rows: remaining set within current visit ----
        rem = set(codes.tolist())
        for t in range(s, e - 1):  # exclude SEP row for intra-visit logic
            tok = v[t - s]
            if tok >= special_floor and tok != eow_id and tok != sep_id:
                rem.discard(tok)
            per_step[t] = list(rem)
            supervise_mask[t] = True
            if len(rem) > maxK: maxK = len(rem)

        # ---- SEP row: supervise next visit's FULL set (if fully visible) ----
        sep_idx = e - 1
        next_start = e
        per_step[sep_idx] = None  # default: don't supervise this SEP

        if next_start <= last:
            rel = np.where(tokens[next_start:last+1] == sep_id)[0]
            if rel.size > 0:
                e_next = next_start + rel[0] + 1  # include next visit's SEP
                nv = tokens[next_start:e_next]
                is_code_nv = (nv >= special_floor) & (nv != eow_id) & (nv != sep_id)
                nv_codes = nv[is_code_nv]
                if nv_codes.size > 0:
                    uniq = list(set(nv_codes.tolist()))
                    per_step[sep_idx] = uniq
                    supervise_mask[sep_idx] = True
                    if len(uniq) > maxK: maxK = len(uniq)

    K = maxK if maxK > 0 else 1
    remaining = np.full((T, K), -1, np.int64)
    need_sep  = np.zeros(T, bool)

    for t in range(last + 1):
        s_list = per_step[t]
        if s_list is None:
            continue
        if len(s_list) == 0:
            # only code rows can have empty (meaning: next should be SEP)
            need_sep[t] = True
        else:
            remaining[t, :len(s_list)] = np.asarray(s_list, np.int64)

    sup = np.zeros(T, dtype=bool)
    sup[:last+1] = supervise_mask
    return remaining, need_sep, sup


# =========================
# UnifiedSeqEHRDataset
# =========================
class UnifiedSeqEHRDataset(Dataset):
    """
    Produces (token_arr, days_embed_arr, targets, patient_ids) for NTP.
    Optional multiset supervision emits:
      - remaining_indices: (T, Kmax) long with -1 padding
      - need_sep_mask: (T,) bool
    - Per-visit serialization with category-aware ordering and optional within-group shuffle.
    - SEP appended after every visit; SEP's day = next visit's day (or same if last).
    - Windows padded to max_length; targets ignore PAD/EOW; SEP is learned.
    - 'days_embed_arr' holds integer days (for RoPE/additive sinusoid).
    - 'patient_ids' is a segment indicator for stitched multiwin (1..K; 0 on pad), else zeros.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        max_length: int = 512,
        model: str = 'GPT',
        folder: str = "./data/",
        age=None,
        train: bool = False,
        agg_labels: bool = False,     # no-op for NTP
        temporal_decay=None,          # no-op for NTP
        multiwin: bool = False,
        header_file=None,
        shuffle_within_groups: bool = True,
        build_multiset_targets: bool = False,
        max_eval_windows_per_patient: int | None = 64
    ):
        if header_file is None:
            header_file = os.path.join(folder, "data_files", "filtered_headers_token_merge.json")
        self.shuffle_within_groups = shuffle_within_groups
        self.build_multiset_targets = build_multiset_targets

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

        print("Open db files:")
        self.db_dict = {part: shelve.open(self.folder + '/processed_part_' + str(part) + '.shelve') for part in tqdm(range(40))}

        # Original space -> masked space
        remove_path = os.path.join(self.folder, "data_files", "delete_inds.npy")
        self.remove_inds = np.load(remove_path)
        total_prev_tokens = 57735
        self.keep_mask = np.ones(total_prev_tokens, dtype=bool)
        self.keep_mask[self.remove_inds] = False

        # Day-of-visit column in MASKED space (per your setup)
        self.day_col = 35845

        # Masked-space headers aligned with a[:, keep_mask]
        with open(header_file, "rb") as f:
            headers = json.load(f)
        self.headers = headers  # masked-space headers

        # Build masked-col → category table (vectorized)
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

        # Persistent RNG
        self.rng = np.random.default_rng()

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
        age  = masked_cols[cats == _CAT_AGE]
        demo = masked_cols[cats == _CAT_DEMO]
        diag = masked_cols[cats == _CAT_DIAG]
        lab  = masked_cols[cats == _CAT_LAB]
        med  = masked_cols[cats == _CAT_MED]

        if self.shuffle_within_groups:
            if diag.size > 1: diag = self.rng.permutation(diag)
            if lab.size  > 1: lab  = self.rng.permutation(lab)
            if med.size  > 1: med  = self.rng.permutation(med)

        return np.concatenate([age, demo, diag, lab, med], axis=0)

    # -------- Per-patient serialization (masked space, CSR-indptr fast path)
    def _serialize_patient(self, a_masked, days_data, rng: np.random.Generator):
        """
        Returns:
          token_list_final: np.int64 [N_tokens]
          pos_embed_days_final: np.int64 [N_tokens]
        """
        n_visits = a_masked.shape[0]
        token_list = []
        day_list   = []

        indptr = a_masked.indptr
        indices = a_masked.indices

        for r in range(n_visits):
            s, e = indptr[r], indptr[r + 1]
            cols = indices[s:e]  # masked col indices present this visit (view, no copy)

            # Empty (or day-only) visit → still place SEP with next-visit day semantics
            if cols.size == 0 or (cols.size == 1 and cols[0] == self.day_col):
                next_day = days_data[r+1] if (r+1) < n_visits else days_data[r]
                token_list.append(SEP_ID)
                day_list.append(next_day)
                continue

            ordered_cols = self._order_visit_cols(cols, rng)
            if ordered_cols.size > 0:
                ordered_tok_ids = ordered_cols + 6  # SHIFT=6 for specials
                token_list.extend(ordered_tok_ids.tolist())
                day_list.extend([days_data[r]] * ordered_tok_ids.size)

            # end-of-visit SEP uses next-visit day (or current if last)
            next_day = days_data[r+1] if (r+1) < n_visits else days_data[r]
            token_list.append(SEP_ID)
            day_list.append(next_day)

        token_list_final     = np.asarray(token_list, dtype=np.int64)
        pos_embed_days_final = np.asarray(day_list,   dtype=np.int64)
        return token_list_final, pos_embed_days_final

    # -------- Sample a training window from serialized stream (<= max_length//2)
    def _sample_train_window_from_serialized(self, token_list_final, pos_embed_days_final):
        # compute visit boundaries from days (changes in day index)
        change_indices = np.where(np.diff(pos_embed_days_final) != 0)[0] + 1
        starting_indices = np.concatenate(([0], change_indices))
        starting_indices[1:] = starting_indices[1:] + 1

        if starting_indices.size <= 1:
            # degenerate: emit [SEP, EOW]
            tokens = np.full(self.max_length, PAD_ID, np.int64)
            days   = np.zeros(self.max_length, np.int64)
            tokens[:2] = [SEP_ID, EOW_ID]
            targets = build_ntp_targets(tokens, pad_id=PAD_ID, eow_id=EOW_ID)
            item = {
                "token_arr": tokens,
                "days_embed_arr": days,
                "targets": targets,
                "patient_ids": np.zeros_like(tokens, np.int64)
            }
            if self.build_multiset_targets:
                rem, needsep, sup = build_future_set_targets_with_boundary(tokens)
                item["remaining_indices"] = rem
                item["need_sep_mask"] = needsep
                item["supervise_mask"] = sup
            return item

        # sample a starting anchor (visit boundary index)
        start_visit_idx = np.random.randint(0, starting_indices.shape[0] - 1)
        start = starting_indices[start_visit_idx]

        # greedily expand up to <= max_length//2 tokens
        cur_end_visit = start_visit_idx + 1
        max_tokens = self.max_length // 2
        while cur_end_visit < starting_indices.shape[0]:
            tentative_end = starting_indices[cur_end_visit]
            if tentative_end - start > max_tokens:
                cur_end_visit -= 1
                break
            cur_end_visit += 1
        cur_end_visit = min(cur_end_visit, starting_indices.shape[0] - 1)

        if cur_end_visit == start_visit_idx:
            # take a prefix within one visit and ensure SEP at end
            next_visit_start = starting_indices[start_visit_idx + 1]
            span_tokens = token_list_final[start:next_visit_start]
            span_days   = pos_embed_days_final[start:next_visit_start]
            if span_tokens.size > (max_tokens - 1):
                cap = max_tokens - 1
                span_tokens = np.concatenate([span_tokens[:cap], [SEP_ID]])
                span_days   = np.concatenate([span_days[:cap],   [span_days[-1]]])
        else:
            end = starting_indices[cur_end_visit] if cur_end_visit < len(starting_indices) else len(token_list_final)
            span_tokens = token_list_final[start:end]
            span_days   = pos_embed_days_final[start:end]

        # normalize days to start at 0
        span_days = span_days - span_days[0]
        tokens = _pad_to_length_1d(span_tokens.astype(np.int64), self.max_length, PAD_ID)
        days   = _pad_to_length_1d(span_days.astype(np.int64),   self.max_length, 0)

        # only add EOW when NOT stitching multiple windows
        nz = np.where(tokens != PAD_ID)[0]
        if (not self.multiwin) and nz.size and tokens[nz[-1]] != EOW_ID and (nz[-1] + 1) < self.max_length:
            tokens[nz[-1] + 1] = EOW_ID


        # optional EOW right after last real if room and not already there
        # if nz.size and tokens[nz[-1]] != EOW_ID and (nz[-1] + 1) < self.max_length:
        #     tokens[nz[-1] + 1] = EOW_ID

        targets = build_ntp_targets(tokens, pad_id=PAD_ID)
        pids    = np.zeros_like(tokens, np.int64)

        item = {"token_arr": tokens, "days_embed_arr": days, "targets": targets, "patient_ids": pids}
        if self.build_multiset_targets:
            rem, needsep, sup = build_future_set_targets_with_boundary(tokens)
            item["remaining_indices"] = rem
            item["need_sep_mask"] = needsep
            item["supervise_mask"] = sup

        return item

    # -------- Combine multiple windows from multiple patients (multiwin=True)
    # def _combine_windows(self, windows):
    #     """
    #     Concatenate multiple short windows with a single EOW between them.

    #     Each window dict contains:
    #       token_arr (T_i), days_embed_arr (T_i), targets (T_i), patient_ids (T_i),
    #       and optionally remaining_indices (T_i, K_i), need_sep_mask (T_i,)

    #     Returns a single fixed-length sample with stitched & padded fields.
    #     """
    #     T = self.max_length

    #     # Determine Kmax if multiset targets are present
    #     Kmax = 1
    #     has_ms = self.build_multiset_targets and all(("remaining_indices" in w and "need_sep_mask" in w) for w in windows)
    #     if has_ms:
    #         for w in windows:
    #             ri = w["remaining_indices"]
    #             if isinstance(ri, np.ndarray) and ri.ndim == 2:
    #                 Kmax = max(Kmax, ri.shape[1])

    #     tokens = np.full(T, PAD_ID, dtype=np.int64)
    #     days   = np.zeros(T, dtype=np.int64)
    #     pids   = np.zeros(T, dtype=np.int64)

    #     if has_ms:
    #         rem    = np.full((T, Kmax), -1, dtype=np.int64)
    #         needsep= np.zeros(T, dtype=bool)

    #     pos = 0
    #     day_offset = 0
    #     for i, w in enumerate(windows):
    #         t = w['token_arr']; d_raw = w['days_embed_arr']; pid = w['patient_ids']
    #         # trim trailing PADs for this segment
    #         real_len = int((t != PAD_ID).sum())
    #         t = t[:real_len]
    #         d = d_raw[:real_len] + day_offset
    #         pid = pid[:real_len]

    #         seg_len = min(real_len, max(0, T - pos))
    #         if seg_len <= 0: break

    #         tokens[pos:pos+seg_len] = t[:seg_len]
    #         days[pos:pos+seg_len]   = d[:seg_len]
    #         pids[pos:pos+seg_len]   = (i + 1)

    #         if has_ms:
    #             ri = w['remaining_indices'][:real_len]
    #             ns = w['need_sep_mask'][:real_len]
    #             curK = ri.shape[1]
    #             if curK < Kmax:
    #                 pad_block = -np.ones((seg_len, Kmax - curK), dtype=np.int64)
    #                 rem[pos:pos+seg_len] = np.concatenate([ri[:seg_len], pad_block], axis=1)
    #             else:
    #                 rem[pos:pos+seg_len] = ri[:seg_len, :Kmax]
    #             needsep[pos:pos+seg_len] = ns[:seg_len]

    #         pos += seg_len

    #         # insert EOW between segments if room
    #         if i < len(windows) - 1 and pos < T:
    #             tokens[pos] = EOW_ID
    #             days[pos]   = days[pos - 1]
    #             pids[pos]   = (i + 1)
    #             # rem[pos] stays -1 row; needsep[pos] stays False
    #             pos += 1

    #         if real_len > 0:
    #             last_raw = int(d_raw[min(seg_len, real_len) - 1]) if seg_len > 0 else 0
    #             day_offset += last_raw + 1000
    #         if pos >= T: break

    #     targets = build_ntp_targets(tokens, pad_id=PAD_ID)
    #     out = {'token_arr': tokens, 'days_embed_arr': days, 'targets': targets, 'patient_ids': pids}
    #     if has_ms:
    #         out['remaining_indices'] = rem
    #         out['need_sep_mask'] = needsep
    #     return out

    def _combine_windows(self, windows):
        """
        Concatenate multiple short windows with a single EOW between them.

        Each window dict contains:
          token_arr (T_i), days_embed_arr (T_i), targets (T_i), patient_ids (T_i),
          and optionally remaining_indices (T_i, K_i), need_sep_mask (T_i,), supervise_mask (T_i,)

        Returns a single fixed-length sample with stitched & padded fields.
        """
        T = self.max_length

        # Determine Kmax if multiset targets are present
        Kmax = 1
        has_ms = self.build_multiset_targets and all(("remaining_indices" in w and "need_sep_mask" in w) for w in windows)
        has_sup = has_ms and all(("supervise_mask" in w) for w in windows)
        if has_ms:
            for w in windows:
                ri = w["remaining_indices"]
                if isinstance(ri, np.ndarray) and ri.ndim == 2:
                    Kmax = max(Kmax, ri.shape[1])

        tokens = np.full(T, PAD_ID, dtype=np.int64)
        days   = np.zeros(T, dtype=np.int64)
        pids   = np.zeros(T, dtype=np.int64)

        if has_ms:
            rem    = np.full((T, Kmax), -1, dtype=np.int64)
            needsep= np.zeros(T, dtype=bool)
            sup    = np.zeros(T, dtype=bool) if has_sup else None

        pos = 0
        day_offset = 0
        for i, w in enumerate(windows):
            t = w['token_arr']; d_raw = w['days_embed_arr']; pid = w['patient_ids']
            # trim trailing PADs for this segment
            real_len = int((t != PAD_ID).sum())
            t = t[:real_len]
            d = d_raw[:real_len] + day_offset
            pid = pid[:real_len]

            seg_len = min(real_len, max(0, T - pos))
            if seg_len <= 0: break

            tokens[pos:pos+seg_len] = t[:seg_len]
            days[pos:pos+seg_len]   = d[:seg_len]
            pids[pos:pos+seg_len]   = (i + 1)

            if has_ms:
                ri = w['remaining_indices'][:real_len]
                ns = w['need_sep_mask'][:real_len]
                curK = ri.shape[1]
                if curK < Kmax:
                    pad_block = -np.ones((seg_len, Kmax - curK), dtype=np.int64)
                    rem[pos:pos+seg_len] = np.concatenate([ri[:seg_len], pad_block], axis=1)
                else:
                    rem[pos:pos+seg_len] = ri[:seg_len, :Kmax]
                needsep[pos:pos+seg_len] = ns[:seg_len]
                if has_sup:
                    sm = w['supervise_mask'][:real_len]
                    sup[pos:pos+seg_len] = sm[:seg_len]

            pos += seg_len

            # insert EOW between segments if room
            if i < len(windows) - 1 and pos < T:
                tokens[pos] = EOW_ID
                days[pos]   = days[pos - 1]
                pids[pos]   = (i + 1)
                # rem[pos] stays -1 row; needsep[pos] stays False; sup[pos] stays False
                pos += 1

            if real_len > 0:
                last_raw = int(d_raw[min(seg_len, real_len) - 1]) if seg_len > 0 else 0
                day_offset += last_raw + 1000
            if pos >= T: break

        targets = build_ntp_targets(tokens, pad_id=PAD_ID)
        out = {'token_arr': tokens, 'days_embed_arr': days, 'targets': targets, 'patient_ids': pids}
        if has_ms:
            out['remaining_indices'] = rem
            out['need_sep_mask'] = needsep
            if has_sup:
                out['supervise_mask'] = sup
        return out

    def __getitem__(self, idx):
        loc = self.df.iloc[idx]
        part = int(loc["part"])
        patient_id = str(loc["patient_id"])

        # Load masked-space matrix
        a = self.db_dict[part][patient_id]   # CSR in original space
        a = a[:, self.keep_mask]             # -> masked space (headers align with this)

        if self.model == 'GPT' and self.train:
            a = self._remove_random_rows_csr(a, percentage=0.1)

        # days from masked day column (dense extract is straightforward)
        days_data = a[:, self.day_col].toarray().astype("int").ravel()

        # Serialize per-visit (fast CSR path + vectorized grouping)
        rng = self.rng if self.shuffle_within_groups else np.random.default_rng(0)
        token_list_final, pos_embed_days_final = self._serialize_patient(a, days_data, rng)

        # Build visit starts for eval windowing
        change_indices   = np.where(np.diff(pos_embed_days_final) != 0)[0] + 1
        starting_indices = np.concatenate(([0], change_indices))
        starting_indices[1:] = starting_indices[1:] + 1

        if (not self.train) or (not self.multiwin):
            if self.train:
                window = self._sample_train_window_from_serialized(token_list_final, pos_embed_days_final)
                return [window]  # list so collate can flatten
            else:
                # evaluation: split into consecutive windows up to max_length
                items = []
                current_start = 0
                end_ind = starting_indices.shape[0] - 1
                while current_start < end_ind:
                    current_end = current_start
                    accumulated = 0
                    while current_end < end_ind:
                        if current_end + 1 >= len(starting_indices):
                            next_len = len(token_list_final) - starting_indices[current_end]
                        else:
                            next_len = starting_indices[current_end + 1] - starting_indices[current_end]
                        if accumulated + next_len > self.max_length: break
                        accumulated += next_len
                        current_end += 1

                    if current_end == current_start:
                        start = starting_indices[current_start]
                        item_token_arr = token_list_final[start:start + self.max_length]
                        item_days_embed_arr = pos_embed_days_final[start:start + self.max_length]
                        item_days_embed_arr = item_days_embed_arr - item_days_embed_arr[0]
                        tokens = _pad_to_length_1d(item_token_arr.astype(np.int64), self.max_length, PAD_ID)
                        days = _pad_to_length_1d(item_days_embed_arr.astype(np.int64), self.max_length, 0)
                    else:
                        start = starting_indices[current_start]
                        end = starting_indices[current_end] if current_end < len(starting_indices) else len(token_list_final)
                        tokens_raw = token_list_final[start:end]
                        days_raw = pos_embed_days_final[start:end] - pos_embed_days_final[start]
                        tokens = _pad_to_length_1d(tokens_raw.astype(np.int64), self.max_length, PAD_ID)
                        days = _pad_to_length_1d(days_raw.astype(np.int64), self.max_length, 0)

                    # optional EOS
                    last_real = np.where(tokens != PAD_ID)[0]
                    if last_real.size > 0:
                        last_idx = last_real[-1]
                        if tokens[last_idx] != EOW_ID and last_idx + 1 < self.max_length:
                            tokens[last_idx + 1] = EOW_ID

                    targets = build_ntp_targets(tokens, pad_id=PAD_ID)
                    pids = np.zeros_like(tokens, dtype=np.int64)

                    item = {'token_arr': tokens, 'days_embed_arr': days, 'targets': targets, 'patient_ids': pids}
                    if self.build_multiset_targets:
                        rem, needsep, sup = build_future_set_targets_with_boundary(tokens)
                        item["remaining_indices"] = rem
                        item["need_sep_mask"] = needsep
                        item["supervise_mask"] = sup


                    items.append(item)
                    current_start = current_end
                    if (self.max_eval_windows_per_patient is not None and len(items) >= self.max_eval_windows_per_patient):
                        break
                return items
        else:
            # multiwin training: stitch multiple short windows (possibly different patients)
            windows = []
            total_len = 0
            # first window from current patient
            win = self._sample_train_window_from_serialized(token_list_final, pos_embed_days_final)
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
                tl, dl = self._serialize_patient(a_i, days_i, rng)

                win_i = self._sample_train_window_from_serialized(tl, dl)
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
