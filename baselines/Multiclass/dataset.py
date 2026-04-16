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

# =========================
# UnifiedSeqEHRDataset
# =========================
class UnifiedSeqEHRDataset(Dataset):
    """
    Produces (token_arr, days_embed_arr, targets, patient_ids) for NTP.
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
        max_eval_windows_per_patient: int | None = 64
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

        # Persistent RNG (cheap & deterministic if desired)
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
            if diag.size > 1: diag = rng.permutation(diag)
            if lab.size  > 1: lab  = rng.permutation(lab)
            if med.size  > 1: med  = rng.permutation(med)

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

            # If empty visit → still place SEP with next-visit day semantics
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
    def _sample_train_window_from_serialized(self, token_list_final, pos_embed_days_final,*, add_eow: bool = False):
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
            return {
                "token_arr": tokens,
                "days_embed_arr": days,
                "targets": targets,
                "patient_ids": np.zeros_like(tokens, np.int64)
            }

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

        # optional EOW right after last real if room and not already there
        if add_eow:
            nz = np.where(tokens != PAD_ID)[0]
            if nz.size and tokens[nz[-1]] != EOW_ID and (nz[-1] + 1) < self.max_length:
                tokens[nz[-1] + 1] = EOW_ID

        targets = build_ntp_targets(tokens, pad_id=PAD_ID)
        pids    = np.zeros_like(tokens, np.int64)
        return {"token_arr": tokens, "days_embed_arr": days, "targets": targets, "patient_ids": pids}

    # -------- Combine multiple windows from multiple patients (multiwin=True)
    def _combine_windows(self, windows):
        T = self.max_length
        tokens = np.full(T, PAD_ID, dtype=np.int64)
        days = np.zeros(T, dtype=np.int64)
        patient_ids = np.zeros(T, dtype=np.int64)

        pos = 0
        day_offset = 0
        for i, w in enumerate(windows):
            t = w['token_arr']; d = w['days_embed_arr']
            real_len = int((t != PAD_ID).sum())
            t = t[:real_len]; d = d[:real_len] + day_offset

            seg_len = min(real_len, max(0, T - pos))
            if seg_len <= 0: break

            tokens[pos:pos + seg_len] = t[:seg_len]
            days[pos:pos + seg_len]   = d[:seg_len]
            patient_ids[pos:pos + seg_len] = (i + 1)
            pos += seg_len

            # insert EOW if room and not last window
            if i < len(windows) - 1 and pos < T:
                if pos == 0 or tokens[pos-1] != EOW_ID:
                    tokens[pos] = EOW_ID
                    days[pos]   = days[pos - 1]
                    patient_ids[pos] = (i + 1)
                    pos += 1

            if real_len > 0:
                day_offset += int(d[seg_len - 1]) + 1000  # keep segments far apart
            if pos >= T: break

        targets = build_ntp_targets(tokens, pad_id=PAD_ID)
        return {'token_arr': tokens, 'days_embed_arr': days, 'targets': targets, 'patient_ids': patient_ids}

    def __getitem__(self, idx):
        loc = self.df.iloc[idx]
        part = int(loc["part"])
        patient_id = str(loc["patient_id"])

        # Load masked-space matrix
        a = self.db_dict[part][patient_id]   # CSR in original space
        a = a[:, self.keep_mask]             # -> masked space (headers align with this)

        if self.model == 'GPT' and self.train:
            a = self._remove_random_rows_csr(a, percentage=0.1)

        # days from masked day column (dense extract is fine & simple)
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
                window = self._sample_train_window_from_serialized(token_list_final, pos_embed_days_final,add_eow=True)
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

                    items.append({
                        'token_arr': tokens,
                        'days_embed_arr': days,
                        'targets': targets,
                        'patient_ids': pids
                    })
                    current_start = current_end
                    if (self.max_eval_windows_per_patient is not None and len(items) >= self.max_eval_windows_per_patient):
                        break

                return items
        else:
            # multiwin training: stitch multiple short windows (possibly different patients)
            windows = []
            total_len = 0
            # first window from current patient
            win = self._sample_train_window_from_serialized(token_list_final, pos_embed_days_final,add_eow=False)
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

                win_i = self._sample_train_window_from_serialized(tl, dl,add_eow=False)
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
