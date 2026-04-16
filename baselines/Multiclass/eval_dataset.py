# condition_helper.py (or wherever SeqCLSDataset lives)

import os
import json
from glob import glob
import shelve
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

GPTCONFIG_PARAMS = [
    'block_size',
    'vocab_size',
    'n_layer',
    'n_head',
    'n_embd',
    'dropout',
    'bias',
    'gamma',
    'alpha',
    'loss_type',
    'pos_weight',
    'loss_weighting',
    'loss_top_perc',
    'rotary',
    'use_xpos'
]


def _strip_trailing_sep_dup(tokens_np, days_np, sep_id=1):
    # tokens_np: 1D np.int64, unpadded tail (non-PAD region)
    # days_np  : 1D (direct) or 2D [H, T'] (intermediate), same length T' as tokens_np
    if tokens_np.size >= 2 and tokens_np[-1] == sep_id and tokens_np[-2] == sep_id:
        tokens_np = tokens_np[:-1]
        if days_np.ndim == 1:
            days_np = days_np[:-1]
        else:
            days_np = days_np[:, :-1]
    return tokens_np, days_np


# Special token IDs
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4
EOW_ID = 5
SHIFT  = 6  # base code index -> token id offset

# Category tags (masked-space headers)
_CAT_OTHER = 0
_CAT_AGE   = 1
_CAT_DEMO  = 2
_CAT_DIAG  = 3
_CAT_LAB   = 4
_CAT_MED   = 5

class SeqCLSDataset(Dataset):
    """
    Visit-aware onset evaluation dataset:
      - Reads shelves in masked space
      - Per-visit ordering (Age → Demo → Dx → Lab → Med), optional shuffle within groups
      - Appends SEP at end of each visit; SEP's day = next-visit day (or same if last)
      - Builds attention bias and pred_mask targeting the *last SEP* token
      - Supports:
          * approach="direct": single time_horizon
          * approach="intermediate_pred": multiple horizons packed in days tensor
    """
    def __init__(
        self,
        mode="val",
        max_length=512,
        folder="./data/",
        header_file=None,
        multiwin=False,
        time_horizon=365*4,
        approach="direct",           # "direct" | "intermediate_pred"
        num_patients=-1,
        horizon=365,
        shuffle_within_groups=True
    ):
        if header_file is None:
            header_file = os.path.join(folder, "data_files", "filtered_headers_token_merge.json")
        self.max_length = max_length
        self.folder = folder
        self.mode = mode
        self.time_horizon = time_horizon
        self.horizon = horizon
        self.approach = approach
        self.multiwin = multiwin
        self.shuffle_within_groups = shuffle_within_groups

        # Open shelves
        files = sorted(glob(os.path.join(folder, f"data_{mode}*.shelve.dat")))
        if not files:
            raise FileNotFoundError(f"No shelve dat files for mode={mode} in {folder}")
        print("Open db files:\n")
        self.db_dict = {}
        for i, file in tqdm(enumerate(files), total=len(files)):
            self.db_dict[i] = shelve.open(file.strip(".dat"))

        # Index all patient keys
        self.data_list = []
        for k, v in self.db_dict.items():
            for key in v.keys():
                self.data_list.append((k, key))
        if num_patients != -1:
            rng = np.random.default_rng()
            rng.shuffle(self.data_list)
            self.data_list = self.data_list[:num_patients]

        # Mask from v4 (masked-space)
        remove_inds = np.load(os.path.join(folder, "data_files", "delete_inds.npy"))
        total_prev_tokens = 57735
        self.keep_mask = np.ones(total_prev_tokens, dtype=bool)
        self.keep_mask[remove_inds] = False

        # Day-of-visit column in masked space
        self.day_col = 35845

        # Headers (masked-space), build category table
        with open(header_file, "rb") as f:
            headers = json.load(f)
        self.headers = headers

        def _cat_from_header(txt: str) -> int:
            t = txt or ""
            if t.startswith("demographics_age"): return _CAT_AGE
            if t.startswith("demographics_"):    return _CAT_DEMO
            if t.startswith("Diagnosis:"):       return _CAT_DIAG
            if t.startswith("Lab:"):             return _CAT_LAB
            if t.startswith("Medication:"):      return _CAT_MED
            return _CAT_OTHER

        self.col_category_masked = np.fromiter(
            (_cat_from_header(t) for t in self.headers),
            dtype=np.int8,
            count=len(self.headers)
        )

        # Persistent RNG for within-group shuffle
        self.rng = np.random.default_rng()

    # ---------- Helpers ----------

    def _order_visit_cols(self, masked_cols: np.ndarray) -> np.ndarray:
        """Order masked column ids inside one visit by category; optional shuffle within Dx/Lab/Med."""
        if masked_cols.size == 0:
            return masked_cols
        masked_cols = masked_cols[masked_cols != self.day_col]
        if masked_cols.size == 0:
            return masked_cols

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

    def _serialize_patient(self, a_masked, days_data):
        """
        Returns:
          token_list_final: np.int64 [N_tokens]
          days_final:       np.int64 [N_tokens]
        Rules:
          - Codes in a visit: ordered by category; ids shifted by +6
          - SEP appended after every visit; SEP day = next-visit day (or same if last)
        """
        n_visits = a_masked.shape[0]
        token_list, day_list = [], []

        indptr  = a_masked.indptr
        indices = a_masked.indices

        for r in range(n_visits):
            s, e = indptr[r], indptr[r + 1]
            cols = indices[s:e]  # masked col ids in visit r

            # Empty/day-only visit => still emit SEP stamped with next-visit day semantics
            if cols.size == 0 or (cols.size == 1 and cols[0] == self.day_col):
                next_day = days_data[r+1] if (r+1) < n_visits else days_data[r]
                token_list.append(SEP_ID)
                day_list.append(int(next_day))
                continue

            ordered_cols = self._order_visit_cols(cols)
            if ordered_cols.size > 0:
                tok_ids = ordered_cols + SHIFT  # shift specials
                token_list.extend(tok_ids.tolist())
                day_list.extend([int(days_data[r])] * tok_ids.size)

            next_day = days_data[r+1] if (r+1) < n_visits else days_data[r]
            token_list.append(SEP_ID)
            day_list.append(int(next_day))

        return np.asarray(token_list, np.int64), np.asarray(day_list, np.int64)
    
    def _pad_matrix(self, matrix, target_size):
        T = matrix.shape[0]
        out = np.zeros((target_size, target_size), dtype=matrix.dtype)
        out[:T, :T] = matrix
        np.fill_diagonal(out, 1)
        return out

    def _create_attention_matrix(self, days):
        # same semantics you used: allow same-day and backward-in-time attention
        T = len(days)
        return (days[:, None] == days) | (days[:, None] >= days)

    # ---------- Main ----------

    # def __getitem__(self, idx):
    #     part_idx, key = self.data_list[idx]
    #     datum = self.db_dict[part_idx][key]   # your shelf row layout: (id, CSR, label, ...)
    #     a = datum[1]                          # CSR (original space)
    #     label = datum[2]

    #     # -> masked space
    #     a = a[:, self.keep_mask]

    #     # dense extract of day col in masked space
    #     days_data = a[:, self.day_col].toarray().astype("int").ravel()
    #     if days_data.size == 0:
    #         # degenerate: no visits — emit a minimal [SEP]
    #         tokens_stream = np.array([SEP_ID], dtype=np.int64)
    #         days_stream   = np.array([0], dtype=np.int64)
    #     else:
    #         # serialize per-visit with ordering & SEP stamping
    #         tokens_stream, days_stream = self._serialize_patient(a, days_data)

    #     # -------- Build the window for onset prediction --------
    #     # We want the trailing history and then place a SEP position we will predict at
    #     # (You previously did: take last max_length-1, then append a SEP)
    #     item_token_arr = tokens_stream[-self.max_length + 1:]
    #     item_days_arr  = days_stream[-self.max_length + 1:]

    #     # Ensure a final SEP position to predict from (append SEP with horizon-adjusted day)
    #     item_token_arr = np.append(item_token_arr, SEP_ID)

    #     if self.approach == "direct":
    #         # The final SEP day = last day + time_horizon (relative)
    #         # Then normalize days to start at 0
    #         final_day = (item_days_arr[-1] if item_days_arr.size > 0 else 0) + self.time_horizon
    #         item_days_arr = np.append(item_days_arr, final_day)
    #         item_days_arr = item_days_arr - item_days_arr[0]

    #         # encounter index (optional, kept from your code if you use it elsewhere)
    #         item_encounter_pos_embed = np.cumsum(np.diff(item_days_arr, prepend=0) > 0)

    #         # attention bias uses "att_days" (you had a slightly different off-by-1 trick before;
    #         # here we simply use the same normalized days so same/equal/causal logic holds.)
    #         att_days = item_days_arr.copy()

    #         # Pad to max_length
    #         cur = item_token_arr.size
    #         tokens = np.pad(item_token_arr, (0, self.max_length - cur), constant_values=PAD_ID)
    #         days   = np.pad(item_days_arr, (0, self.max_length - cur), constant_values=0)
    #         encpos = np.pad(item_encounter_pos_embed, (0, self.max_length - cur), constant_values=0)

    #         pred_mask = np.zeros_like(tokens, dtype=bool)
    #         sep_indices = np.where(tokens == SEP_ID)[0]
    #         if sep_indices.size > 0:
    #             pred_mask[sep_indices[-1]] = True  # last SEP is prediction anchor

    #         pad_mask = (tokens != PAD_ID)

    #         attention_bias = self._create_attention_matrix(att_days)
    #         attention_bias = self._pad_matrix(attention_bias, self.max_length)

    #         item = {
    #             "token_arr": tokens,
    #             "days_embed_arr": days,
    #             "pred_mask": pred_mask,
    #             "pad_mask": pad_mask,
    #             "attention_bias": attention_bias,
    #             "labels": torch.from_numpy(np.array([label])).to(torch.int64),
    #         }
    #         if self.multiwin:
    #             item["patient_ids"] = np.zeros_like(tokens)
    #         return item

    #     elif self.approach == "intermediate_pred":
    #         # Multiple horizons: stack days tensors [num_horizons, T]
    #         horizons = np.arange(self.horizon, self.time_horizon, self.horizon)
    #         horizons = np.append(horizons, self.time_horizon)

    #         base_tokens = item_token_arr
    #         base_days   = item_days_arr

    #         all_days = []
    #         for h in horizons:
    #             d = np.append(base_days, (base_days[-1] if base_days.size > 0 else 0) + h)
    #             d = d - d[0]
    #             all_days.append(d)

    #         # pad tokens once
    #         cur = base_tokens.size
    #         tokens = np.pad(base_tokens, (0, self.max_length - cur), constant_values=PAD_ID)

    #         # stack and pad days for each horizon
    #         days_stack = np.stack(all_days)  # [H, cur]
    #         days_stack = np.pad(days_stack, ((0,0), (0, self.max_length - cur)), constant_values=0)

    #         # attention bias from the first horizon (compatible with your prior code)
    #         att_days = (all_days[0]).copy()
    #         attention_bias = self._create_attention_matrix(att_days)
    #         attention_bias = self._pad_matrix(attention_bias, self.max_length)

    #         pred_mask = np.zeros_like(tokens, dtype=bool)
    #         sep_indices = np.where(tokens == SEP_ID)[0]
    #         if sep_indices.size > 0:
    #             pred_mask[sep_indices[-1]] = True

    #         pad_mask = (tokens != PAD_ID)

    #         item = {
    #             "token_arr": tokens,
    #             "days_embed_arr": days_stack,   # shape [num_horizons, max_length]
    #             "pred_mask": pred_mask,
    #             "pad_mask": pad_mask,
    #             "attention_bias": attention_bias,
    #             "labels": torch.from_numpy(np.array([label])).to(torch.int64),
    #         }
    #         if self.multiwin:
    #             item["patient_ids"] = np.zeros_like(tokens)
    #         return item

    #     else:
    #         raise ValueError(f"Unknown approach: {self.approach}")

    def __getitem__(self, idx):
        # ---------- load one patient ----------
        part_idx, key = self.data_list[idx]
        datum = self.db_dict[part_idx][key]   # layout: (id, CSR, label, ...)
        a = datum[1]                          # CSR (original space)
        label = datum[2]

        # -> masked space
        a = a[:, self.keep_mask]

        # dense extract of day col in masked space
        days_data = a[:, self.day_col].toarray().astype("int").ravel()
        if days_data.size == 0:
            # degenerate: no visits — emit minimal [SEP]
            tokens_stream = np.array([SEP_ID], dtype=np.int64)
            days_stream   = np.array([0], dtype=np.int64)
        else:
            # serialize per-visit with ordering & SEP stamping
            tokens_stream, days_stream = self._serialize_patient(a, days_data)

        # -------- window tail (keep last max_length-1, we'll ensure 1 SEP anchor) --------
        item_token_arr = tokens_stream[-self.max_length + 1:].astype(np.int64, copy=False)
        item_days_arr  = days_stream[-self.max_length + 1:].astype(np.int64, copy=False)

        # ---- helper: strip trailing SEP-SEP, length-aware for 1D or [H, T] days ----
        def _strip_trailing_sep_dup(tokens_np, days_np, sep_id=SEP_ID):
            if tokens_np.size >= 2 and tokens_np[-1] == sep_id and tokens_np[-2] == sep_id:
                tokens_np = tokens_np[:-1]
                if days_np.ndim == 1:   days_np = days_np[:-1]
                else:                   days_np = days_np[:, :-1]
            return tokens_np, days_np

        if self.approach == "direct":
            # ensure exactly one tail SEP (anchor) and set its day = last_day + horizon
            anchor_day = (item_days_arr[-1] if item_days_arr.size > 0 else 0) + self.time_horizon
            if item_token_arr.size > 0 and item_token_arr[-1] == SEP_ID:
                # reuse existing SEP → retime day
                item_days_arr[-1] = anchor_day
            else:
                # no SEP at tail → append one
                item_token_arr = np.append(item_token_arr, SEP_ID)
                item_days_arr  = np.append(item_days_arr,  anchor_day)

            # safety: remove accidental SEP-SEP
            item_token_arr, item_days_arr = _strip_trailing_sep_dup(item_token_arr, item_days_arr)

            # normalize days to start at 0
            item_days_arr = item_days_arr - item_days_arr[0]

            # encounter index (optional; kept for compatibility)
            item_encounter_pos_embed = np.cumsum(np.diff(item_days_arr, prepend=0) > 0)

            # attention bias uses normalized days (same/backward time allowed)
            att_days = item_days_arr.copy()

            # Pad to max_length
            cur = item_token_arr.size
            tokens = np.pad(item_token_arr, (0, self.max_length - cur), constant_values=PAD_ID)
            days   = np.pad(item_days_arr,  (0, self.max_length - cur), constant_values=0)
            encpos = np.pad(item_encounter_pos_embed, (0, self.max_length - cur), constant_values=0)

            pred_mask = np.zeros_like(tokens, dtype=bool)
            sep_indices = np.where(tokens == SEP_ID)[0]
            if sep_indices.size > 0:
                pred_mask[sep_indices[-1]] = True  # last SEP is prediction anchor

            pad_mask = (tokens != PAD_ID)

            attention_bias = self._create_attention_matrix(att_days)
            attention_bias = self._pad_matrix(attention_bias, self.max_length)

            item = {
                "token_arr": tokens,
                "days_embed_arr": days,
                "pred_mask": pred_mask,
                "pad_mask": pad_mask,
                "attention_bias": attention_bias,
                "labels": torch.from_numpy(np.array([label])).to(torch.int64),
            }
            if self.multiwin:
                item["patient_ids"] = np.zeros_like(tokens)
            return item

        elif self.approach == "intermediate_pred":
            # horizons = [h, 2h, ..., time_horizon]; ensure exactly one tail SEP and retime it per horizon
            horizons = np.arange(self.horizon, self.time_horizon, self.horizon)
            horizons = np.append(horizons, self.time_horizon)

            have_sep_tail = (item_token_arr.size > 0 and item_token_arr[-1] == SEP_ID)
            if not have_sep_tail:
                item_token_arr = np.append(item_token_arr, SEP_ID)

            base_days = item_days_arr.copy()
            all_days = []
            for h in horizons:
                d = base_days.copy()
                target_day = (base_days[-1] if base_days.size > 0 else 0) + h
                if have_sep_tail:
                    d[-1] = target_day
                else:
                    d = np.append(d, target_day)
                d = d - d[0]
                all_days.append(d)

            days_stack = np.stack(all_days)  # [H, T']

            # safety: remove accidental SEP-SEP
            item_token_arr, days_stack = _strip_trailing_sep_dup(item_token_arr, days_stack)

            # Pad once for tokens; pad days per horizon along axis=1
            cur = item_token_arr.size
            tokens = np.pad(item_token_arr, (0, self.max_length - cur), constant_values=PAD_ID)
            days_stack = np.pad(days_stack, ((0, 0), (0, self.max_length - cur)), constant_values=0)

            # attention bias from first horizon
            att_days = all_days[0].copy()
            attention_bias = self._create_attention_matrix(att_days)
            attention_bias = self._pad_matrix(attention_bias, self.max_length)

            pred_mask = np.zeros_like(tokens, dtype=bool)
            sep_indices = np.where(tokens == SEP_ID)[0]
            if sep_indices.size > 0:
                pred_mask[sep_indices[-1]] = True

            pad_mask = (tokens != PAD_ID)

            item = {
                "token_arr": tokens,
                "days_embed_arr": days_stack,   # [H, L]
                "pred_mask": pred_mask,
                "pad_mask": pad_mask,
                "attention_bias": attention_bias,
                "labels": torch.from_numpy(np.array([label])).to(torch.int64),
            }
            if self.multiwin:
                item["patient_ids"] = np.zeros_like(tokens)
            return item

        else:
            raise ValueError(f"Unknown approach: {self.approach}")


    def __len__(self):
        return len(self.data_list)

    def _cleanup(self):
        for k in list(self.db_dict.keys()):
            try:
                self.db_dict[k].close()
            except Exception:
                pass
        self.db_dict = {}
