# condition_helper.py

import os, json
from glob import glob
import shelve
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ------------------------------------------------------------------
# Model arg pass-through (unchanged)
# ------------------------------------------------------------------
GPTCONFIG_PARAMS = [
    'block_size','vocab_size','n_layer','n_head','n_embd','dropout','bias',
    'gamma','alpha','loss_type','pos_weight','loss_weighting','loss_top_perc',
    'rotary','use_xpos'
]

# ------------------------------------------------------------------
# Token layout for DURATION setup
# ------------------------------------------------------------------
PAD_ID = 0
EOW_ID = 5               # not used here unless you later stitch windows
DUR_BASE = 6
DURATION_UNITS_DAYS = [180, 60, 30, 14, 7, 3, 1]  # descending “coins”
NUM_DUR = len(DURATION_UNITS_DAYS)                # 7
CODES_START = DUR_BASE + NUM_DUR                  # 13  (all clinical codes start here)

# Category tags (masked-space headers)
_CAT_OTHER = 0
_CAT_AGE   = 1
_CAT_DEMO  = 2
_CAT_DIAG  = 3
_CAT_LAB   = 4
_CAT_MED   = 5

def _quantize_days(delta_days: int) -> int:
    """<=180: exact days; >180: floor to 30-day multiples."""
    d = max(0, int(delta_days))
    if d <= 180: return d
    return (d // 30) * 30

def _emit_duration_tokens_from_days(delta_days: int) -> list[int]:
    """Quantize then greedy-factorize into duration coins (in DAYS)."""
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


class SeqCLSDataset(Dataset):
    """
    Duration-token EHR serialization for the new model/evaluator.

    Emits per patient a single tail window (right-padded to max_length):
      {
        "token_arr": np.int64[max_length],   # clinical codes (masked_col + CODES_START) + duration tokens
        "labels":    torch.int64[1]
      }

    Differences vs legacy:
      - No SEP tokens.
      - No 'days_embed_arr' and no 'patient_ids'.
      - Between consecutive visits, insert duration tokens based on Δdays (coin system above).
      - Clinical codes are shifted by CODES_START (13), i.e., after the duration band.
    """
    def __init__(
        self,
        mode="val",
        max_length=512,
        folder="./data/",
        header_file=None,
        num_patients=-1,
        shuffle_within_groups=True,
        tail_window=True  # if False you could later switch to first or chunked windows
    ):
        if header_file is None:
            header_file = os.path.join(folder, "data_files", "filtered_headers_token_merge.json")
        self.max_length = max_length
        self.folder = folder
        self.mode = mode
        self.shuffle_within_groups = shuffle_within_groups
        self.tail_window = bool(tail_window)

        # Open shelves
        files = sorted(glob(os.path.join(folder, f"data_{mode}*.shelve.dat")))
        if not files:
            raise FileNotFoundError(f"No shelve dat files for mode={mode} in {folder}")
        print("Open db files (duration layout):\n")
        self.db_dict = {i: shelve.open(f.strip(".dat")) for i, f in tqdm(list(enumerate(files)), total=len(files))}

        # Index all patient keys
        self.data_list = []
        for k, v in self.db_dict.items():
            for key in v.keys():
                self.data_list.append((k, key))
        if num_patients != -1:
            rng = np.random.default_rng()
            rng.shuffle(self.data_list)
            self.data_list = self.data_list[:num_patients]

        # Masking to masked-space
        remove_inds = np.load(os.path.join(folder, "data_files", "delete_inds.npy"))
        total_prev_tokens = 57735
        self.keep_mask = np.ones(total_prev_tokens, dtype=bool)
        self.keep_mask[remove_inds] = False

        # Day-of-visit column in masked space
        self.day_col = 35845

        # Headers → category map (for reproducible visit ordering)
        with open(header_file, "r") as f:
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
        self.rng = np.random.default_rng()

    # ------- per-visit helpers -------
    def _order_visit_cols(self, masked_cols: np.ndarray) -> np.ndarray:
        if masked_cols.size == 0: return masked_cols
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

    def _serialize_patient_duration(self, a_masked, days_data):
        """
        Returns: np.int64 token stream
          - visit codes: masked_col + CODES_START (after duration band)
          - durations:   DUR_BASE..DUR_BASE+NUM_DUR-1 from Δdays
        """
        n_visits = a_masked.shape[0]
        token_list = []

        indptr, indices = a_masked.indptr, a_masked.indices
        for r in range(n_visits):
            s, e = indptr[r], indptr[r+1]
            cols = indices[s:e]

            # visit codes
            if cols.size > 0 and not (cols.size == 1 and cols[0] == self.day_col):
                ordered_cols = self._order_visit_cols(cols)
                if ordered_cols.size > 0:
                    tok_ids = ordered_cols + CODES_START
                    token_list.extend(tok_ids.tolist())

            # duration to next visit
            if r + 1 < n_visits:
                delta_days = int(days_data[r+1]) - int(days_data[r])
                token_list.extend(_emit_duration_tokens_from_days(delta_days))

        return np.asarray(token_list, dtype=np.int64)

    # ------- Dataset API -------
    def __getitem__(self, idx):
        part_idx, key = self.data_list[idx]
        datum = self.db_dict[part_idx][key]   # expected layout: (id, CSR, label, ...)
        a = datum[1]                          # CSR matrix in original space
        label = int(datum[2])

        # -> masked space
        a = a[:, self.keep_mask]

        # visit days (dense extract from masked day column)
        days_data = a[:, self.day_col].toarray().astype("int").ravel()

        # print("Days Data", days_data)

        if days_data.size == 0:
            tokens_stream = np.array([], dtype=np.int64)
        else:
            tokens_stream = self._serialize_patient_duration(a, days_data)

        # Tail window (right-padded to max_length)
        T = self.max_length
        out_tokens = np.full(T, PAD_ID, dtype=np.int64)
        if self.tail_window:
            span = tokens_stream[-T:]
        else:
            span = tokens_stream[:T]
        L = min(T, span.size)
        if L > 0:
            out_tokens[:L] = span[:L]

        return {
            "token_arr": out_tokens,
            "labels": torch.tensor([label], dtype=torch.int64)
        }

    def __len__(self):
        return len(self.data_list)

    def _cleanup(self):
        for k in list(self.db_dict.keys()):
            try: self.db_dict[k].close()
            except Exception:
                pass
        self.db_dict = {}
