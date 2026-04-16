import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4


class SeqCLSDatasetEHRSHOTBERT(Dataset):
    """
    BERT-style dataset adapter for EHRSHOT inference pickle files.

    Expected input file format:
      List[{
        'subject_id': int,
        'boolean_value': int/bool,
        'visits': List[{'days': float/int, 'tokens': List[int]}]
      }]

    Output keys mirror cls_dataloader.SeqCLSDataset:
      - input_ids
      - attention_mask
      - token_type_ids
      - position_ids
      - labels
      - subject_id (optional)
    """

    def __init__(
        self,
        data_path: str,
        max_length: int = 512,
        min_num_visits: int = 1,
        random_window: bool = False,
        token_offset: int = 5,
        vocab_size: int = 57740,
        max_position_embeddings: int = 4000,
        type_vocab_size: int = 4000,
        return_subject_id: bool = True,
    ):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.max_length = max_length
        self.min_num_visits = min_num_visits
        self.random_window = random_window
        self.token_offset = token_offset
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.return_subject_id = return_subject_id

        print(f"Loaded {len(self.data)} examples from {data_path}")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _sanitize_visits(visits: List[dict]) -> List[dict]:
        clean = []
        for v in visits:
            toks = v.get("tokens", [])
            if toks is None:
                continue
            toks = [int(t) for t in toks if t is not None]
            if len(toks) == 0:
                continue
            clean.append({"days": int(v["days"]), "tokens": toks})
        return clean

    def _flatten_visits(self, visits: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flatten visits into token/day arrays with SEP between visits.
        SEP day follows cls_dataloader behavior (next visit day).
        Returns:
          flat_tokens, flat_days, visit_start_indices
        """
        flat_tokens = []
        flat_days = []
        visit_starts = []

        for i, visit in enumerate(visits):
            day = int(visit["days"])
            toks = []
            for t in visit["tokens"]:
                tid = int(t) + self.token_offset
                if tid < 0 or tid >= self.vocab_size:
                    tid = UNK_ID
                toks.append(tid)

            visit_starts.append(len(flat_tokens))
            flat_tokens.extend(toks)
            flat_days.extend([day] * len(toks))

            if i < len(visits) - 1:
                next_day = int(visits[i + 1]["days"])
                flat_tokens.append(SEP_ID)
                flat_days.append(next_day)

        return np.asarray(flat_tokens, dtype=np.int64), np.asarray(flat_days, dtype=np.int64), np.asarray(visit_starts, dtype=np.int64)

    def _select_window(
        self, flat_tokens: np.ndarray, flat_days: np.ndarray, visit_starts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select a visit-aligned window with at most max_length pre-CLS tokens.
        Mirrors eval-style deterministic behavior when random_window=False.
        """
        if len(flat_tokens) == 0:
            return np.asarray([SEP_ID], dtype=np.int64), np.asarray([0], dtype=np.int64)

        n_visits = len(visit_starts)
        if n_visits == 0:
            toks = flat_tokens[: self.max_length]
            days = flat_days[: self.max_length]
            return toks, days

        if self.random_window:
            high_val = n_visits - self.min_num_visits + 1
            start_visit = random.randint(0, high_val - 1) if high_val > 0 else 0
        else:
            start_visit = 0

        start_idx = int(visit_starts[start_visit])
        end_idx = start_idx

        # Expand by whole visits while staying under max_length tokens.
        for j in range(start_visit, n_visits):
            candidate_end = (
                int(visit_starts[j + 1]) if (j + 1) < n_visits else len(flat_tokens)
            )
            if candidate_end - start_idx <= self.max_length:
                end_idx = candidate_end
            else:
                break

        # Fallback for very dense single-visit segments.
        if end_idx <= start_idx:
            end_idx = min(start_idx + self.max_length, len(flat_tokens))

        toks = flat_tokens[start_idx:end_idx]
        days = flat_days[start_idx:end_idx]

        # Final safety crop.
        if len(toks) > self.max_length:
            toks = toks[: self.max_length]
            days = days[: self.max_length]

        return toks, days

    def __getitem__(self, idx):
        example = self.data[idx]
        visits = self._sanitize_visits(example.get("visits", []))
        label = int(example.get("boolean_value", 0))
        subject_id = int(example.get("subject_id", -1))

        if len(visits) == 0:
            token_arr = np.asarray([SEP_ID], dtype=np.int64)
            days_embed_arr = np.asarray([0], dtype=np.int64)
        else:
            flat_tokens, flat_days, visit_starts = self._flatten_visits(visits)
            token_arr, days_embed_arr = self._select_window(flat_tokens, flat_days, visit_starts)

            if len(days_embed_arr) > 0:
                days_embed_arr = days_embed_arr - days_embed_arr[0]

        encounter_pos_embed = np.cumsum(np.diff(days_embed_arr, prepend=0) > 0).astype(np.int64)
        curr_size = len(token_arr)

        # Match cls_dataloader tensor construction (prepend CLS and shift embeddings).
        if len(days_embed_arr) > 0:
            days_embed_arr = np.concatenate(([0], days_embed_arr[:-1]))
            encounter_pos_embed = np.concatenate(([0], encounter_pos_embed[:-1]))
        else:
            days_embed_arr = np.asarray([0], dtype=np.int64)
            encounter_pos_embed = np.asarray([0], dtype=np.int64)

        # Keep indices in embedding-table bounds.
        # position_ids/token_type_ids are later shifted by +1, with 0 reserved for PAD.
        max_pos_raw = max(0, self.max_position_embeddings - 2)
        max_type_raw = max(0, self.type_vocab_size - 2)
        days_embed_arr = np.clip(days_embed_arr, 0, max_pos_raw)
        encounter_pos_embed = np.clip(encounter_pos_embed, 0, max_type_raw)

        pad_len = max(0, self.max_length - curr_size)

        attention_mask = np.concatenate(
            ([1], np.pad(np.ones((curr_size), dtype=np.int64), (0, pad_len), mode="constant", constant_values=PAD_ID))
        )
        input_ids = np.concatenate(
            ([CLS_ID], np.pad(token_arr, (0, pad_len), mode="constant", constant_values=PAD_ID))
        )
        position_ids = np.concatenate(
            ([PAD_ID], np.pad(days_embed_arr + 1, (0, pad_len), mode="constant", constant_values=PAD_ID))
        )
        token_type_ids = np.concatenate(
            ([PAD_ID], np.pad(encounter_pos_embed + 1, (0, pad_len), mode="constant", constant_values=PAD_ID))
        )

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "labels": torch.tensor([label], dtype=torch.long),
        }

        if self.return_subject_id:
            item["subject_id"] = torch.tensor([subject_id], dtype=torch.long)

        return item
