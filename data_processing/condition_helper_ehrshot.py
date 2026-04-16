import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

# Special token IDs
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4
EOW_ID = 5

GPTCONFIG_PARAMS = [
    'block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd',
    'dropout', 'bias', 'gamma', 'alpha', 'loss_type',
    'pos_weight', 'loss_weighting', 'loss_top_perc', 'rotary', 'use_xpos'
]

def aggregate_logits(logits_masked, mask, method="sum", dim=1):
    """
    Aggregate logits using different methods
    """
    if method == "sum":
        return torch.sigmoid(logits_masked[..., mask].sum(dim=dim))
    elif method == "noisy_or":
        probs = torch.sigmoid(logits_masked[..., mask])
        neg_probs = 1 - probs
        prod_neg_probs = neg_probs.prod(dim=dim)
        return 1 - prod_neg_probs
    elif method == "max":
        return torch.sigmoid(torch.max(logits_masked[..., mask], dim=dim)[0])
    elif method == "logsumexp":
        return torch.sigmoid(torch.logsumexp(logits_masked[..., mask], dim=dim))
    return torch.sigmoid(logits_masked[..., mask].sum(dim=dim))

class SeqCLSDatasetEHRSHOT(Dataset):
    def __init__(self, data_path, max_length=512, time_horizon=365, approach="direct", multiwin=False):
        """
        Args:
            data_path: Path to ehrshot_inference_data.pkl
            max_length: Context window size
            time_horizon: Prediction window size in days
            approach: 'direct' or 'intermediate_pred'
        """
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.max_length = max_length
        self.time_horizon = time_horizon
        self.approach = approach
        self.multiwin = multiwin
        
        print(f"Loaded {len(self.data)} examples.")

    def _pad_matrix(self, matrix, target_size):
        T = matrix.shape[0]
        padded_matrix = np.zeros((target_size, target_size), dtype=matrix.dtype)
        padded_matrix[:T, :T] = matrix
        np.fill_diagonal(padded_matrix, 1)
        return padded_matrix
        
    def _create_attention_matrix(self, days):
        # Causal mask based on days: Can attend to anything in the past or same day
        attention_matrix = (days[:, None] == days) | (days[:, None] >= days)
        return attention_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        visits = example['visits']
        label = int(example['boolean_value']) # True/False -> 1/0
        
        flat_tokens = []
        flat_days = []
        flat_att_days = []
        
        num_visits = len(visits)
        
        for i, visit in enumerate(visits):
            current_day = int(visit['days'])
            
            visit_tokens = visit['tokens']
            visit_tokens = [t + 6 for t in visit_tokens] # Removed this assumption. User provided token IDs.
            
            flat_tokens.extend(visit_tokens)
            flat_days.extend([current_day] * len(visit_tokens))
            flat_att_days.extend([current_day] * len(visit_tokens))
            
            # If not last visit, insert SEP
            if i < num_visits - 1:
                next_day = int(visits[i+1]['days'])
                
                flat_tokens.append(SEP_ID)
                flat_days.append(next_day)        # pos_embed gets next day
                flat_att_days.append(next_day - 1) # att gets next day - 1
                
        # Convert to numpy
        item_token_arr = np.array(flat_tokens)
        item_days_embed_arr = np.array(flat_days)
        item_att_days_embed_arr = np.array(flat_att_days)
        
        # Truncate / Window Selection (Take latest `max_length - 1` to leave room for final SEP)
        # condition_helper: item_token_arr = token_list_final[-self.max_length+1:]
        
        item_token_arr = item_token_arr[-self.max_length+1:]
        item_days_embed_arr = item_days_embed_arr[-self.max_length+1:]
        item_att_days_embed_arr = item_att_days_embed_arr[-self.max_length+1:]
        
        # Append final SEP/Query token
        item_token_arr = np.append(item_token_arr, SEP_ID)
        
        # For 'direct' approach:
        if self.approach == "direct":
            # Add time horizon to last day for the query position
            # item_days_embed_arr = np.append(item_days_embed_arr, item_days_embed_arr[-1]+self.time_horizon)
            
            # Wait, if array is empty (no history?), handle clean.
            if len(item_days_embed_arr) > 0:
                last_day = item_days_embed_arr[-1]
                last_att = item_att_days_embed_arr[-1]
            else:
                last_day = 0
                last_att = 0
                
            item_days_embed_arr = np.append(item_days_embed_arr, last_day + self.time_horizon)
            item_att_days_embed_arr = np.append(item_att_days_embed_arr, last_att + self.time_horizon)
            
            # Relative days adjustment (start from 0)
            item_days_embed_arr = item_days_embed_arr - item_days_embed_arr[0]
            item_att_days_embed_arr = item_att_days_embed_arr - item_att_days_embed_arr[0]
            
            # Padding
            curr_size = len(item_token_arr)
            pad_len = self.max_length - curr_size
            
            if pad_len > 0:
                item_token_arr = np.pad(item_token_arr, (0, pad_len), mode='constant', constant_values=PAD_ID)
                item_days_embed_arr = np.pad(item_days_embed_arr, (0, pad_len), mode='constant', constant_values=PAD_ID)
                
            sep_indices = np.where(item_token_arr == SEP_ID)[0]
            pred_mask = np.zeros_like(item_token_arr, dtype=bool)
            if len(sep_indices) > 0:
                pred_mask[sep_indices[-1]] = True # Mask the last SEP (the query position)
            
            pad_mask = (item_token_arr != PAD_ID)
            
            attention_bias = self._create_attention_matrix(item_att_days_embed_arr)
            attention_bias = self._pad_matrix(attention_bias, self.max_length)
            
            item = {
                'token_arr': torch.LongTensor(item_token_arr),
                'days_embed_arr': torch.LongTensor(item_days_embed_arr),
                'pred_mask': torch.BoolTensor(pred_mask),
                'pad_mask': torch.BoolTensor(pad_mask),
                'attention_bias': torch.BoolTensor(attention_bias),
                'labels': torch.LongTensor([label])
            }
            
            if self.multiwin:
                item['patient_ids'] = torch.zeros_like(item['token_arr'])
                
            return item
            
        else:
            raise NotImplementedError("Only 'direct' approach implemented for simplified dataset.")

