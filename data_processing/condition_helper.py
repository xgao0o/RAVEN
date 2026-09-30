"""Condition-evaluation datasets and score aggregation."""
from glob import glob
import json
import os
from pathlib import Path
import shelve

import numpy as np
import torch
from torch.utils.data import Dataset

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



# Special token IDs
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4
EOW_ID = 5

class SeqCLSDataset(Dataset):
    def __init__(self,mode = "val", max_length=512, folder="./data/",  multiwin=False,time_horizon = 365*4,approach = "direct", num_patients = -1,horizon = 365,
                 metadata_folder=None, seed=1037, vocab_size=None):
        if max_length < 2 or time_horizon <= 0 or horizon <= 0:
            raise ValueError('max_length must be >= 2 and horizons must be positive')
        if num_patients < -1 or num_patients == 0:
            raise ValueError('num_patients must be -1 or positive')
        if approach not in ('direct', 'intermediate_pred'):
            raise ValueError(f'Unknown approach: {approach}')
        # Support both dbm.dumb (*.dat) and single-file dbm shelves.
        files = sorted({str(path).removesuffix('.dat') for pattern in
                        (f'data_{mode}*.shelve.dat', f'data_{mode}*.shelve')
                        for path in Path(folder).glob(pattern)})
        if not files:
            raise FileNotFoundError(f'No {mode} shelves found in {folder}')
        self.keys, self.feats, self.labels = [], [], []
        self.folder = folder
        self.db_dict = {}
        self.db_paths = files
        self.db_pid = os.getpid()
        self.data_list = []
        for i, file in enumerate(files):
            with shelve.open(file, flag='r') as db:
                self.data_list.extend((i, key) for key in sorted(db.keys()))

        if num_patients!=-1:
            # All ranks must select the same cohort before dividing it up.
            np.random.RandomState(seed).shuffle(self.data_list)
            self.data_list = self.data_list[:num_patients]
        if not self.data_list:
            raise ValueError(f'The {mode} cohort is empty')

                    
        self.max_length = max_length
        self.horizon = horizon
        
        print(f"Loaded {len(self.data_list)} {mode} cohort records")
        self.time_horizon = time_horizon
        self.approach = approach
        
        self.multiwin = multiwin
        metadata_folder = Path(metadata_folder or folder)
        manifest_path = metadata_folder / 'dataset_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest and manifest.get('status') != 'complete':
            raise ValueError(f'Dataset metadata is incomplete: {metadata_folder}')
        remove_inds = np.load(metadata_folder / 'data_files/delete_inds.npy')
        total_prev_tokens = manifest.get('source_width', 57735)
        self.keep_mask = np.ones(total_prev_tokens, dtype=bool)
        self.keep_mask[remove_inds] = False
        self.day_column = manifest.get('relative_day_filtered_column', 35845)
        if vocab_size is not None and int(self.keep_mask.sum()) + 6 != vocab_size:
            raise ValueError('Checkpoint vocabulary does not match the downstream feature metadata')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['db_dict'] = {}
        return state

    def _get_db(self, part):
        # Open independent, read-only handles inside each loader worker.
        if self.db_pid != os.getpid():
            self._cleanup()
            self.db_pid = os.getpid()
        if part not in self.db_dict:
            self.db_dict[part] = shelve.open(self.db_paths[part], flag='r')
        return self.db_dict[part]
    

    def _pad_matrix(self, matrix, target_size):
        T = matrix.shape[0]
        padded_matrix = np.zeros((target_size, target_size), dtype=matrix.dtype)
        padded_matrix[:T, :T] = matrix
        np.fill_diagonal(padded_matrix, 1)
        return padded_matrix
        
    def _create_attention_matrix(self, days):
        T = len(days)
        attention_matrix = (days[:, None] == days) | (days[:, None] >= days)
        return attention_matrix


    def __getitem__(self, idx):
        k, key = self.data_list[idx]
        datum = self._get_db(k)[key]

        """
        'Diagnosis:Anterograde amnesia R41.1' (15099),
        'Diagnosis:Retrograde amnesia R41.2' (15100),
        'Diagnosis:Other amnesia R41.3'(15101)'
        """

        a = datum[1]
        # a[:, [15099, 15100, 15101]] = 0
        label = datum[2]
        if a.shape[1] != len(self.keep_mask):
            raise ValueError('Downstream feature width does not match the selected metadata')
        a = a[:, self.keep_mask]

        column_to_delete = self.day_column
        days_data = a[:, column_to_delete].toarray().astype("int").ravel()

        days_arr, token_arr = np.nonzero(a)
        delete_idx = (token_arr != column_to_delete)
        days_arr = days_arr[delete_idx]
        token_arr = token_arr[delete_idx] + 6

        pos_embed_days = days_data[days_arr]

        change_indices = np.where(np.diff(pos_embed_days) != 0)[0] + 1
        pos_embed_days_final = np.insert(pos_embed_days, change_indices, np.array(days_data)[1:])
        att_days = np.insert(pos_embed_days, change_indices, np.array(days_data)[1:] - 1)
        token_list_final = np.insert(token_arr, change_indices, SEP_ID)
        change_indices = np.where(np.diff(pos_embed_days_final) != 0)[0] + 1

        starting_indices = np.concatenate(([0], change_indices))
        starting_indices[1:] = starting_indices[1:] + 1



       


        item_token_arr = token_list_final[-self.max_length+1:]
        item_token_arr = np.append(item_token_arr, SEP_ID)
        item_days_embed_arr = pos_embed_days_final[-self.max_length+1:]

        if self.approach == "direct":
            item_days_embed_arr = np.append(item_days_embed_arr, item_days_embed_arr[-1]+self.time_horizon)
            item_att_days_embed_arr = att_days[-self.max_length+1:]
            item_att_days_embed_arr = np.append(item_att_days_embed_arr, item_att_days_embed_arr[-1]+self.time_horizon)

                



            item_days_embed_arr = item_days_embed_arr - item_days_embed_arr[0]
            item_att_days_embed_arr = item_att_days_embed_arr - item_att_days_embed_arr[0]

            item_encounter_pos_embed = np.cumsum(np.diff(item_days_embed_arr, prepend=0) > 0)

            # Padding
            curr_size = len(item_token_arr)
            item_token_arr = np.pad(item_token_arr, (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)
            item_days_embed_arr = np.pad(item_days_embed_arr, (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)
            item_encounter_pos_embed = np.pad(item_encounter_pos_embed, (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)

            sep_indices = np.where(item_token_arr == SEP_ID)[0]

            pred_mask = np.zeros_like(item_token_arr, dtype=bool)

            pred_mask[sep_indices[-1]] = True

            pad_mask = (item_token_arr != PAD_ID)

            attention_bias = self._create_attention_matrix(item_att_days_embed_arr)
            attention_bias = self._pad_matrix(attention_bias, self.max_length)


            item = {
                'token_arr': item_token_arr,
                'days_embed_arr': item_days_embed_arr,
                'pred_mask': pred_mask,
                'pad_mask': pad_mask,
                'attention_bias': attention_bias,
                'patient_id': str(key).split('_', 1)[0],
                'record_id': f'{k}:{key}',
                'labels':torch.from_numpy(np.array([label])).to(torch.int64)
            }

            if self.multiwin:
                item['patient_ids'] = np.zeros_like(item['token_arr'])
            
                



            return item
        elif self.approach == "intermediate_pred":
            
            horizons = np.arange(self.horizon, self.time_horizon, self.horizon)
            horizons = np.append(horizons,self.time_horizon)
            
            horizon_token_arr = np.copy(item_token_arr)
            
            all_days_embed_arrs = []
            all_att_days_embed_arrs = []
            
            for horizon in horizons:
                horizon_days_embed = np.copy(item_days_embed_arr)

                horizon_days_embed = np.append(horizon_days_embed, horizon_days_embed[-1]+horizon)
                horizon_att_days = np.copy(att_days[-self.max_length+1:])
                horizon_att_days = np.append(horizon_att_days, horizon_att_days[-1] + horizon)
                
                horizon_days_embed = horizon_days_embed - horizon_days_embed[0]
                horizon_att_days = horizon_att_days - horizon_att_days[0]
                
                all_days_embed_arrs.append(horizon_days_embed)
                all_att_days_embed_arrs.append(horizon_att_days)
            
            combined_days_embed_arr = np.stack(all_days_embed_arrs)
            
            item_days_embed_arr = all_days_embed_arrs[0]
            item_att_days_embed_arr = all_att_days_embed_arrs[0]
            
            item_encounter_pos_embed = np.cumsum(np.diff(item_days_embed_arr, prepend=0) > 0)
            
            curr_size = len(horizon_token_arr)
            horizon_token_arr = np.pad(horizon_token_arr, (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)
            item_encounter_pos_embed = np.pad(item_encounter_pos_embed, (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)
            
            # for i in range(len(horizons)):
            #     combined_days_embed_arr[i] = np.pad(combined_days_embed_arr[i], (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)
            combined_days_embed_arr = np.pad(
                        combined_days_embed_arr,
                        pad_width=((0, 0), (0, self.max_length - curr_size)),
                        mode='constant',
                        constant_values=PAD_ID
                    )

            
            sep_indices = np.where(horizon_token_arr == SEP_ID)[0]
            pred_mask = np.zeros_like(horizon_token_arr, dtype=bool)
            pred_mask[sep_indices[-1]] = True
            
            pad_mask = (horizon_token_arr != PAD_ID)
            
            attention_bias = self._create_attention_matrix(item_att_days_embed_arr)
            attention_bias = self._pad_matrix(attention_bias, self.max_length)
            
            item = {
                'token_arr': horizon_token_arr,
                'days_embed_arr': combined_days_embed_arr,  # Shape: [num_horizons, max_length]
                'pred_mask': pred_mask,
                'pad_mask': pad_mask,
                'attention_bias': attention_bias,
                'patient_id': str(key).split('_', 1)[0],
                'record_id': f'{k}:{key}',
                'labels': torch.from_numpy(np.array([label])).to(torch.int64)
            }
            
            if self.multiwin:
                item['patient_ids'] = np.zeros_like(horizon_token_arr)
            
            return item



            
    def __len__(self):
        return len(self.data_list)

    def _cleanup(self):
        for db in self.db_dict.values():
            db.close()
        self.db_dict.clear()
def aggregate_logits(logits_masked, mask, method="sum", dim=1):
    """
    Aggregate logits using different methods
    
    Args:
        logits_masked: tensor of logits
        mask: boolean tensor indicating which classes to aggregate
        method: aggregation method to use
        dim: dimension to aggregate over (1 for processing, 2 for eval)
        
    Returns:
        aggregated logits tensor
    """
    if method == "sum":
        # Simple sum of logits
        return torch.sigmoid(logits_masked[..., mask].sum(dim=dim))
    
    elif method == "noisy_or":
        # Noisy-OR aggregation
        probs = torch.sigmoid(logits_masked[..., mask])
        neg_probs = 1 - probs
        prod_neg_probs = neg_probs.prod(dim=dim)
        return 1 - prod_neg_probs
    
    elif method == "max":
        # Max pooling
        return torch.sigmoid(torch.max(logits_masked[..., mask], dim=dim)[0])
    
    elif method == "logsumexp":
        # Log-sum-exp pooling
        return torch.sigmoid(torch.logsumexp(logits_masked[..., mask], dim=dim))
    
    elif method == "attention":
        # Attention-weighted sum
        weights = torch.softmax(logits_masked[..., mask], dim=dim)
        return torch.sigmoid((logits_masked[..., mask] * weights).sum(dim=dim))
    
    elif method == "geometric_mean":
        # Geometric mean of probabilities
        probs = torch.sigmoid(logits_masked[..., mask])
        return torch.exp(torch.log(probs + 1e-6).mean(dim=dim))
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def load_latest_checkpoint(out_dir, device):
    checkpoint_files = glob(os.path.join(out_dir, 'ckpt-*.pt'))
    
    with open(os.path.join(out_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {out_dir}")
    
    checkpoint_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    
    latest_checkpoint = checkpoint_files[-1]
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    return checkpoint, checkpoint_model_args,config
