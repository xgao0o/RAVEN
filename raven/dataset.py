import sys
import os
import pandas as pd
import shelve
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm



def custom_collate(batch):
    # Flatten the batch
    flattened_batch = [item for patient_items in batch for item in patient_items]
    
    # If the flattened batch is empty, return an empty dict
    if len(flattened_batch) == 0:
        return {}

    # Get the keys from the first item
    keys = flattened_batch[0].keys()
    
    # Initialize the result dictionary
    result = {key: [] for key in keys}
    
    # Collect items
    for item in flattened_batch:
        for key in keys:
            result[key].append(item[key])
    
    for key in keys:
        if key == 'patient_id':
            continue  # Keep patient_id as is
        elif key == 'temporal_weights':
            # Special handling for temporal_weights
            if all(x is None for x in result[key]):
                continue  # Keep it as a list of None
            if isinstance(result[key][0], np.ndarray):
                # Convert to tensors and stack if they're numpy arrays
                tensors = [torch.from_numpy(arr) for arr in result[key] if arr is not None]
                if tensors:  # Only stack if we have tensors
                    result[key] = torch.stack(tensors)
        else:
            # For all other numpy array fields
            if isinstance(result[key][0], np.ndarray):
                tensors = [torch.from_numpy(arr) for arr in result[key]]
                result[key] = torch.stack(tensors)
    
    return result

# Special token IDs
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4
EOW_ID = 5

class UnifiedSeqEHRDataset(Dataset):
    def __init__(self, df, max_length=512, model='BERT', folder="./data/", age=None, train=False, agg_labels=False, temporal_decay=None,multiwin=False):
        self.df = df
        self.max_length = max_length
        self.folder = folder
        print("Open db files:")
        self.db_dict = {part: shelve.open(self.folder+'/processed_part_'+str(part)+'.shelve') for part in tqdm(range(40))}
        self.age = age
        self.model = model
        self.train = train
        remove_inds = np.load(os.path.join(folder, "data_files", "delete_inds.npy"))
        total_prev_tokens = 57735
        self.keep_mask = np.ones(total_prev_tokens, dtype=bool)
        self.keep_mask[remove_inds] = False
        self.agg_labels = agg_labels
        self.temporal_decay = temporal_decay
        self.multiwin = multiwin
        print("temporal decay: ", self.temporal_decay)
    
    def _compute_temporal_weights(self, labels, decay_factor=0.5,min_weight = 0.1):
        n_t, n_classes = labels.shape
        decay_factor = float(decay_factor)
        
        
        indices = np.arange(n_t)[:, None]
        mask = labels.astype(bool)
                
        cumulative_counts = np.cumsum(mask, axis=0) * mask
        
        positions = cumulative_counts - 1
    
        decayed_weights = decay_factor ** positions

        
        weights = np.where(mask, np.maximum(decayed_weights, min_weight), 1)
        return weights


    def _sample_train_window(self, a, days_data):
        """Helper function to sample and process a single training window with length constraints"""
        days_arr, token_arr = np.nonzero(a)
        column_to_delete = 35845
        delete_idx = (token_arr != column_to_delete)
        days_arr = days_arr[delete_idx]
        token_arr = token_arr[delete_idx] + 6

        pos_embed_days = days_data[days_arr]
        
        # Find visit changes
        change_indices = np.where(np.diff(pos_embed_days) != 0)[0] + 1
        
        # Insert SEP tokens
        token_list_final = np.insert(token_arr, change_indices, SEP_ID)
        pos_embed_days_final = np.insert(pos_embed_days, change_indices, np.array(days_data)[1:])
        att_days = np.insert(pos_embed_days, change_indices, np.array(days_data)[1:] - 1)

        # Recompute indices after insertion
        change_indices = np.where(np.diff(pos_embed_days_final) != 0)[0] + 1
        starting_indices = np.concatenate(([0], change_indices))
        starting_indices[1:] = starting_indices[1:] + 1

        # Sample start index
        start_ind = np.random.randint(0, high=starting_indices.shape[0] - 1)
        start = starting_indices[start_ind]

        # Try to find suitable end_ind that keeps sequence length <= max_length//4
        current_end_ind = start_ind + 1
        while current_end_ind < starting_indices.shape[0]:
            tentative_end = starting_indices[current_end_ind]
            if tentative_end - start > self.max_length//2:
                current_end_ind -= 1
                break
            current_end_ind += 1
        current_end_ind = min(current_end_ind,starting_indices.shape[0]-1)

        #print(current_end_ind)

        if current_end_ind == start_ind:
            # First visit is too long - sample subset of tokens
            next_visit_start = starting_indices[start_ind+1]
            available_tokens = token_list_final[start:next_visit_start-1]
            
            # Sample tokens
            total_tokens = self.max_length//2 - 1  # Leave space for SEP
            if len(available_tokens) > total_tokens:
                sampled_indices = np.random.choice(len(available_tokens), total_tokens, replace=False)
                sampled_indices.sort()  # Keep temporal order
                item_token_arr = np.concatenate([available_tokens[sampled_indices], [SEP_ID]])
                
                # Sample corresponding days and attention days
                item_days_embed_arr = pos_embed_days_final[start:next_visit_start][sampled_indices]
                item_days_embed_arr = np.append(item_days_embed_arr, pos_embed_days_final[next_visit_start-1])
                item_att_days_embed_arr = att_days[start:next_visit_start][sampled_indices]
                item_att_days_embed_arr = np.append(item_att_days_embed_arr, att_days[next_visit_start-1])
            else:
                # If visit is shorter than max length, take all tokens
                item_token_arr = np.concatenate([available_tokens, [SEP_ID]])
                item_days_embed_arr = pos_embed_days_final[start:next_visit_start+1]
                item_att_days_embed_arr = att_days[start:next_visit_start+1]
            
            # For labels, we only predict at the SEP token
            end_ind = start_ind + 1
        else:
            # Normal case - we found a suitable end point
            end_ind = np.random.randint(start_ind + 1, current_end_ind+1)
            end = starting_indices[end_ind]
            item_token_arr = token_list_final[start:end]
            item_days_embed_arr = pos_embed_days_final[start:end]
            item_att_days_embed_arr = att_days[start:end]

        # Normalize days embeddings
        item_days_embed_arr = item_days_embed_arr - item_days_embed_arr[0]
        item_att_days_embed_arr = item_att_days_embed_arr - item_att_days_embed_arr[0]
        
        # Create labels
        all_labels = a.toarray() != 0
        if self.agg_labels:
            all_labels = self._fix_labels(all_labels, days_data)
                
        pred_mask = (item_token_arr == SEP_ID)
        pad_mask = np.ones_like(item_token_arr, dtype=bool)
        num_sep = np.sum(pred_mask)
        
        labels = np.zeros((len(item_token_arr), a.shape[1]))
        labels[pred_mask] = all_labels[start_ind+1:start_ind+1+num_sep]
        labels[:, column_to_delete] = False
        labels = np.append(np.zeros((len(item_token_arr), 6)), labels, axis=1)

        if self.temporal_decay is not None:
            temporal_weights = self._compute_temporal_weights(labels, self.temporal_decay)
        else:
            temporal_weights = None
                
        return {
            'token_arr': item_token_arr,
            'days_embed_arr': item_days_embed_arr,
            'att_days': item_att_days_embed_arr,
            'labels': labels,
            'pred_mask': pred_mask,
            'pad_mask': pad_mask,
            'temporal_weights': temporal_weights,
            'sequence_length': len(item_token_arr)
        }
    
    
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

    def _combine_windows(self, windows):
        """Helper function to combine multiple windows into one sequence"""
        total_length = sum(w['sequence_length'] for w in windows[:-1]) + \
                        sum(1 for _ in windows[:-1]) + \
                        windows[-1]['sequence_length']  # Add EOW tokens between windows except last
                        
        combined_tokens = np.zeros(self.max_length, dtype=np.int64)
        combined_days = np.zeros(self.max_length, dtype=np.int64)
        combined_dummy_days = np.zeros(self.max_length, dtype=np.int64)
        combined_pad_mask = np.zeros(self.max_length, dtype=bool)
        combined_pred_mask = np.zeros(self.max_length, dtype=bool)
        combined_labels = np.zeros((self.max_length, windows[0]['labels'].shape[1]), dtype=bool)
        patient_ids = np.zeros(self.max_length, dtype=np.int64)

        has_temporal_weights = all(w.get('temporal_weights') is not None for w in windows)
        if has_temporal_weights:
            combined_temporal_weights = np.ones((self.max_length, windows[0]['labels'].shape[1]), dtype=np.float32)
        else:
            combined_temporal_weights = None



        current_pos = 0
        day_offset = 0  # Keep track of day offset for each patient

        # Combine windows
        for i, window in enumerate(windows):
            seq_len = window['sequence_length']
            
            # Add window content
            combined_tokens[current_pos:current_pos + seq_len] = window['token_arr']
            # Add offset to days for this patient
            combined_dummy_days[current_pos:current_pos + seq_len] = window['days_embed_arr'] + day_offset
            combined_days[current_pos:current_pos + seq_len] = window['days_embed_arr'] 
            combined_pad_mask[current_pos:current_pos + seq_len] = window['pad_mask']
            combined_pred_mask[current_pos:current_pos + seq_len] = window['pred_mask']
            combined_labels[current_pos:current_pos + seq_len] = window['labels']
            patient_ids[current_pos:current_pos + seq_len] = i

            if has_temporal_weights:
                combined_temporal_weights[current_pos:current_pos + seq_len] = window['temporal_weights']

            
            # Add EOW token if not last window
            if i < len(windows) - 1:
                combined_tokens[current_pos + seq_len] = EOW_ID
                combined_dummy_days[current_pos + seq_len] = window['days_embed_arr'][-1] + day_offset
                combined_days[current_pos + seq_len] = window['days_embed_arr'][-1]
                combined_pad_mask[current_pos + seq_len] = True
                patient_ids[current_pos + seq_len] = i
                

                if has_temporal_weights:
                    combined_temporal_weights[current_pos + seq_len] = 1

                current_pos += seq_len + 1
            else:
                current_pos += seq_len
                
            # Update offset for next patient - add some gap (e.g. 1000) to ensure separation
            day_offset += window['days_embed_arr'][-1] + 1000

        sep_mask = (combined_tokens==1)
        #combined_dummy_days[sep_mask]-=0.5 
        combined_dummy_days[sep_mask] = combined_dummy_days[sep_mask].astype(float) - 0.5



        # Create single attention matrix using modified days
        attention_matrix = self._create_attention_matrix(combined_dummy_days[:current_pos])
        attention_bias = self._pad_matrix(attention_matrix, self.max_length)

        result = {
            'token_arr': combined_tokens,
            'days_embed_arr': combined_days,
            'labels': combined_labels,
            'pred_mask': combined_pred_mask,
            'pad_mask': combined_pad_mask,
            'attention_bias': attention_bias,
            'patient_ids': patient_ids
        }

        if has_temporal_weights:
            result['temporal_weights'] = combined_temporal_weights


        return result

    def __getitem__(self, idx):
        if (not self.train) or (not self.multiwin):
            loc = self.df.iloc[idx]
            part = loc["part"]
            patient_id = loc["patient_id"]
            a = self.db_dict[part][patient_id]

            a = a[:, self.keep_mask]

            if self.model == 'GPT' and self.train:
                a = self._remove_random_rows_csr(a, percentage=0.1)

            column_to_delete = 35845  # day of the visit
            days_data = a[:, column_to_delete].toarray().astype("int").squeeze()

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

            if self.train:
                start_ind = np.random.randint(0, high = starting_indices.shape[0] - 1)
                end_ind = np.random.randint(start_ind + 1, high=starting_indices.shape[0])
            else:
                start_ind = 0
                end_ind = starting_indices.shape[0] - 1

            items = []
            current_start = start_ind


            # while current_start < end_ind:
            #     current_end = current_start
            #     current_length = 0

            #     while current_end <= end_ind and current_length < self.max_length:
            #         next_length = starting_indices[current_end + 1] - starting_indices[current_start] if current_end + 1 <= end_ind else len(token_list_final) - starting_indices[current_start]
            #         if current_length + next_length > self.max_length:
            #             break
            #         current_end += 1
            #         current_length = next_length
            #     if current_end == current_start and next_length>self.max_length:
            #         current_end +=1
            #     start = starting_indices[current_start]
            #     end = starting_indices[current_end] if current_end < len(starting_indices) else len(token_list_final)
                
            while current_start < end_ind:
                current_end = current_start
                accumulated_length = 0
                
                # Keep adding visits as long as they fit within max_length
                while current_end < end_ind:
                    if current_end + 1 >= len(starting_indices):
                        next_visit_length = len(token_list_final) - starting_indices[current_end]
                    else:
                        next_visit_length = starting_indices[current_end + 1] - starting_indices[current_end]
                    
                    if accumulated_length + next_visit_length > self.max_length:
                        break
                        
                    accumulated_length += next_visit_length
                    current_end += 1

                if current_end == current_start:
                    start = starting_indices[current_start]
                    item_token_arr = token_list_final[start:start + self.max_length]
                    item_days_embed_arr = pos_embed_days_final[start:start + self.max_length]
                    item_att_days_embed_arr = att_days[start:start + self.max_length]
                    current_end = current_start + 1
                else:
                    start = starting_indices[current_start]
                    end = starting_indices[current_end] if current_end < len(starting_indices) else len(token_list_final)
                    item_token_arr = token_list_final[start:end]
                    item_days_embed_arr = pos_embed_days_final[start:end]
                    item_att_days_embed_arr = att_days[start:end]

                



                item_days_embed_arr = item_days_embed_arr - item_days_embed_arr[0]
                item_att_days_embed_arr = item_att_days_embed_arr - item_att_days_embed_arr[0]

                item_encounter_pos_embed = np.cumsum(np.diff(item_days_embed_arr, prepend=0) > 0)

                # Padding
                curr_size = len(item_token_arr)
                item_token_arr = np.pad(item_token_arr, (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)
                item_days_embed_arr = np.pad(item_days_embed_arr, (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)
                item_encounter_pos_embed = np.pad(item_encounter_pos_embed, (0, self.max_length - curr_size), mode='constant', constant_values=PAD_ID)

                pred_mask = (item_token_arr == SEP_ID)
                pad_mask = (item_token_arr != PAD_ID)

                attention_bias = self._create_attention_matrix(item_att_days_embed_arr)
                attention_bias = self._pad_matrix(attention_bias, self.max_length)

                # Create labels
                labels = np.zeros((pred_mask.shape[0], a.shape[1]))
                pred_inds = np.where(pred_mask)[0]
                all_labels = a.toarray() != 0
                if self.agg_labels:
                    all_labels = self._fix_labels(all_labels,days_data)
                
                labels[pred_inds] = all_labels[current_start+1:current_end+1]

                labels[:, column_to_delete] = False
                labels = np.append(np.zeros((self.max_length, 6)), labels, axis=1).astype(bool)
                

                if self.temporal_decay is not None:
                    temporal_weights = self._compute_temporal_weights(labels,self.temporal_decay)
                    temporal_weights = temporal_weights.astype('float')
                else:
                    temporal_weights = None

                item = {
                    'token_arr': item_token_arr,
                    'days_embed_arr': item_days_embed_arr,
                    'labels': labels.astype('int'),
                    'pred_mask': pred_mask,
                    'pad_mask': pad_mask,
                    'attention_bias': attention_bias,
                    'temporal_weights': temporal_weights,
                    'patient_id': patient_id,
                    'first_visit_labels': np.concatenate([np.zeros(6), all_labels[current_start]])
                }

                if (not self.train) and self.multiwin:
                    item['patient_ids'] = np.zeros_like(item['token_arr'])
                

                items.append(item)
                current_start = current_end
                # if self.train: 
                break # to have consistent batch sizes when eval (while subbatching helps,
                        # we only want what fits on context)

            return items
            
        else:
            # Training logic - combine windows from multiple patients
            windows = []
            total_length = 0
            
            while total_length < self.max_length:
                # Sample patient
                if not windows:
                    loc = self.df.iloc[idx]
                else:
                    loc = self.df.iloc[np.random.randint(len(self.df))]
                    
                part = loc["part"]
                patient_id = loc["patient_id"]
                a = self.db_dict[part][patient_id]
                a = a[:, self.keep_mask]
                
                days_data = a[:, 35845].toarray().astype("int").squeeze()
                
                # Sample window
                window = self._sample_train_window(a, days_data)
                
                # Check if window fits
                if total_length >0 and total_length + window['sequence_length'] + 1 > self.max_length:
                    break
                    
                    
                windows.append(window)
                total_length += window['sequence_length'] + 1  # +1 for EOW token

            #print("windows ",windows)
                
            # Combine all windows
            combined_sequence = self._combine_windows(windows)
            return combined_sequence  # Return as list for consistency with eval mode

    def __len__(self):
        return len(self.df)

    def _cleanup(self):
        del self.db_dict