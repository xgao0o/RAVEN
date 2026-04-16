import os
import sys
import argparse
import json
import shelve
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertForPreTraining, BertConfig
from tqdm import tqdm
from glob import glob
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

# --- 1. Constants (From your working block) ---
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
MASK_ID = 3
UNK_ID = 4
headers_special = ['PAD_ID', 'SEP_ID', 'CLS_ID', 'MASK_ID', 'UNK_ID']
N_BOOTSTRAP_FOLDS = 100

# --- 2. Your Exact Working Dataset Class ---
class SeqCLSDataset(Dataset):
    def __init__(self, mode='train', n_tokens=57735, max_length=512, min_num_visits=1, model='BERT', folder="./", age=None, random_window=True):
        self.folder = folder
        # Ensure trailing slash for glob concatenation
        if not self.folder.endswith("/"):
            self.folder += "/"
            
        # EXACT glob pattern from your working code
        files = sorted(glob(self.folder + "data_{}*.shelve.dat".format(mode)))
        
        self.keys, self.feats, self.labels = [], [], []
        self.db_dict = {}
        self.data_list = []
        
        print(f"Loading {mode} files from: {self.folder}")
        print(f"Found {len(files)} files: {files}")

        for i, file in tqdm(enumerate(files), desc="Opening shelves"):
            # Your exact opening logic: strip .dat (safer replace used here to avoid accidental char stripping)
            # If your system produced .shelve.dat, shelve.open usually wants the path WITHOUT .dat
            try:
                base_path = file.replace(".dat", "") 
                self.db_dict[i] = shelve.open(base_path, flag='r')
            except Exception as e:
                print(f"Error opening {file}: {e}")
                continue

        for k, v in self.db_dict.items():
            for key in v.keys():
                self.data_list.append((k, key))
                    
        self.max_length = max_length
        self.min_num_visits = min_num_visits
        self.n_tokens = n_tokens + 5
        self.age = age
        self.random_window = random_window
        self.model = model
        if self.age is not None:
            print("USE AGE as positional embedding")
                
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        k, key = self.data_list[idx]
        datum = self.db_dict[k][key]
        a = datum[1]
        label = datum[2]
        column_to_delete = 51235 
        days_data = a[:, column_to_delete].toarray().astype("int").squeeze()
        
        days_arr , token_arr = np.nonzero(a)
        delete_idx = (token_arr != column_to_delete)
        days_arr = days_arr[delete_idx]
        token_arr = token_arr[delete_idx] + 5
        
        pos_embed_days = days_data[days_arr]

        change_indices = np.where(np.diff(pos_embed_days) != 0)[0] + 1
        pos_embed_days_final = np.insert(pos_embed_days, change_indices, np.array(days_data)[1:])
        att_days = np.insert(pos_embed_days, change_indices, np.array(days_data)[1:] - 1)
        token_list_final = np.insert(token_arr, change_indices, SEP_ID)
        change_indices = np.where(np.diff(pos_embed_days_final) != 0)[0] + 1
        
        starting_indices = np.concatenate(([0], change_indices))
        starting_indices[1:] = starting_indices[1:]+1
        if self.random_window:
            # Safety for negative high
            high_val = starting_indices.shape[0]-self.min_num_visits+1
            start_ind = np.random.randint(0, high=high_val) if high_val > 0 else 0
        else:
            start_ind = 0
            
        end_arr_fin = starting_indices - starting_indices[start_ind]
        
        # Safety for empty where result
        valid_ends = np.where(end_arr_fin < self.max_length)[0]
        end_ind = valid_ends[-1] if len(valid_ends) > 0 else 0
        
        if (start_ind != end_ind):
            days_embed_arr = pos_embed_days_final[starting_indices[start_ind]:starting_indices[end_ind]]
            att_days_embed_arr = att_days[starting_indices[start_ind]:starting_indices[end_ind]]
            token_arr = token_list_final[starting_indices[start_ind]:starting_indices[end_ind]]
        else:
            # Safety check for slice indices
            end_slice = starting_indices[end_ind+1] if (end_ind+1) < len(starting_indices) else len(pos_embed_days_final)
            
            days_embed_arr = pos_embed_days_final[starting_indices[start_ind]:end_slice][-self.max_length:]
            att_days_embed_arr = att_days[starting_indices[start_ind]:end_slice][-self.max_length:]
            token_arr = token_list_final[starting_indices[start_ind]:end_slice]
            
            if len(token_arr) > 1:
                sampled_elements = np.random.choice(token_arr[:-1], min(len(token_arr)-1, self.max_length - 1), replace=False)
                token_arr = np.concatenate([sampled_elements, [token_arr[-1]]])
                
        pred_mask = (token_arr == SEP_ID)
        
        if len(days_embed_arr) > 0:
            days_embed_arr = days_embed_arr - days_embed_arr[0]
            att_days_embed_arr = att_days_embed_arr - att_days_embed_arr[0]
        
        encounter_pos_embed = np.cumsum(np.diff(days_embed_arr, prepend=0) > 0)
        curr_size = len(token_arr)
        
        # Safe concatenation
        days_embed_arr = np.concatenate(([0], days_embed_arr[:-1])) if len(days_embed_arr) > 0 else np.array([0])
        encounter_pos_embed = np.concatenate(([0], encounter_pos_embed[:-1])) if len(encounter_pos_embed) > 0 else np.array([0])
        
        pad_len = max(0, self.max_length - curr_size)
        
        attn_mask = np.concatenate(([1], np.pad(np.ones((curr_size)), (0, pad_len), mode='constant', constant_values=PAD_ID)))
        token_arr = np.concatenate(([CLS_ID], np.pad(token_arr,(0, pad_len), mode='constant', constant_values=PAD_ID)))
        days_embed_arr = np.concatenate(([PAD_ID], np.pad(days_embed_arr + 1,(0, pad_len), mode='constant', constant_values=PAD_ID)))
        encounter_pos_embed = np.concatenate(([PAD_ID], np.pad(encounter_pos_embed + 1,(0, pad_len), mode='constant', constant_values=PAD_ID)))
        
        # Tensor conversion
        return {
            'input_ids': torch.tensor(token_arr, dtype=torch.long),
            'attention_mask':torch.tensor(attn_mask, dtype=torch.long), 
            'token_type_ids':torch.tensor(encounter_pos_embed, dtype=torch.long),
            'position_ids':torch.tensor(days_embed_arr, dtype=torch.long),
            'labels':torch.tensor([label], dtype=torch.long)
        }

# --- 3. Your Exact Model Class ---
class Finetune_cls(nn.Module):
    def __init__(self, model, d_hidden, label_classes = 2):
        super().__init__()
        self.model = model
        self.proj_label_classes = nn.Linear(d_hidden, label_classes)

    def forward(self, tokens_input_ids, tokens_type_ids, tokens_attention_mask, position_ids):
        tokens_hidden_states, _ = self.model(input_ids = tokens_input_ids,
                                             token_type_ids = tokens_type_ids,
                                             attention_mask = tokens_attention_mask,
                                             position_ids = position_ids).values()
        tokens_state = tokens_hidden_states[:, 0, :]
        return self.proj_label_classes(tokens_state)

# --- 4. Evaluation Logic (Headless) ---
def load_checkpoint_flexible(model, checkpoint_path, device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    is_model_dp = isinstance(model, nn.DataParallel)
    is_ckpt_dp = any(k.startswith('module.') for k in checkpoint.keys())
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    # Reconcile DataParallel keys
    if is_model_dp and not is_ckpt_dp:
        for k, v in checkpoint.items(): new_state_dict['module.' + k] = v
    elif not is_model_dp and is_ckpt_dp:
        for k, v in checkpoint.items(): new_state_dict[k.replace('module.', '')] = v
    else:
        new_state_dict = checkpoint

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Warning: Strict load failed ({e}). Retrying with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
    
    return model

@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluating"):
        logits = model(
            batch['input_ids'].to(device),
            batch['token_type_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['position_ids'].to(device)
        )
        probs = nn.Softmax(dim=1)(logits)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch['labels'].flatten().cpu().numpy())

    return np.array(all_probs), np.array(all_labels)

def save_metrics_and_plot(probs, labels, output_dir, file_prefix):
    if len(labels) == 0: return None
    
    # Calculate CI using Bootstrap
    aps, aucs = [], []
    mean_rec = np.linspace(0, 1, 100)
    mean_fpr = np.linspace(0, 1, 100)
    ppvs, tprs = [], []

    for _ in range(N_BOOTSTRAP_FOLDS):
        ix = np.random.choice(len(labels), size=int(0.8*len(labels)), replace=True)
        if len(np.unique(labels[ix])) < 2: continue
        
        # Metrics
        aps.append(average_precision_score(labels[ix], probs[ix]))
        aucs.append(roc_auc_score(labels[ix], probs[ix]))
        
        # Curves
        p, r, _ = precision_recall_curve(labels[ix], probs[ix])
        ppvs.append(np.interp(mean_rec, r[::-1], p[::-1]))
        
        fpr, tpr, _ = roc_curve(labels[ix], probs[ix])
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    # Stats
    auprc_mean, auprc_std = np.mean(aps), np.std(aps)
    auroc_mean, auroc_std = np.mean(aucs), np.std(aucs)

    # Plot
    plt.figure(figsize=(20, 9))
    
    # PR
    ax1 = plt.subplot(121)
    mean_ppv = np.mean(ppvs, axis=0)
    ax1.plot(mean_rec, mean_ppv, label=f'AUPRC: {auprc_mean:.4f}')
    ax1.fill_between(mean_rec, np.maximum(mean_ppv-np.std(ppvs, axis=0),0), np.minimum(mean_ppv+np.std(ppvs, axis=0),1), alpha=0.2, color='blue')
    ax1.set_title("PR Curve"); ax1.legend()
    
    # ROC
    ax2 = plt.subplot(122)
    mean_tpr = np.mean(tprs, axis=0)
    ax2.plot(mean_fpr, mean_tpr, label=f'AUROC: {auroc_mean:.4f}')
    ax2.fill_between(mean_fpr, np.maximum(mean_tpr-np.std(tprs, axis=0),0), np.minimum(mean_tpr+np.std(tprs, axis=0),1), alpha=0.2, color='blue')
    ax2.set_title("ROC Curve"); ax2.legend()
    
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_curves.png"))
    plt.close()

    return {
        "auprc": auprc_mean, "auprc_std": auprc_std,
        "auroc": auroc_mean, "auroc_std": auroc_std
    }

# --- 5. Main Execution ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, required=True)
    parser.add_argument('--condition', type=str, required=True)
    parser.add_argument('--model_base_dir', type=str, required=True)
    parser.add_argument('--data_base_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    model_dir = os.path.join(args.model_base_dir, f"{args.setting}_finetuning_rmAmnesia")
    model_path = os.path.join(model_dir, "best_model_val.pt")
    data_full_path = os.path.join(args.data_base_dir, args.setting)
    
    # Fallback if setting folder is not nested
    if not os.path.exists(data_full_path):
        data_full_path = args.data_base_dir

    print(f"Config: {args.condition} | {args.setting}")
    print(f"Data Path: {data_full_path}")
    print(f"Model Path: {model_path}")

    # Load Data (Try Test, then Val)
    try:
        dataset = SeqCLSDataset(mode='test', folder=data_full_path, random_window=False)
        if len(dataset) == 0:
            print("Test empty. Trying Val...")
            dataset = SeqCLSDataset(mode='val', folder=data_full_path, random_window=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if len(dataset) == 0:
        print("CRITICAL: Dataset empty. Check paths and shelve files.")
        return

    # Initialize Model
    config = BertConfig(num_hidden_layers=12, vocab_size=57735+5, max_position_embeddings=4000, type_vocab_size=4000)
    bert_base = BertForPreTraining(config).bert
    model = Finetune_cls(bert_base, d_hidden=768)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    model = load_checkpoint_flexible(model, model_path, device)

    # Evaluate
    probs, labels = evaluate(model, dataset, args.batch_size, device)
    
    # Save Results
    metrics = save_metrics_and_plot(probs, labels, model_dir, f"results_{args.setting}_{args.condition}")
    if metrics:
        with open(os.path.join(model_dir, f"results_{args.setting}_{args.condition}.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Done.")

if __name__ == "__main__":
    main()