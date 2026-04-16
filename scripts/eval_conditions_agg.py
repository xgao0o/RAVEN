import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pandas as pd
from dataclasses import dataclass
import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import Pool, cpu_count

from raven.model import GPTConfig, GPT
from raven.dataset import UnifiedSeqEHRDataset, custom_collate
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import argparse
import torch.multiprocessing as mp
import json
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


def parse_args():
    parser = argparse.ArgumentParser(description='Compute optimal thresholds and evaluate model.')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--folder', type=str, default='./folder', help='Data directory')
    parser.add_argument('--csv_folder', type=str, default='./csv_folder', help='CSV path directory')
    parser.add_argument('--num_batches', type=int, default=512, help='Number of batches to process for threshold computation')
    parser.add_argument('--seed', type=int, default=1001, help='seed for test sampling')
    parser.add_argument('--test_patients', type=int, default=-1, help='Number of patients to test on')
    parser.add_argument('--n_gather', type=int, default=100, help='when to gather')
    parser.add_argument('--perc', type=int, default=90, help='Percentage for eval processing')
    parser.add_argument('--mode', type=str, default='all', help='Evaluation mode: threshold, evaluate, or all')
    parser.add_argument('--opt_thresh_name', type=str, default='all', help='Evaluation mode: threshold, evaluate, or all')
    parser.add_argument('--agg_method', type=str, default='sum', choices=['sum', 'noisy_or','max','logsumexp','geometric_mean'])
    parser.add_argument('--threshold_method', type=str, default='f1', choices=['f1', 'quantile'],
                        help='Method to compute threshold: "f1" for optimizing F1 score (default) or "quantile" for quantile-based thresholds')
    parser.add_argument('--short_term_threshold', type=int, default=30, help='Short term threshold for time gap evaluation')
    parser.add_argument('--save_logits_labels', action='store_true', 
                        help='If set, save the aggregated logits and labels when computing thresholds (only available when computing thresholds)')
    parser.add_argument('--all_conditions', action='store_true', 
                        help='If set, load "conditions2inds_update.json" (all diseases), otherwise load "conditions2inds_pcancer_dementia_update.json"')
    parser.add_argument('--iteration', type=int, default=None, help='model at which iteration to evaluate')
    return parser.parse_args()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_metrics(TP, TN, FP, FN):
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    total_cases = TP + TN + FP + FN
    total_actual_positive = TP + FN       # Ground truth positives
    total_actual_negative = TN + FP       # Ground truth negatives
    return precision, recall, specificity, f1_score, total_cases, total_actual_positive, total_actual_negative


def worker(all_logits_sigmoid, all_labels, i):
    if all_labels[:, i].sum() == 0:
        return -1
    precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_logits_sigmoid[:, i])
    
    eps = 1e-10
    f1_scores = 2 * (precision * recall) / (precision + recall + eps)
    max_f1_index = f1_scores.argmax()
    
    if max_f1_index == len(thresholds):  
        opt_threshold = 1.0  
    else:
        opt_threshold = thresholds[max_f1_index]
    
    return opt_threshold


def get_optimal_thresholds(all_logits, all_labels):
    optimal_thresholds = np.full(all_logits.shape[1], np.nan)
    pr_curves = {
        'precision': {},
        'recall': {},
        'thresholds': {}
    }
    
    for ind in range(all_labels.shape[-1]):
        if all_labels[:, ind].sum() == 0:
            optimal_thresholds[ind] = -1
            continue
            
        precision, recall, thresholds = precision_recall_curve(all_labels[:, ind], all_logits[:, ind])
        
        eps = 1e-10
        f1_scores = 2 * (precision * recall) / (precision + recall + eps)
        max_f1_index = f1_scores.argmax()
        
        if max_f1_index == len(thresholds):  
            opt_threshold = 1.0  
        else:
            opt_threshold = thresholds[max_f1_index]
        
        optimal_thresholds[ind] = opt_threshold
        
        # Store as dictionary with condition index as key
        pr_curves['precision'][str(ind)] = precision.tolist()
        pr_curves['recall'][str(ind)] = recall.tolist()
        pr_curves['thresholds'][str(ind)] = thresholds.tolist() if thresholds is not None else []
    
    return optimal_thresholds, pr_curves


def get_quantile_thresholds(all_logits, all_labels):
    """
    Compute thresholds for each disease based on quantiles.
    For each condition, the threshold is the (1 - prevalence) quantile of its logits.
    If a condition has no positive samples, threshold is set to +inf,
    and if all samples are positive, threshold is set to -inf.
    """
    prevalence_rates = all_labels.sum(axis=0) / all_labels.shape[0]
    num_diseases = all_labels.shape[1]
    thresholds = np.zeros(num_diseases)
    
    for d in range(num_diseases):
        prevalence = prevalence_rates[d]
        if prevalence == 0:
            thresholds[d] = np.inf
            print(f"Disease {d + 1}: Prevalence = 0.0, Threshold set to +inf")
        elif prevalence == 1:
            thresholds[d] = -np.inf
            print(f"Disease {d + 1}: Prevalence = 1.0, Threshold set to -inf")
        else:
            threshold = np.quantile(all_logits[:, d], 1 - prevalence)
            thresholds[d] = threshold
            print(f"Disease {d + 1}: Prevalence = {prevalence:.2%}, Threshold = {threshold:.4f}")
    return thresholds


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


def gather_variable_results(tensor, rank, world_size):
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [size.item() for size in all_sizes]
    
    max_size = max(all_sizes)
    
    padded_tensor = torch.zeros((max_size,) + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:tensor.shape[0]] = tensor
    
    gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded_tensor)
    
    result = torch.cat([tensor[:size] for tensor, size in zip(gathered_tensors, all_sizes)])
    
    return result.cpu()


@torch.no_grad()
def process_batches_aggregated(model, loader, condition_indices, rank, device, bs, world_size, num_batches, n_gather=100, agg_method="sum"):
    model.eval()
    all_logits = []
    all_labels = []
    
    # Create condition masks
    N_C = next(iter(loader))['labels'].shape[-1]
    condition_masks = torch.zeros((len(condition_indices), N_C), dtype=torch.bool, device=device)
    for idx, (condition, indices) in enumerate(condition_indices.items()):
        condition_masks[idx, indices] = True
    
    batch_logits = []
    batch_labels = []
    
    for batch_idx, batch in enumerate(tqdm(loader)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        token_arr = batch['token_arr']
        days_embed_arr = batch['days_embed_arr']
        labels = batch['labels']
        pred_mask = batch['pred_mask']
        pad_mask = batch['pad_mask']
        attention_bias = batch['attention_bias']

        batch_size = token_arr.shape[0]
        
        for start_idx in range(0, batch_size, bs):
            end_idx = min(start_idx + bs, batch_size)
            
            sub_token_arr = token_arr[start_idx:end_idx]
            sub_days_embed_arr = days_embed_arr[start_idx:end_idx]
            sub_labels = labels[start_idx:end_idx].float()
            sub_pred_mask = pred_mask[start_idx:end_idx]
            sub_pad_mask = pad_mask[start_idx:end_idx]
            sub_attention_bias = attention_bias[start_idx:end_idx]
            sub_patient_ids = None
            if 'patient_ids' in batch:
                sub_patient_ids = batch['patient_ids'][start_idx:end_idx]
            
            N_C = sub_labels.shape[-1]
            
            logits, _, _, _ = model(sub_token_arr, sub_days_embed_arr, sub_labels, sub_pred_mask, sub_pad_mask, sub_attention_bias,patient_ids = sub_patient_ids)
            
            # Reshape to (batch * seq_len, N_C)
            logits = logits.reshape(-1, N_C)
            sub_labels_flat = sub_labels.reshape(-1, N_C)
            sub_pred_mask_flat = sub_pred_mask.reshape(-1)
            
            # Apply pred_mask
            logits_masked = logits[sub_pred_mask_flat]
            labels_masked = sub_labels_flat[sub_pred_mask_flat]
            
            # Aggregate logits and labels per condition
            agg_logits = torch.zeros((logits_masked.size(0), len(condition_indices)), device=device)
            agg_labels = torch.zeros((logits_masked.size(0), len(condition_indices)), device=device)
            
            for cond_idx in range(len(condition_indices)):
                mask = condition_masks[cond_idx]
                agg_labels[:, cond_idx] = (labels_masked[:, mask].sum(dim=1) > 0).float()

                agg_logits[:, cond_idx] = aggregate_logits(logits_masked, mask, method=agg_method, dim=1)
            
            batch_logits.append(agg_logits)
            batch_labels.append(agg_labels)

            del logits, sub_labels, sub_pred_mask, logits_masked, labels_masked
            del sub_token_arr, sub_days_embed_arr, sub_pad_mask, sub_attention_bias
            torch.cuda.empty_cache()
        
        del batch, token_arr, days_embed_arr, labels, pred_mask, pad_mask, attention_bias
        torch.cuda.empty_cache()
        
        if (batch_idx + 1) % n_gather == 0 or batch_idx == num_batches - 1:
            batch_logits = torch.cat(batch_logits)
            batch_labels = torch.cat(batch_labels)
            
            dist.barrier()
            gathered_logits = gather_variable_results(batch_logits, rank, world_size)
            gathered_labels = gather_variable_results(batch_labels, rank, world_size)
            
            if rank == 0:
                all_logits.append(gathered_logits.numpy())
                all_labels.append(gathered_labels.numpy())
            
            batch_logits = []
            batch_labels = []

            del gathered_logits, gathered_labels
            torch.cuda.empty_cache()
        
        if batch_idx >= num_batches - 1:
            break
    
    if rank == 0:
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_logits, all_labels
    else:
        return None, None
    

def process_patient_vectorized(patient_data):
    """Process single patient - vectorized version"""
    condition_exists = np.any(patient_data['condition_labels'], axis=0)
    first_visit_conditions = np.any(patient_data['first_visit_labels'][None, :] * patient_data['condition_masks'], axis=1)
    valid_conditions = condition_exists & ~first_visit_conditions
    
    if not np.any(valid_conditions):
        return {}
    
    # For all valid conditions, find first occurrence in GT
    condition_gt = patient_data['condition_labels'][:, valid_conditions]
    gt_first_occur = np.argmax(condition_gt, axis=0)
    gt_days = patient_data['days'][gt_first_occur]
    
    # For predictions
    condition_preds = patient_data['condition_preds'][:, valid_conditions]
    has_prediction = np.any(condition_preds, axis=0)
    pred_first_occur = np.argmax(condition_preds, axis=0)
    pred_days = np.where(has_prediction, patient_data['days'][pred_first_occur], -1)
    
    # Compute on_time in vectorized way
    on_time = (pred_days <= gt_days) & (pred_days != -1)
    
    # Create results dict
    valid_indices = np.where(valid_conditions)[0]
    return {
        idx: {
            "on_time": on_time[i],
            "pred_day": pred_days[i],
            "gt_day": gt_days[i]
        }
        for i, idx in enumerate(valid_indices)
    }


def evaluate_on_time_parallel(binary_preds, labels, pred_mask, first_visit_labels, days_embed, condition_masks, device):
    """
    Vectorized evaluation function with proper dtype handling
    """
    batch_size = binary_preds.shape[0]
    num_conditions = condition_masks.shape[0]
    
    batch_results = {
        condition_idx: {
            "gt_days": [],
            "pred_days": [],
            "on_time": []
        }
        for condition_idx in range(num_conditions)
    }
    
    condition_masks = condition_masks.float()
    
    for patient_idx in range(batch_size):
        valid_preds = binary_preds[patient_idx][pred_mask[patient_idx]].float()  # ensure float
        valid_labels = labels[patient_idx][pred_mask[patient_idx]].float()       # ensure float
        valid_days = days_embed[patient_idx][pred_mask[patient_idx]]
        
        conditions_exist = torch.any(torch.matmul(valid_labels, condition_masks.t()), dim=0)
        first_visit_conditions = torch.any(first_visit_labels[patient_idx].unsqueeze(0).float() * condition_masks, dim=1)
        valid_conditions = conditions_exist & ~first_visit_conditions
        
        if not torch.any(valid_conditions):
            continue
        
        condition_pred_seqs = valid_preds #torch.matmul(valid_preds, condition_masks.t()) > 0  # convert back to bool
        condition_label_seqs = torch.matmul(valid_labels, condition_masks.t()) > 0  # convert back to bool
        
        gt_days = torch.full((num_conditions,), -1.0, device=device)
        pred_days = torch.full((num_conditions,), -1.0, device=device)
        
        for c_idx in torch.where(valid_conditions)[0]:
            gt_idxs = torch.where(condition_label_seqs[:, c_idx])[0]
            pred_idxs = torch.where(condition_pred_seqs[:, c_idx])[0]
            
            if len(gt_idxs) > 0:
                gt_days[c_idx] = valid_days[gt_idxs[0]]
                if len(pred_idxs) > 0:
                    pred_days[c_idx] = valid_days[pred_idxs[0]]
        
        valid_mask = valid_conditions & (gt_days != -1)
        has_pred = pred_days != -1
        on_time = has_pred & (pred_days <= gt_days)
        
        for c_idx in torch.where(valid_mask)[0]:
            batch_results[c_idx.item()]["gt_days"].append(float(gt_days[c_idx].item()))
            batch_results[c_idx.item()]["pred_days"].append(float(pred_days[c_idx].item()))
            batch_results[c_idx.item()]["on_time"].append(bool(on_time[c_idx].item()))
    
    return batch_results

@torch.no_grad()
def evaluate_model(model, loader, condition_indices, opt_thresh, rank, device, bs=8, approach="thresholding"):
    model.eval()
    
    # Create condition to index mapping
    condition_to_idx = {condition: idx for idx, condition in enumerate(condition_indices.keys())}
    num_conditions = len(condition_indices)
    opt_thresh_tensor = torch.from_numpy(opt_thresh).to(device)

    
    result_ontime = {
        condition: {
            "on_time": [], 
            "pred_day": [], 
            "gt_day": []
        } 
        for condition in condition_indices.keys()
    }
    
    # Initialize metrics for each condition
    full_TP = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_TN = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FP = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FN = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    
    # Create condition masks for all possible indices
    N_C = next(iter(loader))['labels'].shape[-1]  # Get total number of classes from first batch
    condition_masks = torch.zeros((num_conditions, N_C), dtype=torch.bool, device=device)
    for idx, (condition, indices) in enumerate(condition_indices.items()):
        condition_masks[idx, indices] = True
            
    print(f"GPU {rank} starts evaluation with {device}")
    
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        token_arr = batch['token_arr']
        days_embed_arr = batch['days_embed_arr']
        labels = batch['labels']
        pred_mask = batch['pred_mask']
        pad_mask = batch['pad_mask']
        attention_bias = batch['attention_bias']
        first_visit_labels = batch['first_visit_labels']
        
        batch_size = token_arr.shape[0]
        
        for start_idx in range(0, batch_size, bs):
            end_idx = min(start_idx + bs, batch_size)
            
            sub_batch = {
                k: v[start_idx:end_idx] if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            logits, _, _, _ = model(
                sub_batch['token_arr'], 
                sub_batch['days_embed_arr'],
                sub_batch['labels'].float(),
                sub_batch['pred_mask'],
                sub_batch['pad_mask'],
                sub_batch['attention_bias']
            )
            
            batch_results = evaluate_on_time_parallel(
                logits >= opt_thresh_tensor,
                sub_batch['labels'].float(),
                sub_batch['pred_mask'],
                sub_batch['first_visit_labels'],
                sub_batch['days_embed_arr'],
                condition_masks,
                device
            )
            
            for condition_idx, values in batch_results.items():
                condition_name = list(condition_indices.keys())[condition_idx]
                result_ontime[condition_name]["gt_day"].extend(values["gt_days"])
                result_ontime[condition_name]["pred_day"].extend(values["pred_days"])
                result_ontime[condition_name]["on_time"].extend(values["on_time"])

            logits = logits.reshape(-1, N_C)
            sub_labels = sub_batch['labels'].reshape(-1, N_C)
            sub_pred_mask = sub_batch['pred_mask'].reshape(-1)
            
            logits_masked = torch.sigmoid(logits[sub_pred_mask])
            labels_masked = sub_labels[sub_pred_mask]
            
            # First apply threshold to get binary predictions
            predicted_indices = (logits_masked >= opt_thresh_tensor).float()
           
            condition_preds = torch.any(predicted_indices.unsqueeze(1) * condition_masks, dim=2).float()
            condition_labels = torch.any(labels_masked.unsqueeze(1) * condition_masks, dim=2).float()

            TP = torch.sum((condition_preds == 1) & (condition_labels == 1), dim=0)
            TN = torch.sum((condition_preds == 0) & (condition_labels == 0), dim=0)
            FP = torch.sum((condition_preds == 1) & (condition_labels == 0), dim=0)
            FN = torch.sum((condition_preds == 0) & (condition_labels == 1), dim=0)
            
            # Update full metrics
            full_TP += TP
            full_TN += TN
            full_FP += FP
            full_FN += FN
            
            # Clean up memory
            del logits, sub_labels, sub_pred_mask, logits_masked, labels_masked, predicted_indices
            del sub_batch #sub_token_arr, sub_days_embed_arr, sub_pad_mask, sub_attention_bias
            torch.cuda.empty_cache()
        
        # Clean up batch memory
        del batch, token_arr, days_embed_arr, labels, pred_mask, pad_mask, attention_bias
        torch.cuda.empty_cache()


    
    print(f"GPU {rank} finished processing all batches")
    
    # Synchronize before all_reduce
    torch.cuda.synchronize(device)
    dist.barrier()
    
    # All-reduce operations
    for tensor in [full_TP, full_TN, full_FP, full_FN]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Final synchronization
    dist.barrier()
    
    if rank == 0:
        return (full_TP.cpu().numpy(), full_TN.cpu().numpy(), 
                full_FP.cpu().numpy(), full_FN.cpu().numpy(), 
                condition_to_idx,result_ontime)
    else:
        return None, None, None, None, None, result_ontime
    

@torch.no_grad()
def evaluate_model_agg(model, loader, condition_indices, opt_thresh, rank, device, bs=8, approach="thresholding", agg_method="sum"):
    model.eval()
    num_conditions = len(condition_indices)
    opt_thresh_tensor = torch.from_numpy(opt_thresh).to(device)
    print(f"GPU {rank} starts evaluation with {device}")
    
    # Initialize metrics
    result_ontime = {
        condition: {
            "on_time": [], 
            "pred_day": [], 
            "gt_day": []
        } 
        for condition in condition_indices.keys()
    }
    full_TP = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_TN = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FP = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FN = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    
    # Create condition masks
    N_C = next(iter(loader))['labels'].shape[-1]
    condition_masks = torch.zeros((num_conditions, N_C), dtype=torch.bool, device=device)
    for idx, (condition, indices) in enumerate(condition_indices.items()):
        condition_masks[idx, indices] = True
    
    for batch in tqdm(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        token_arr = batch['token_arr']
        days_embed_arr = batch['days_embed_arr']
        labels = batch['labels']
        pred_mask = batch['pred_mask']
        pad_mask = batch['pad_mask']
        attention_bias = batch['attention_bias']
        
        
        
        batch_size = token_arr.shape[0]
        
        for start_idx in range(0, batch_size, bs):
            end_idx = min(start_idx + bs, batch_size)
            
            sub_batch = {k: v[start_idx:end_idx] for k, v in batch.items()}
            if 'patient_ids' not in batch:
                sub_batch['patient_ids'] = None
            
            logits, _, _, _ = model(
                sub_batch['token_arr'],
                sub_batch['days_embed_arr'],
                sub_batch['labels'].float(),
                sub_batch['pred_mask'],
                sub_batch['pad_mask'],
                sub_batch['attention_bias'],
                patient_ids = sub_batch['patient_ids']
            )
            
            # Aggregate logits per condition
            agg_logits = torch.zeros((*logits.shape[:2], num_conditions), device=device)

            for cond_idx in range(num_conditions):
                mask = condition_masks[cond_idx]
                agg_logits[:, :, cond_idx] = aggregate_logits(logits, mask, method=agg_method, dim=2)
            
            batch_results = evaluate_on_time_parallel(
                agg_logits >= opt_thresh_tensor,
                sub_batch['labels'].float(),
                sub_batch['pred_mask'],
                sub_batch['first_visit_labels'],
                sub_batch['days_embed_arr'],
                condition_masks,
                device
            )
            for condition_idx, values in batch_results.items():
                condition_name = list(condition_indices.keys())[condition_idx]
                result_ontime[condition_name]["gt_day"].extend(values["gt_days"])
                result_ontime[condition_name]["pred_day"].extend(values["pred_days"])
                result_ontime[condition_name]["on_time"].extend(values["on_time"])

            
            # Reshape and mask
            agg_logits = agg_logits.reshape(-1, num_conditions)
            sub_labels = sub_batch['labels'].reshape(-1, N_C)
            sub_pred_mask = sub_batch['pred_mask'].reshape(-1)
            
            agg_logits_masked = agg_logits[sub_pred_mask]
            labels_aggregated = torch.zeros((sub_labels.size(0), num_conditions), device=device)
            for cond_idx in range(num_conditions):
                mask = condition_masks[cond_idx]
                labels_aggregated[:, cond_idx] = (sub_labels[:, mask].sum(dim=1) > 0).float()
            labels_aggregated_masked = labels_aggregated[sub_pred_mask]
            
            # Apply threshold
            predicted = (agg_logits_masked >= opt_thresh_tensor).float()
            
            # Compute metrics
            TP = torch.sum((predicted == 1) & (labels_aggregated_masked == 1), dim=0)
            TN = torch.sum((predicted == 0) & (labels_aggregated_masked == 0), dim=0)
            FP = torch.sum((predicted == 1) & (labels_aggregated_masked == 0), dim=0)
            FN = torch.sum((predicted == 0) & (labels_aggregated_masked == 1), dim=0)
            
            full_TP += TP
            full_TN += TN
            full_FP += FP
            full_FN += FN

            del logits, sub_labels, sub_pred_mask, agg_logits_masked, labels_aggregated_masked, predicted
            del sub_batch #sub_token_arr, sub_days_embed_arr, sub_pad_mask, sub_attention_bias
            torch.cuda.empty_cache()

        # Clean up batch memory
        del batch, token_arr, days_embed_arr, labels, pred_mask, pad_mask, attention_bias
        torch.cuda.empty_cache()
    
    # Synchronize metrics across GPUs
    print(f"GPU {rank} finished processing all batches")
    torch.cuda.synchronize(device)
    dist.barrier() 
    for tensor in [full_TP, full_TN, full_FP, full_FN]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        return (full_TP.cpu().numpy(), full_TN.cpu().numpy(),
                full_FP.cpu().numpy(), full_FN.cpu().numpy(),
                {condition: idx for idx, condition in enumerate(condition_indices.keys())}, result_ontime)
    else:
        return None, None, None, None, None, result_ontime
    

@torch.no_grad()
def evaluate_model_agg_time_gap(model, loader, condition_indices, opt_thresh, rank, device, bs=8, approach="thresholding", short_term_threshold=365, agg_method="sum"):
    model.eval()
    num_conditions = len(condition_indices)
    opt_thresh_tensor = torch.from_numpy(opt_thresh).to(device)
    print(f"GPU {rank} starts evaluation with {device}")
    
    # Initialize metrics
    result_ontime = {
        condition: {
            "on_time": [], 
            "pred_day": [], 
            "gt_day": []
        } 
        for condition in condition_indices.keys()
    }
    full_TP = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_TN = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FP = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FN = torch.zeros(num_conditions, dtype=torch.float32, device=device)

    full_TP_short = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_TN_short = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FP_short = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FN_short = torch.zeros(num_conditions, dtype=torch.float32, device=device)

    full_TP_long = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_TN_long = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FP_long = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    full_FN_long = torch.zeros(num_conditions, dtype=torch.float32, device=device)
    
    # Create condition masks
    N_C = next(iter(loader))['labels'].shape[-1]
    condition_masks = torch.zeros((num_conditions, N_C), dtype=torch.bool, device=device)
    for idx, (condition, indices) in enumerate(condition_indices.items()):
        condition_masks[idx, indices] = True
    
    for batch in tqdm(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        token_arr = batch['token_arr']
        days_embed_arr = batch['days_embed_arr']
        labels = batch['labels']
        pred_mask = batch['pred_mask']
        pad_mask = batch['pad_mask']
        attention_bias = batch['attention_bias']
        
        
        
        batch_size = token_arr.shape[0]
        
        for start_idx in range(0, batch_size, bs):
            end_idx = min(start_idx + bs, batch_size)
            
            sub_batch = {k: v[start_idx:end_idx] for k, v in batch.items()}
            if 'patient_ids' not in batch:
                sub_batch['patient_ids'] = None
            
            logits, _, _, _ = model(
                sub_batch['token_arr'],
                sub_batch['days_embed_arr'],
                sub_batch['labels'].float(),
                sub_batch['pred_mask'],
                sub_batch['pad_mask'],
                sub_batch['attention_bias'],
                patient_ids = sub_batch['patient_ids']
            )
            
            # Aggregate logits per condition
            agg_logits = torch.zeros((*logits.shape[:2], num_conditions), device=device)

            for cond_idx in range(num_conditions):
                mask = condition_masks[cond_idx]
                agg_logits[:, :, cond_idx] = aggregate_logits(logits, mask, method=agg_method, dim=2)
            
            batch_results = evaluate_on_time_parallel(
                agg_logits >= opt_thresh_tensor,
                sub_batch['labels'].float(),
                sub_batch['pred_mask'],
                sub_batch['first_visit_labels'],
                sub_batch['days_embed_arr'],
                condition_masks,
                device
            )
            for condition_idx, values in batch_results.items():
                condition_name = list(condition_indices.keys())[condition_idx]
                result_ontime[condition_name]["gt_day"].extend(values["gt_days"])
                result_ontime[condition_name]["pred_day"].extend(values["pred_days"])
                result_ontime[condition_name]["on_time"].extend(values["on_time"])
            
            # Reshape and mask
            agg_logits = agg_logits.reshape(-1, num_conditions)
            sub_labels = sub_batch['labels'].reshape(-1, N_C)
            sub_pred_mask = sub_batch['pred_mask'].reshape(-1)
            
            agg_logits_masked = agg_logits[sub_pred_mask]
            labels_aggregated = torch.zeros((sub_labels.size(0), num_conditions), device=device)
            for cond_idx in range(num_conditions):
                mask = condition_masks[cond_idx]
                labels_aggregated[:, cond_idx] = (sub_labels[:, mask].sum(dim=1) > 0).float()
            labels_aggregated_masked = labels_aggregated[sub_pred_mask]

            # After getting aggregated logits and labels
            days_embed_flat = sub_batch['days_embed_arr'].reshape(-1)
            days_masked = days_embed_flat[sub_pred_mask]
            days_masked = torch.diff(days_masked, prepend=torch.tensor([0.0], device=device))

            short_term_mask = (days_masked <= short_term_threshold)
            long_term_mask = (days_masked > short_term_threshold)
            
            # Apply threshold
            predicted = (agg_logits_masked >= opt_thresh_tensor).float()
            
            # Compute metrics
            TP = torch.sum((predicted == 1) & (labels_aggregated_masked == 1), dim=0)
            TN = torch.sum((predicted == 0) & (labels_aggregated_masked == 0), dim=0)
            FP = torch.sum((predicted == 1) & (labels_aggregated_masked == 0), dim=0)
            FN = torch.sum((predicted == 0) & (labels_aggregated_masked == 1), dim=0)
            
            full_TP += TP
            full_TN += TN
            full_FP += FP
            full_FN += FN

            # Compute metrics separately for each cohort
            TP_short = ((predicted[short_term_mask] == 1) & (labels_aggregated_masked[short_term_mask] == 1)).sum(0)
            TN_short = ((predicted[short_term_mask] == 0) & (labels_aggregated_masked[short_term_mask] == 0)).sum(0)
            FP_short = ((predicted[short_term_mask] == 1) & (labels_aggregated_masked[short_term_mask] == 0)).sum(0)
            FN_short = ((predicted[short_term_mask] == 0) & (labels_aggregated_masked[short_term_mask] == 1)).sum(0)

            TP_long = ((predicted[long_term_mask] == 1) & (labels_aggregated_masked[long_term_mask] == 1)).sum(0)
            TN_long = ((predicted[long_term_mask] == 0) & (labels_aggregated_masked[long_term_mask] == 0)).sum(0)
            FP_long = ((predicted[long_term_mask] == 1) & (labels_aggregated_masked[long_term_mask] == 0)).sum(0)
            FN_long = ((predicted[long_term_mask] == 0) & (labels_aggregated_masked[long_term_mask] == 1)).sum(0)

            # Accumulate metrics
            full_TP_short += TP_short
            full_TN_short += TN_short
            full_FP_short += FP_short
            full_FN_short += FN_short

            full_TP_long += TP_long
            full_TN_long += TN_long
            full_FP_long += FP_long
            full_FN_long += FN_long

            del logits, sub_labels, sub_pred_mask, agg_logits_masked, labels_aggregated_masked, predicted, days_masked, short_term_mask, long_term_mask
            del sub_batch #sub_token_arr, sub_days_embed_arr, sub_pad_mask, sub_attention_bias
            torch.cuda.empty_cache()

        # Clean up batch memory
        del batch, token_arr, days_embed_arr, labels, pred_mask, pad_mask, attention_bias
        torch.cuda.empty_cache()
    
    # Synchronize metrics across GPUs
    print(f"GPU {rank} finished processing all batches")
    torch.cuda.synchronize(device)
    dist.barrier() 
    for tensor in [full_TP, full_TN, full_FP, full_FN]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Replace existing all-reduce with
    for tensor in [
        full_TP_short, full_TN_short, full_FP_short, full_FN_short,
        full_TP_long, full_TN_long, full_FP_long, full_FN_long
    ]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    

    if rank == 0:
        return (full_TP.cpu().numpy(), full_TN.cpu().numpy(),
                full_FP.cpu().numpy(), full_FN.cpu().numpy(),
                full_TP_short.cpu().numpy(), full_TN_short.cpu().numpy(),
                full_FP_short.cpu().numpy(), full_FN_short.cpu().numpy(),
                full_TP_long.cpu().numpy(), full_TN_long.cpu().numpy(),
                full_FP_long.cpu().numpy(), full_FN_long.cpu().numpy(),
                {condition: idx for idx, condition in enumerate(condition_indices.keys())}, result_ontime)
    else:
        return None, None, None, None, None, None, None, None, None, None, None, None, None, result_ontime


def load_latest_checkpoint(out_dir, device, iteration = None):
    if iteration is not None:
        checkpoint_files = glob.glob(os.path.join(out_dir, f'ckpt-{iteration}.pt'))
    else:
        checkpoint_files = glob.glob(os.path.join(out_dir, 'ckpt-*.pt'))
    
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

def main():
    dist.init_process_group(backend="nccl")

    
    local_rank = int(os.environ["LOCAL_RANK"])
    ddp_rank = int(os.environ['RANK'])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Process {local_rank+1}/{world_size} starting")
    
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0


    args = parse_args()


    folder = args.folder
    batch_size = 8
    
    checkpoint, checkpoint_model_args, config = load_latest_checkpoint(args.out_dir, device, args.iteration)
    print("Latest checkpoint loaded successfully")

    agg_labels = config.get('agg_labels', False)
    multiwin = config.get('multiwin', False)

    model_args = {} 
    full_args = checkpoint_model_args
    keys_to_remove = [key for key in model_args if key not in GPTCONFIG_PARAMS]
    for key in GPTCONFIG_PARAMS:
        model_args[key] =  full_args[key]


    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.to(device)

    model = DDP(model, device_ids=[local_rank])

    torch.cuda.synchronize(device)
    dist.barrier()
    print(f"GPU {local_rank} model initialized and ready")

    if args.all_conditions:
        cond_file = os.path.join(args.folder, "conditions2inds_ICLR_plus_one.json")
    else:
        cond_file = os.path.join(args.folder, "conditions2inds_pcancer_dementia_update.json")
    with open(cond_file, "rb") as f:
        condition_indices = json.load(f)


    opt_thresh_name = args.opt_thresh_name


    
    AUROC, AUPRC = None, None


    opt_thresh_path = f"{args.out_dir}/{opt_thresh_name}_{args.threshold_method}_all_condition_{args.all_conditions}_{args.iteration}.npy"
    eval_inds = np.load(os.path.join(args.folder, "data_files", f"loss_mask_{args.perc}_perc.npy"))
    eval_inds = np.append([False], eval_inds)

    print(opt_thresh_path)

    if not os.path.exists(opt_thresh_path):
        print("Computing aggregated thresholds...")
        val_df = pd.read_csv(f"{args.csv_folder}/val.csv")
        val_dataset = UnifiedSeqEHRDataset(val_df, max_length=model_args["block_size"], model='GPT', folder=folder, age=None, train=False, agg_labels=agg_labels, multiwin=multiwin)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=custom_collate, num_workers=8)
        
        all_agg_logits, all_agg_labels = process_batches_aggregated(model, val_dataloader, condition_indices, local_rank, device, batch_size, world_size, args.num_batches//world_size, 100, args.agg_method)
        
        if master_process:
            if args.threshold_method == 'f1':
                opt_thresh, pr_curves = get_optimal_thresholds(all_agg_logits, all_agg_labels)
                # Save PR curves for F1-based thresholding
                pr_curves_path = f"{args.out_dir}/{opt_thresh_name}_{args.threshold_method}_{args.agg_method}_all_condition_{args.all_conditions}_{args.iteration}_pr_curves.json"
                with open(pr_curves_path, 'w') as f:
                    json.dump(pr_curves, f)
            elif args.threshold_method == 'quantile':
                opt_thresh = get_quantile_thresholds(all_agg_logits, all_agg_labels)
            np.save(opt_thresh_path, opt_thresh)

            # # opt_thresh = get_optimal_thresholds(all_agg_logits, all_agg_labels)
            # opt_thresh, pr_curves = get_optimal_thresholds(all_agg_logits, all_agg_labels)
            # np.save(opt_thresh_path, opt_thresh)
            # # Save PR curves as JSON since they're now dictionaries
            # pr_curves_path = f"{args.out_dir}/{opt_thresh_name}_{args.agg_method}_{args.iteration}_pr_curves.json"
            # with open(pr_curves_path, 'w') as f:
            #     json.dump(pr_curves, f)

            if args.save_logits_labels:
                logits_save_path = os.path.join(args.out_dir, f"all_agg_logits_{args.agg_method}_all_condition_{args.all_conditions}_{args.iteration}.npy")
                labels_save_path = os.path.join(args.out_dir, f"all_agg_labels_{args.agg_method}_all_condition_{args.all_conditions}_{args.iteration}.npy")
                np.save(logits_save_path, all_agg_logits)
                np.save(labels_save_path, all_agg_labels)
                print(f"Aggregated logits saved to {logits_save_path}")
                print(f"Aggregated labels saved to {labels_save_path}")

            print("Aggregated thresholds computed and saved.")

            AUROC = roc_auc_score(all_agg_labels, all_agg_logits, average=None)
            AUPRC = average_precision_score(all_agg_labels, all_agg_logits, average=None)
            print("AUROC and AUPRC computed.")
        dist.barrier()
        
        
        opt_thresh = np.load(opt_thresh_path)
    else:
        opt_thresh = np.load(opt_thresh_path)
    

    
    mode = args.mode
        
    df = pd.read_csv(f"{args.csv_folder}/{mode}.csv")
    if args.test_patients !=-1:
        df = df.sample(n=args.test_patients,replace= False, random_state = args.seed).reset_index(drop=True)
    dataset = UnifiedSeqEHRDataset(df, max_length=model_args["block_size"], model='GPT', folder=folder, age=None, train=False,agg_labels=agg_labels, multiwin=multiwin)
    sampler = DistributedSampler(dataset, shuffle = False)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                        sampler=sampler, collate_fn=custom_collate, num_workers=8)



    dist.barrier()

    print(f"GPU {local_rank} has received optimal thresholds")

    print(f"GPU {local_rank} is gonna start eval")


    full_TP, full_TN, full_FP, full_FN, full_TP_short, full_TN_short, full_FP_short, full_FN_short, full_TP_long, full_TN_long, full_FP_long, full_FN_long, condition_to_idx, result_ontime= evaluate_model_agg_time_gap(model, dataloader, condition_indices, opt_thresh, ddp_rank, device, bs=batch_size, approach="thresholding", short_term_threshold = args.short_term_threshold, agg_method=args.agg_method)
    save_name = f"aggregate_conditions_{opt_thresh_name}_{args.agg_method}_{args.threshold_method}_all_condition_{args.all_conditions}"

    save_path = os.path.join(args.out_dir, f"result_ontime_{ddp_rank}_{opt_thresh_name}_{args.agg_method}_{args.threshold_method}_all_condition_{args.all_conditions}_{args.iteration}.json")
    with open(save_path, 'w') as f:
        json.dump(result_ontime, f)

    
    if master_process:
        np.save(f"{args.out_dir}/full_TP_{mode}_{save_name}_{args.iteration}.npy", full_TP)
        np.save(f"{args.out_dir}/full_TN_{mode}_{save_name}_{args.iteration}.npy", full_TN)
        np.save(f"{args.out_dir}/full_FP_{mode}_{save_name}_{args.iteration}.npy", full_FP)
        np.save(f"{args.out_dir}/full_FN_{mode}_{save_name}_{args.iteration}.npy", full_FN)

        precision, recall, specificity, f1_score, total_cases, total_actual_postive, total_actual_negative = calculate_metrics(full_TP, full_TN, full_FP, full_FN)
        
        results = pd.DataFrame({
            "Aggregator_Condition": list(condition_indices.keys()),
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1_Score': f1_score,
            'Total_Cases': total_cases,
            'Total_Actual_Positive': total_actual_postive,
            'Total_Actual_Negative': total_actual_negative,
            'TP': full_TP,
            'TN': full_TN,
            'FP': full_FP,
            'FN': full_FN
        })

        if AUROC is not None and AUPRC is not None:
            results['AUROC'] = AUROC
            results['AUPRC'] = AUPRC
        
        precision_short, recall_short, specificity_short, f1_short, total_cases_short, total_actual_postive_short, total_actual_negative_short = calculate_metrics(
            full_TP_short, full_TN_short, full_FP_short, full_FN_short
        )
        results_short = pd.DataFrame({
            "Aggregator_Condition": list(condition_indices.keys()),
            'Precision': precision_short,
            'Recall': recall_short,
            'Specificity': specificity_short,
            'F1_Score': f1_short,
            'Total_Cases': total_cases_short,
            'Total_Actual_Positive': total_actual_postive_short,
            'Total_Actual_Negative': total_actual_negative_short,
            'TP': full_TP_short,
            'TN': full_TN_short,
            'FP': full_FP_short,
            'FN': full_FN_short
        })
        
        # Long-term results
        precision_long, recall_long, specificity_long, f1_long, total_cases_long, total_actual_postive_long, total_actual_negative_long = calculate_metrics(
            full_TP_long, full_TN_long, full_FP_long, full_FN_long
        )
        results_long = pd.DataFrame({
            "Aggregator_Condition": list(condition_indices.keys()),
            'Precision': precision_long,
            'Recall': recall_long,
            'Specificity': specificity_long,
            'F1_Score': f1_long,
            'Total_Cases': total_cases_long,
            'Total_Actual_Positive': total_actual_postive_long,
            'Total_Actual_Negative': total_actual_negative_long,
            'TP': full_TP_long,
            'TN': full_TN_long,
            'FP': full_FP_long,
            'FN': full_FN_long
        })

        idx_to_condition = {str(idx): condition for condition, idx in condition_to_idx.items()}
        
        save_path = os.path.join(args.out_dir, f"idx_to_condition_{opt_thresh_name}_all_condition_{args.all_conditions}.json")
        with open(save_path, 'w') as f:
            json.dump(idx_to_condition, f, indent=4)
        results.to_csv(f"{args.out_dir}/metrics_{mode}_{save_name}_{opt_thresh_name}_{args.iteration}.csv", index=False)

        results_short.to_csv(f"{args.out_dir}/metrics_short_term_{mode}_{save_name}_{opt_thresh_name}_{args.short_term_threshold}_{args.iteration}.csv")
        results_long.to_csv(f"{args.out_dir}/metrics_long_term_{mode}_{save_name}_{opt_thresh_name}_{args.short_term_threshold}_{args.iteration}.csv")    
        

        print(f"Evaluation for {mode} set completed and results saved.")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()



