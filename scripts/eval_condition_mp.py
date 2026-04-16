import sys
import os
os.environ["NCCL_TIMEOUT"] = "1200"  # 20 minutes instead of 10

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
from glob import glob
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from raven.model import GPTConfig, GPT
from raven.dataset import custom_collate
from data_processing.condition_helper import SeqCLSDataset, aggregate_logits, GPTCONFIG_PARAMS

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Condition Model Evaluation')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for model checkpoints and results')
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--condition_indices_path', type=str, required=True, help='Path to condition indices json file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--agg_method', type=str, default='sum', choices=['sum', 'noisy_or', 'max', 'logsumexp', 'geometric_mean', 'attention'], help='Method for aggregating logits')
    parser.add_argument('--condition_name', type=str, default='Dementia', help='Condition to evaluate')
    parser.add_argument('--multiwin', action='store_true', help='Use multi-window approach')
    parser.add_argument('--time_horizon', type=int, default=730, help='Time horizon in days (default: 365*2)')
    parser.add_argument('--n_gather', type=int, default=100, help='Frequency of gathering results across GPUs')
    parser.add_argument('--num_patients', type=int, default=512, help='No of patiemts to use')
    parser.add_argument('--approach', type=str, default='direct', choices=['direct', 'intermediate_pred'], help='Method for multi horizon prediction')
    parser.add_argument('--intermediate_horizon', type=int, default=365, help='intermediate horizon to use')
    parser.add_argument('--intermediate_agg', type=str, default='mean', choices=['mean', 'max'], help='Method for aggregation multi horizon prediction')
    
    
    return parser.parse_args()

def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """
    Find the optimal threshold to maximize a given metric.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores
        metric: Metric to optimize ('f1', 'precision', or 'recall')
        
    Returns:
        optimal_threshold: The threshold that maximizes the metric
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    if metric == 'f1':
        optimal_idx = np.argmax(f1_scores)
    elif metric == 'precision':
        optimal_idx = np.argmax(precisions)
    elif metric == 'recall':
        optimal_idx = np.argmax(recalls)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Handle edge case where the optimal threshold is at the end of the array
    if optimal_idx >= len(thresholds):
        return 1.0
    
    return thresholds[optimal_idx]

def gather_variable_results(tensor, rank, world_size):
    """
    Gather tensors of variable sizes from all GPUs.
    
    Args:
        tensor: Local tensor to gather
        rank: Rank of current process
        world_size: Total number of processes
        
    Returns:
        result: Gathered tensor on rank 0, None on other ranks
    """
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [size.item() for size in all_sizes]
    
    max_size = max(all_sizes)
    
    padded_tensor = torch.zeros((max_size,) + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:tensor.shape[0]] = tensor
    
    gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded_tensor)
    
    if rank == 0:
        result = torch.cat([tensor[:size] for tensor, size in zip(gathered_tensors, all_sizes)])
        return result.cpu()
    else:
        return None

@torch.no_grad()
def evaluate_model_distributed(model, dataloader, device, condition_indices, condition_name="Dementia", 
                              agg_method="sum", threshold=0.5, compute_metrics=True, rank=0, world_size=1, n_gather=100, 
                              approach="direct", intermediate_agg="mean", save_sample_logits=True, output_path=None):
    """
    Evaluate the model on a dataset in a distributed setting.
    
    Args:
        model: The GPT model
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        condition_indices: Dictionary mapping condition names to indices
        condition_name: The condition to evaluate
        agg_method: Method for aggregating logits
        threshold: Classification threshold
        compute_metrics: Whether to compute and return metrics
        rank: Process rank
        world_size: Total number of processes
        n_gather: How often to gather results across processes
        
    Returns:
        all_labels: Ground truth labels (on rank 0 only)
        all_scores: Predicted scores (on rank 0 only)
        metrics: Dictionary of evaluation metrics (if compute_metrics=True, on rank 0 only)
    """
    model.eval()
    batch_logits = []
    batch_labels = []
    
    # Get condition indices for the specific condition
    cond_ind = condition_indices[condition_name]
    N_C = 42343  # Total number of classes
    
    # Create mask for the condition
    condition_mask = torch.zeros((N_C), dtype=torch.bool, device=device)
    condition_mask[cond_ind] = True
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Handle patient_ids if present (for multiwin)
            patient_ids = batch.get('patient_ids', None)

            if approach =="direct":

            
                # Forward pass
                logits, _, _, _ = model(
                    batch['token_arr'],
                    batch['days_embed_arr'],
                    None,
                    batch['pred_mask'],
                    batch['pad_mask'],
                    batch['attention_bias'],
                    patient_ids=patient_ids
                )
                
                # Get masked logits
                masked_logits = logits[batch['pred_mask']]
                
                # Aggregate logits for the condition
                agg_logits = aggregate_logits(masked_logits, condition_mask, method=agg_method, dim=1)
            else:
                full_logits = []

                for itr in range(batch['days_embed_arr'].shape[1]):
                    days_embed_arr = batch['days_embed_arr'][:,itr,:]
                    
                    logits, _, _, _ = model(
                        batch['token_arr'],
                        days_embed_arr,
                        None,
                        batch['pred_mask'],
                        batch['pad_mask'],
                        batch['attention_bias'],
                        patient_ids=patient_ids
                    )
                    
                    # Get masked logits
                    masked_logits = logits[batch['pred_mask']]
                    
                    # Aggregate logits for the condition
                    agg_logits = aggregate_logits(masked_logits, condition_mask, method=agg_method, dim=1)
                    full_logits.append(agg_logits)

                full_logits = torch.stack(full_logits, dim=0)  # Stack along first dimension
                if save_sample_logits and output_path:
                    # Create sample_logits directory if it doesn't exist
                    sample_logits_dir = os.path.join(output_path, 'sample_logits')
                    os.makedirs(sample_logits_dir, exist_ok=True)
                    
                    # Save the full logits tensor with rank and batch index in the filename
                    logits_filename = os.path.join(sample_logits_dir, f'logits_{rank}_{batch_idx}.pt')
                    
                    # Save the logits and corresponding labels as a dictionary
                    save_data = {
                        'logits': full_logits.cpu(),
                        'labels': batch['labels'].cpu(),
                        'batch_idx': batch_idx,
                        'rank': rank
                    }
                    
                    # Also save patient IDs if available
                    if patient_ids is not None:
                        if isinstance(patient_ids, torch.Tensor):
                            save_data['patient_ids'] = patient_ids.cpu()
                        else:
                            save_data['patient_ids'] = patient_ids
                    
                    torch.save(save_data, logits_filename)

                if intermediate_agg == "mean":
                    agg_logits = torch.mean(full_logits, dim=0)  # Mean along horizon dimension
                else:
                    agg_logits = torch.max(full_logits, dim=0).values  # Max along horizon dimension

            # Collect scores and labels
            batch_logits.append(agg_logits)
            batch_labels.append(batch['labels'])
            
            # Periodically gather results across GPUs
            if (batch_idx + 1) % n_gather == 0 or batch_idx == len(dataloader) - 1:
                batch_logits_tensor = torch.cat(batch_logits)
                batch_labels_tensor = torch.cat(batch_labels)
                
                # Synchronize before gathering
                dist.barrier()
                
                # Gather results from all GPUs
                gathered_logits = gather_variable_results(batch_logits_tensor, rank, world_size)
                gathered_labels = gather_variable_results(batch_labels_tensor, rank, world_size)
                
                if rank == 0:
                    if 'all_logits' not in locals():
                        all_logits = gathered_logits.cpu().numpy()
                        all_labels = gathered_labels.cpu().numpy()
                    else:
                        all_logits = np.concatenate([all_logits, gathered_logits.cpu().numpy()])
                        all_labels = np.concatenate([all_labels, gathered_labels.cpu().numpy()])
                
                # Clear batch data
                batch_logits = []
                batch_labels = []
                
                # Clean up memory
                del gathered_logits, gathered_labels, batch_logits_tensor, batch_labels_tensor
                torch.cuda.empty_cache()
    
    # Final barrier to ensure all processes have finished
    dist.barrier()
    
    # Compute metrics if requested (only on rank 0)
    if compute_metrics and rank == 0:
        metrics = {}
        
        # Apply threshold
        predictions = (all_logits >= threshold).astype(int)
        
        # Calculate metrics
        metrics['precision'] = precision_score(all_labels, predictions)
        metrics['recall'] = recall_score(all_labels, predictions)
        metrics['f1'] = f1_score(all_labels, predictions)
        metrics['auroc'] = roc_auc_score(all_labels, all_logits)
        metrics['auprc'] = average_precision_score(all_labels, all_logits)
        
        return all_labels, all_logits, metrics
    
    if rank == 0:
        return all_labels, all_logits, None
    else:
        return None, None, None

def plot_precision_recall_curve(val_labels, val_scores, test_labels, test_scores, 
                                optimal_threshold, output_path):
    """
    Plot and save precision-recall curves for both validation and test sets.
    Also save the raw data for reproducing the plot.
    
    Args:
        val_labels: Validation ground truth labels
        val_scores: Validation predicted scores
        test_labels: Test ground truth labels
        test_scores: Test predicted scores
        optimal_threshold: Optimal threshold from validation set
        output_path: Directory to save the plot and data
    """
    # Ensure arrays are 1-dimensional
    val_labels = np.ravel(val_labels)
    val_scores = np.ravel(val_scores)
    test_labels = np.ravel(test_labels)
    test_scores = np.ravel(test_scores)
    
    # Compute precision-recall curves
    val_precision, val_recall, val_thresholds = precision_recall_curve(val_labels, val_scores)
    test_precision, test_recall, test_thresholds = precision_recall_curve(test_labels, test_scores)
    
    # Compute AUPRC
    val_auprc = auc(val_recall, val_precision)
    test_auprc = auc(test_recall, test_precision)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot validation curve
    plt.plot(val_recall, val_precision, label=f'Validation (AUPRC={val_auprc:.3f})', linewidth=2)
    
    # Plot test curve
    plt.plot(test_recall, test_precision, label=f'Test (AUPRC={test_auprc:.3f})', linewidth=2)
    
    # Add optimal threshold point
    # Find the precision and recall at the optimal threshold
    val_pred = (val_scores >= optimal_threshold).astype(int)
    optimal_precision = precision_score(val_labels, val_pred)
    optimal_recall = recall_score(val_labels, val_pred)
    
    plt.scatter(optimal_recall, optimal_precision, c='red', s=100, 
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    # Add labels and title
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'precision_recall_curve.pdf'), bbox_inches='tight')
    plt.close()
    
    # Save raw data for reproducing the plot
    raw_data = {
        'validation': {
            'precision': val_precision.tolist(),
            'recall': val_recall.tolist(),
            'thresholds': val_thresholds.tolist() if len(val_thresholds) > 0 else [],
            'auprc': float(val_auprc),
            'scores': val_scores.tolist(),
            'labels': val_labels.tolist()
        },
        'test': {
            'precision': test_precision.tolist(),
            'recall': test_recall.tolist(),
            'thresholds': test_thresholds.tolist() if len(test_thresholds) > 0 else [],
            'auprc': float(test_auprc),
            'scores': test_scores.tolist(),
            'labels': test_labels.tolist()
        },
        'optimal_threshold': {
            'value': float(optimal_threshold),
            'precision': float(optimal_precision),
            'recall': float(optimal_recall)
        }
    }
    
    # Save as JSON
    with open(os.path.join(output_path, 'precision_recall_data.json'), 'w') as f:
        json.dump(raw_data, f, indent=4)
        
    # Also save as CSV for easier plotting in other tools
    # Prepare DataFrames for validation and test data
    val_df = pd.DataFrame({
        'precision': val_precision,
        'recall': val_recall,
        'threshold': np.append(val_thresholds, [1.0]) if len(val_thresholds) > 0 else [1.0],
        'dataset': 'validation'
    })
    
    test_df = pd.DataFrame({
        'precision': test_precision,
        'recall': test_recall,
        'threshold': np.append(test_thresholds, [1.0]) if len(test_thresholds) > 0 else [1.0],
        'dataset': 'test'
    })
    
    # Combine and save
    pr_df = pd.concat([val_df, test_df])
    pr_df.to_csv(os.path.join(output_path, 'precision_recall_curves.csv'), index=False)
    
    # Save raw scores and labels as numpy arrays directly to avoid shape issues
    np.savez(os.path.join(output_path, 'scores_and_labels.npz'),
             val_scores=val_scores,
             val_labels=val_labels,
             test_scores=test_scores,
             test_labels=test_labels)
    
    print(f"Raw data saved to {os.path.join(output_path, 'precision_recall_data.json')} and related files")
    
def load_latest_checkpoint(out_dir, device):
    """
    Load the latest checkpoint from the output directory.
    
    Args:
        out_dir: Directory containing checkpoints
        device: Device to load checkpoint to
        
    Returns:
        checkpoint: Loaded checkpoint
        checkpoint_model_args: Model arguments from checkpoint
        config: Configuration dictionary
    """
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
    
    return checkpoint, checkpoint_model_args, config

def main():
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # Whether this is the main process
    master_process = global_rank == 0
    
    print(f"Process {global_rank+1}/{world_size} starting on {device}")
    
    # Parse command line arguments
    args = parse_args()
    
    # Output directories setup
    if args.approach=="direct":
        output_path = os.path.join(args.out_dir, f"{args.condition_name}_evaluation_results_{args.time_horizon}_{args.approach}")
    else:
        output_path = os.path.join(args.out_dir, f"{args.condition_name}_evaluation_results_{args.time_horizon}_{args.approach}_intermdiate_horizon_{args.intermediate_horizon}_{args.intermediate_agg}")


    if master_process:
        os.makedirs(output_path, exist_ok=True)
        print(f"Output path: {output_path}")
        # Create sample_logits directory within the output path
        sample_logits_dir = os.path.join(output_path, 'sample_logits')
        os.makedirs(sample_logits_dir, exist_ok=True)
    
    # Load condition indices
    with open(args.condition_indices_path, "r") as f:
        condition_indices = json.load(f)
    
    # Load model checkpoint
    checkpoint, checkpoint_model_args, config = load_latest_checkpoint(args.out_dir, device)
    print(f"Process {global_rank}: Latest checkpoint loaded successfully")
    
    # Create model
    model_args = {}
    for key in GPTCONFIG_PARAMS:
        if key in checkpoint_model_args:
            model_args[key] = checkpoint_model_args[key]
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Clean up state dict if needed (remove DDP-specific prefixes)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank])
    model.eval()
    
    # Make sure all processes are ready
    torch.cuda.synchronize(device)
    dist.barrier()
    print(f"Process {global_rank}: Model initialized and ready")
    
    # Prepare validation dataset and dataloader
    print(f"Process {global_rank}: Loading validation dataset...")
    val_dataset = SeqCLSDataset(mode="val", folder=args.data_path, multiwin=args.multiwin, time_horizon=args.time_horizon, num_patients = args.num_patients,approach = args.approach, horizon = args.intermediate_horizon)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8)
    
    # Evaluate on validation set to get scores for threshold calculation
    print(f"Process {global_rank}: Evaluating on validation set...")
    val_labels, val_scores, _ = evaluate_model_distributed(
        model, val_dataloader, device, condition_indices, 
        condition_name=args.condition_name, agg_method=args.agg_method, compute_metrics=False,
        rank=global_rank, world_size=world_size, n_gather=args.n_gather, approach=args.approach,
        intermediate_agg=args.intermediate_agg, save_sample_logits=True, output_path=output_path
    )
    
    # Find optimal threshold on validation set (only on rank 0)
    optimal_threshold = None
    if master_process:
        print("Finding optimal threshold...")
        optimal_threshold = find_optimal_threshold(val_labels, val_scores, metric='f1')
        print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Broadcast optimal threshold to all processes
    optimal_threshold = torch.tensor([optimal_threshold if optimal_threshold is not None else 0.0], device=device)
    dist.broadcast(optimal_threshold, src=0)
    optimal_threshold = optimal_threshold.item()
    
    # # Re-evaluate on validation set with optimal threshold
    _, _, val_metrics = evaluate_model_distributed(
        model, val_dataloader, device, condition_indices, 
        condition_name=args.condition_name, agg_method=args.agg_method, threshold=optimal_threshold,
        rank=global_rank, world_size=world_size, n_gather=args.n_gather, approach=args.approach,
        intermediate_agg=args.intermediate_agg, save_sample_logits=False
    )
    
    # Prepare test dataset and dataloader
    print(f"Process {global_rank}: Loading test dataset...")
    test_dataset = SeqCLSDataset(mode="test", folder=args.data_path, multiwin=args.multiwin, time_horizon=args.time_horizon, num_patients = args.num_patients ,approach = args.approach, horizon = args.intermediate_horizon)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=8)
    
    # Evaluate on test set with optimal threshold
    print(f"Process {global_rank}: Evaluating on test set...")
    test_labels, test_scores, test_metrics = evaluate_model_distributed(
        model, test_dataloader, device, condition_indices, 
        condition_name=args.condition_name, agg_method=args.agg_method, threshold=optimal_threshold,
        rank=global_rank, world_size=world_size, n_gather=args.n_gather, approach=args.approach,
        intermediate_agg=args.intermediate_agg, save_sample_logits=False
    )
    
    # Plot and save precision-recall curves (only on rank 0)
    if master_process:
        print("Plotting precision-recall curves...")
        plot_precision_recall_curve(
            val_labels, val_scores, test_labels, test_scores,
            optimal_threshold, output_path
        )
        
        # Save metrics
        metrics = {
            'optimal_threshold': optimal_threshold,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        # Print metrics
        print("\nValidation Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Save metrics to JSON
        with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"\nEvaluation completed. Results saved to {output_path}")
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()