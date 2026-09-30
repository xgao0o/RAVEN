"""Zero-shot condition evaluation with one inference pass per split."""
import argparse
from contextlib import nullcontext
from dataclasses import fields
from datetime import timedelta
from glob import glob
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (precision_recall_curve, auc, average_precision_score,
                             precision_score, recall_score, f1_score, roc_auc_score)

from raven.model import GPTConfig, GPT
from data_processing.condition_helper import SeqCLSDataset, aggregate_logits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', required=True, help='Checkpoint and result directory')
    parser.add_argument('--results_dir', help='Explicit output directory; defaults to a named directory under out_dir')
    parser.add_argument('--data_path', required=True, help='Downstream cohort shelves')
    parser.add_argument('--condition_indices_path', required=True)
    parser.add_argument('--metadata_folder', help='Feature metadata; defaults to checkpoint data_dir')
    parser.add_argument('--iteration', type=int, help='Checkpoint iteration; defaults to latest')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=8, help='Loader workers per GPU')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512, help='History context length, including final SEP')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='float32',
                        help='Inference precision; mixed precision is opt-in')
    parser.add_argument('--agg_method', default='sum',
                        choices=['sum', 'noisy_or', 'max', 'logsumexp', 'geometric_mean', 'attention'])
    parser.add_argument('--score_space', choices=['probability', 'logit'], default='probability',
                        help='logit ranks raw summed logits without sigmoid underflow; requires direct + sum')
    parser.add_argument('--condition_name', default='Dementia')
    parser.add_argument('--multiwin', action='store_true')
    parser.add_argument('--time_horizon', type=int, default=730)
    parser.add_argument('--n_gather', type=int, default=100,
                        help='Batches per CPU score flush; results are gathered once per split')
    parser.add_argument('--num_patients', type=int, default=512,
                        help='Number of cohort records/windows; -1 evaluates all records')
    parser.add_argument('--seed', type=int, default=1037, help='Shared cohort sampling seed')
    parser.add_argument('--approach', default='direct', choices=['direct', 'intermediate_pred'])
    parser.add_argument('--intermediate_horizon', type=int, default=365)
    parser.add_argument('--intermediate_agg', default='mean', choices=['mean', 'max'])
    parser.add_argument('--skip_sample_logits', action='store_true',
                        help='Skip per-batch validation horizon files to reduce filesystem I/O')
    args = parser.parse_args()
    if min(args.batch_size, args.prefetch_factor, args.n_gather) <= 0 or args.num_workers < 0:
        parser.error('batch_size, prefetch_factor, n_gather must be positive; num_workers must be nonnegative')
    if args.score_space == 'logit' and (args.approach != 'direct' or args.agg_method != 'sum'):
        parser.error('logit score_space requires direct prediction and sum aggregation')
    return args


def threshold_above_scores(scores):
    """Represent the PR endpoint that predicts no positives in any score space."""
    values = np.asarray(scores)
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype('float64')
    return float(np.nextafter(values.max().astype(dtype), np.array(np.inf, dtype=dtype)))


def find_optimal_threshold(y_true, y_scores, metric='f1'):
    precisions, recalls, thresholds = precision_recall_curve(np.ravel(y_true), np.ravel(y_scores))
    values = {'f1': 2 * precisions * recalls / (precisions + recalls + 1e-8),
              'precision': precisions, 'recall': recalls}
    if metric not in values:
        raise ValueError(f'Unknown metric: {metric}')
    optimal_idx = np.argmax(values[metric])
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else threshold_above_scores(y_scores)


def classification_metrics(labels, scores, threshold):
    labels, scores = np.ravel(labels), np.ravel(scores)
    predictions = scores >= threshold
    return {
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'auroc': roc_auc_score(labels, scores) if np.unique(labels).size > 1 else float('nan'),
        'auprc': average_precision_score(labels, scores),
    }


@torch.inference_mode(False)
def gather_variable_results(tensor, rank, world_size):
    """Gather once, including empty shards, allocating receive buffers only on rank 0."""
    if world_size == 1:
        return tensor.cpu()
    # Collectives mutate receive buffers from backend threads outside inference
    # mode, so those buffers must be ordinary tensors (including on Gloo).
    local_size = torch.tensor([len(tensor)], dtype=torch.long, device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(size.item()) for size in sizes]
    max_size = max(sizes)
    if max_size == 0:
        return tensor.cpu() if rank == 0 else None
    padded = tensor.new_zeros((max_size,) + tensor.shape[1:])
    padded[:len(tensor)] = tensor
    received = [torch.empty_like(padded) for _ in range(world_size)] if rank == 0 else None
    dist.gather(padded, gather_list=received, dst=0)
    if rank == 0:
        return torch.cat([value[:size] for value, size in zip(received, sizes)]).cpu()
    return None


@torch.inference_mode()
def evaluate_model_distributed(model, dataloader, device, condition_indices, condition_name='Dementia',
                               agg_method='sum', threshold=0.5, compute_metrics=True, rank=0,
                               world_size=1, n_gather=100, approach='direct', intermediate_agg='mean',
                               save_sample_logits=True, output_path=None, dtype='float32',
                               score_space='probability', return_identifiers=False):
    """Score each local record once; uneven ranks communicate only after inference."""
    if n_gather <= 0:
        raise ValueError('n_gather must be positive')
    if approach not in ('direct', 'intermediate_pred') or intermediate_agg not in ('mean', 'max'):
        raise ValueError('Unknown prediction approach or intermediate aggregation')
    if score_space not in ('probability', 'logit'):
        raise ValueError('Unknown score space')
    if score_space == 'logit' and (approach != 'direct' or agg_method != 'sum'):
        raise ValueError('logit score_space requires direct prediction and sum aggregation')
    device = torch.device(device)
    model.eval()
    # Sorting and deduplication preserve the original boolean vocabulary mask's order.
    indices = sorted(set(condition_indices[condition_name]))
    if not indices or min(indices) < 0 or max(indices) >= model.config.vocab_size:
        raise ValueError(f'Invalid condition indices for vocabulary size {model.config.vocab_size}')
    selected = torch.tensor(indices, dtype=torch.long, device=device)
    selected_mask = torch.ones(len(indices), dtype=torch.bool, device=device)
    ctx = (torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))
           if dtype != 'float32' else nullcontext())
    sample_dir = None
    if approach == 'intermediate_pred' and save_sample_logits and output_path:
        sample_dir = Path(output_path) / 'sample_logits'
        sample_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    chunks = []
    local_identifiers = {'patient_ids': [], 'record_ids': []}
    for batch_idx, batch in enumerate(dataloader):
        batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        patient_ids = batch.get('patient_ids')
        days = batch['days_embed_arr']
        horizons = [days] if approach == 'direct' else days.unbind(dim=1)
        horizon_scores = []
        for horizon_days in horizons:
            with ctx:
                logits, _, _, _ = model(
                    batch['token_arr'], horizon_days, None, batch['pred_mask'], batch['pad_mask'],
                    batch['attention_bias'], patient_ids=patient_ids,
                    prediction_only=True, output_indices=selected,
                )
            # Sigmoid is strictly increasing in exact arithmetic, but very negative
            # FP32 sums underflow to zero and erase rankings. Keep the sum itself.
            horizon_scores.append(logits.float().sum(dim=1) if score_space == 'logit' else
                                  aggregate_logits(logits, selected_mask, method=agg_method, dim=1))
        if approach == 'direct':
            scores = horizon_scores[0]
        else:
            full_logits = torch.stack(horizon_scores)
            if sample_dir is not None:
                save_data = {'logits': full_logits.cpu(), 'labels': batch['labels'].cpu(),
                             'batch_idx': batch_idx, 'rank': rank}
                if patient_ids is not None:
                    save_data['patient_ids'] = patient_ids.cpu()
                torch.save(save_data, sample_dir / f'logits_{rank}_{batch_idx}.pt')
            scores = full_logits.mean(0) if intermediate_agg == 'mean' else full_logits.max(0).values
        labels = batch['labels'].reshape(-1)
        if scores.numel() != labels.numel():
            raise ValueError('Expected exactly one prediction per cohort record')
        if return_identifiers:
            for field, output in (('patient_id', 'patient_ids'), ('record_id', 'record_ids')):
                values = batch.get(field)
                if values is None or len(values) != labels.numel():
                    raise ValueError(f'Missing or misaligned {field} for patient-level confidence intervals')
                local_identifiers[output].extend(str(value) for value in values)
        pending.append(torch.stack((scores, labels), dim=1))
        if len(pending) == n_gather:
            chunks.append(torch.cat(pending).cpu())
            pending.clear()
    if pending:
        chunks.append(torch.cat(pending).cpu())
    local_results = torch.cat(chunks) if chunks else torch.empty((0, 2), dtype=torch.float32)
    results = gather_variable_results(local_results.to(device), rank, world_size)
    identifiers = None
    if return_identifiers:
        gathered = [local_identifiers]
        if world_size > 1:
            gathered = [None] * world_size if rank == 0 else None
            with torch.inference_mode(False):
                dist.gather_object(local_identifiers, gathered, dst=0)
        if rank == 0:
            identifiers = {key: np.asarray([value for shard in gathered for value in shard[key]], dtype=str)
                           for key in local_identifiers}
            if any(len(values) != len(results) for values in identifiers.values()):
                raise RuntimeError('Gathered identifiers and predictions have different lengths')
    if rank != 0:
        return (None, None, None, None) if return_identifiers else (None, None, None)
    if not len(results):
        raise ValueError('Evaluation cohort is empty')
    scores = results[:, 0].numpy()
    labels = results[:, 1].numpy().astype(np.int64)
    metrics = classification_metrics(labels, scores, threshold) if compute_metrics else None
    return (labels, scores, metrics, identifiers) if return_identifiers else (labels, scores, metrics)


def plot_precision_recall_curve(val_labels, val_scores, test_labels, test_scores, 
                                optimal_threshold, output_path, val_identifiers=None, test_identifiers=None):
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    # Ensure arrays are 1-dimensional
    val_labels = np.ravel(val_labels)
    val_scores = np.ravel(val_scores)
    test_labels = np.ravel(test_labels)
    test_scores = np.ravel(test_scores)
    saved_identifiers = {}
    if (val_identifiers is None) != (test_identifiers is None):
        raise ValueError('Provide identifiers for both validation and test splits')
    for split, identifiers, labels in [('val', val_identifiers, val_labels),
                                       ('test', test_identifiers, test_labels)]:
        if identifiers is not None:
            for field in ('patient_ids', 'record_ids'):
                values = np.asarray(identifiers[field], dtype=str)
                if values.ndim != 1 or len(values) != len(labels):
                    raise ValueError(f'Misaligned {split}_{field}')
                saved_identifiers[f'{split}_{field}'] = values
    
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
        'threshold': np.append(val_thresholds, threshold_above_scores(val_scores)),
        'dataset': 'validation'
    })
    
    test_df = pd.DataFrame({
        'precision': test_precision,
        'recall': test_recall,
        'threshold': np.append(test_thresholds, threshold_above_scores(test_scores)),
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
             test_labels=test_labels,
             **saved_identifiers)
    
    print(f"Raw data saved to {os.path.join(output_path, 'precision_recall_data.json')} and related files")
    
def load_latest_checkpoint(out_dir, device='cpu', iteration=None):
    paths = glob(os.path.join(out_dir, 'ckpt-*.pt'))
    if iteration is not None:
        latest = Path(out_dir) / f'ckpt-{iteration}.pt'
    else:
        if not paths:
            raise FileNotFoundError(f'No checkpoint files found in {out_dir}')
        latest = Path(max(paths, key=lambda path: int(Path(path).stem.split('-')[-1])))
    print(f'Loading checkpoint: {latest}', flush=True)
    # Optimizer state stays off the GPU; mmap avoids eagerly reading its storage.
    try:
        checkpoint = torch.load(str(latest), map_location=device, mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error):
            raise
        checkpoint = torch.load(str(latest), map_location=device)
    config = checkpoint.get('config')
    if config is None:
        with open(Path(out_dir) / 'config.json') as handle:
            config = json.load(handle)
    return checkpoint, checkpoint['model_args'], config


def make_dataloader(dataset, args, rank, world_size, device):
    # DistributedSampler pads with repeated records. Strided shards include every
    # record exactly once, even when there are fewer records than ranks.
    shard = Subset(dataset, range(rank, len(dataset), world_size))
    options = {'prefetch_factor': args.prefetch_factor} if args.num_workers else {}
    return DataLoader(shard, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=device.type == 'cuda', **options)


def main():
    args = parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo',
                                timeout=timedelta(minutes=20))
    master = rank == 0
    torch.manual_seed(args.seed)
    output_path = os.path.join(args.out_dir,
                              f'{args.condition_name}_evaluation_results_{args.time_horizon}_{args.approach}')
    if args.approach != 'direct':
        output_path += f'_intermdiate_horizon_{args.intermediate_horizon}_{args.intermediate_agg}'
    if args.score_space != 'probability':
        output_path += f'_{args.score_space}'
    if args.results_dir:
        output_path = args.results_dir
    if master:
        os.makedirs(output_path, exist_ok=True)
    with open(args.condition_indices_path) as handle:
        condition_indices = json.load(handle)
    checkpoint, saved_args, config = load_latest_checkpoint(args.out_dir, iteration=args.iteration)
    model_args = {field.name: saved_args[field.name] for field in fields(GPTConfig) if field.name in saved_args}
    if 'data_dir' not in model_args and saved_args.get('data_folder'):
        model_args['data_dir'] = saved_args['data_folder']
    if args.metadata_folder:
        model_args['data_dir'] = args.metadata_folder
    gptconf = GPTConfig(**model_args)
    if args.max_length > gptconf.block_size:
        raise ValueError('max_length exceeds the checkpoint block size')
    model = GPT(gptconf)
    state_dict = {key.removeprefix('_orig_mod.'): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    checkpoint_iteration = checkpoint.get('iter_num')
    del state_dict, checkpoint
    # No DDP wrapper: inference has no gradients or per-forward communication.
    dataset_args = dict(folder=args.data_path, max_length=args.max_length, multiwin=args.multiwin,
                        time_horizon=args.time_horizon, num_patients=args.num_patients,
                        approach=args.approach, horizon=args.intermediate_horizon,
                        metadata_folder=gptconf.data_dir, seed=args.seed, vocab_size=gptconf.vocab_size)
    eval_args = dict(condition_name=args.condition_name, agg_method=args.agg_method,
                     rank=rank, world_size=world_size, n_gather=args.n_gather, approach=args.approach,
                     intermediate_agg=args.intermediate_agg, dtype=args.dtype, score_space=args.score_space)
    start = time.perf_counter()
    print(f'Process {rank}: Evaluating validation...', flush=True)
    val_dataset = SeqCLSDataset(mode='val', **dataset_args)
    val_count = len(val_dataset)
    val_loader = make_dataloader(val_dataset, args, rank, world_size, device)
    val_labels, val_scores, _, val_identifiers = evaluate_model_distributed(
        model, val_loader, device, condition_indices, compute_metrics=False,
        save_sample_logits=not args.skip_sample_logits, output_path=output_path,
        return_identifiers=True, **eval_args)
    val_dataset._cleanup()
    del val_loader, val_dataset
    # Threshold fitting and validation metrics use the scores already computed.
    # Workers can immediately proceed to test inference without a broadcast.
    optimal_threshold = None
    val_metrics = None
    if master:
        if len(val_scores) != val_count:
            raise RuntimeError('Validation result count differs from the cohort size')
        optimal_threshold = float(find_optimal_threshold(val_labels, val_scores))
        val_metrics = classification_metrics(val_labels, val_scores, optimal_threshold)
        print(f'Optimal threshold: {optimal_threshold:.6g}', flush=True)
    print(f'Process {rank}: Evaluating test...', flush=True)
    test_dataset = SeqCLSDataset(mode='test', **dataset_args)
    test_count = len(test_dataset)
    test_loader = make_dataloader(test_dataset, args, rank, world_size, device)
    test_labels, test_scores, _, test_identifiers = evaluate_model_distributed(
        model, test_loader, device, condition_indices, compute_metrics=False,
        save_sample_logits=False, return_identifiers=True, **eval_args)
    test_dataset._cleanup()
    if master:
        if len(test_scores) != test_count:
            raise RuntimeError('Test result count differs from the cohort size')
        test_metrics = classification_metrics(test_labels, test_scores, optimal_threshold)
        plot_precision_recall_curve(val_labels, val_scores, test_labels, test_scores,
                                    optimal_threshold, output_path, val_identifiers, test_identifiers)
        metrics = {'optimal_threshold': optimal_threshold, 'score_space': args.score_space,
                   'validation': val_metrics, 'test': test_metrics}
        with open(os.path.join(output_path, 'metrics.json'), 'w') as handle:
            json.dump(metrics, handle, indent=4)
        report = dict(vars(args), checkpoint_iteration=checkpoint_iteration,
                      metadata_folder=gptconf.data_dir, vocab_size=gptconf.vocab_size,
                      world_size=world_size, validation_records=val_count, test_records=test_count,
                      prediction_order='rank_concatenated', saved_patient_identifiers=True,
                      elapsed_seconds=time.perf_counter() - start)
        with open(os.path.join(output_path, 'evaluation_config.json'), 'w') as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(metrics, indent=2), flush=True)
        print(f'Evaluation completed. Results saved to {output_path}', flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
