# Multiclass Baseline: Next-Token Prediction

This directory implements a standard Next-Token Prediction (NTP) baseline for clinical EHR sequences using a GPT-style architecture.

## Overview

The multiclass baseline models longitudinal patient records as a sequence of clinical event codes and visit delimiters (SEP tokens). It aims to predict the next clinical code or the next visit marker in the sequence.

## Getting Started

### Training
To start training directly with Python:
```bash
python train.py --batch_size=8 --learning_rate=1e-4 --out_dir="out_ntp_full"
```

#### Detailed Training Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--out_dir` | str | `out_ntp_full_lr_1e-4` | Directory to save checkpoints and logs. |
| `--batch_size` | int | 8 | Batch size per GPU/process. |
| `--learning_rate` | float | 1e-4 | Peak learning rate for AdamW. |
| `--block_size` | int | 512 | Maximum sequence length (context window). |
| `--gradient_accumulation_steps` | int | 8 | Number of steps to accumulate gradients before updating weights. |
| `--n_layer` | int | 8 | Number of transformer layers. |
| `--n_head` | int | 8 | Number of attention heads. |
| `--n_embd` | int | 1024 | Embedding dimension. |
| `--max_iters` | int | 500,000 | Total number of training iterations. |
| `--save_interval` | int | 1,000 | Interval (in iterations) to save model checkpoints. |
| `--init_from` | str | `scratch` | Either `scratch` or `resume`. |

### Evaluation
To evaluate on clinical tasks (e.g., Acute MI):
```bash
python eval.py \
    --out_dir "out_ntp_full" \
    --data_path "/path/to/val.csv" \
    --condition_indices_path "/path/to/condition_indices.json" \
    --condition_name "acute_mi" \
    --mode "test"
```

#### Detailed Evaluation Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--out_dir` | str | (required) | Directory containing the trained model checkpoints. |
| `--data_path` | str | (required) | Path to the evaluation CSV file. |
| `--condition_indices_path` | str | (required) | Mapping of conditions to clinical code indices. |
| `--condition_name` | str | (required) | Clinical condition to evaluate. |
| `--mode` | str | `val` | Evaluation mode: `val` or `test`. |
| `--total_months` | int | 24 | Total months for sequential rollout horizon. |
| `--month_step` | int | 3 | Step size (in months) for reporting metrics during rollout. |
| `--R` | int | 100 | Monte Carlo rollouts per patient window. |
| `--batch_size` | int | 16 | Parallel patient windows. |
| `--batch_r` | int | 32 | Parallel rollouts per window per chunk. |
| `--topk` | int | 50 | Top-K sampling cutoff. |
| `--temperature` | float | 1.0 | Sampling temperature. |

## Configuration
Hyperparameters can be overridden through command-line arguments (e.g., `--batch_size=16`) or by creating a custom configuration file as specified in `configurator.py`.
