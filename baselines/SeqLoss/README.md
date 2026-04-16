# SeqLoss: Multiset Loss for EHR Modeling

SeqLoss is a variant of the GPT-based EHR model that supports a "Multiset Loss," providing more flexible supervision for unordered clinical codes within a visit.

## Motivation

In EHR data, the order of clinical codes within a single visit is often arbitrary. Standard Next-Token Prediction (NTP) imposes a strict sequence, which can be semantically misleading. SeqLoss addresses this by introducing a multiset/parallel loss approach.

## Getting Started

### Training
Start training directly with Python, using the `multiset` loss mode:
```bash
python train.py --loss_mode="multiset" --batch_size=8 --out_dir="out_multiset"
```

#### Detailed Training Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--out_dir` | str | `out_multiset_lr_1e-4` | Directory for logs and checkpoints. |
| `--loss_mode` | str | `multiset` | Training objective: `multiset` (unordered) or `ce` (standard causal NTP). |
| `--batch_size` | int | 8 | Batch size per GPU. |
| `--learning_rate` | float | 1e-4 | Learning rate for AdamW. |
| `--block_size` | int | 512 | Max sequence context length. |
| `--gradient_accumulation_steps` | int | 8 | Effective batch size = this * batch_size. |
| `--n_layer` | int | 8 | Number of layers. |
| `--n_head` | int | 8 | Number of attention heads. |
| `--n_embd` | int | 1024 | Embedding dimension size. |
| `--save_interval` | int | 1,000 | Save checkpoints every N iterations. |

### Evaluation
To evaluate on clinical tasks (e.g., CHF):
```bash
python eval.py \
    --out_dir "out_multiset" \
    --data_path "/path/to/val.csv" \
    --condition_indices_path "/path/to/condition_indices.json" \
    --condition_name "chf" \
    --mode "test"
```

#### Detailed Evaluation Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--out_dir` | str | (required) | Directory containing the trained checkpoints. |
| `--data_path` | str | (required) | Path to evaluation CSV data. |
| `--condition_indices_path` | str | (required) | JSON file mapping conditions to clinical codes. |
| `--condition_name` | str | (required) | Outcome condition to evaluate. |
| `--total_months` | int | 24 | Sequence rollout time horizon (months). |
| `--month_step` | int | 3 | Reporting step size for metrics (months). |
| `--R` | int | 100 | Number of rollouts per window. |
| `--batch_size` | int | 16 | Number of parallel patient windows. |
| `--topk` | int | 50 | Top-K sampling cutoff. |

## Configuration
Hyperparameters are managed via `configurator.py`. Use the `--loss_mode` argument to toggle between `multiset` and standard `ce` training objectives.
