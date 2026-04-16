# EGE: Enhanced Generative EHR

EGE is a GPT-based model designed specifically for longitudinal Electronic Health Record (EHR) data. Its primary innovation is an explicit duration modeling system using a "coin factorization" approach.

## Key Features

- **Duration Tokens**: Instead of simple SEP tokens between visits, EGE inserts duration tokens that represent the time elapsed between events. These durations are factorized into coins (e.g., 180 days, 60 days, 30 days, etc.) to keep the vocabulary small while maintaining temporal precision.
- **RoPE (Rotary Positional Embeddings)**: Uses rotary embeddings over absolute positions for efficient long-range dependencies.
- **Next-Token Prediction**: Trained with a standard Cross-Entropy loss for generative tasks.

## Getting Started

### Training
To start training directly with Python:
```bash
python train.py --batch_size=8 --learning_rate=1e-4 --out_dir="out_ntp_duration"
```

#### Detailed Training Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--out_dir` | str | `out_ntp_duration_lr_1e-4` | Directory to save checkpoints and logs. |
| `--batch_size` | int | 8 | Batch size per GPU/process. |
| `--learning_rate` | float | 1e-4 | Peak learning rate for AdamW. |
| `--block_size` | int | 512 | Maximum sequence length (context window). |
| `--gradient_accumulation_steps` | int | 8 | Number of steps to accumulate gradients before updating weights. |
| `--n_layer` | int | 8 | Number of transformer layers. |
| `--n_head` | int | 8 | Number of attention heads. |
| `--n_embd` | int | 1024 | Embedding dimension. |
| `--max_iters` | int | 500,000 | Total number of training iterations. |
| `--warmup_iters` | int | 10,000 | Number of iterations for learning rate warmup. |
| `--save_interval` | int | 25,000 | Interval (in iterations) to save model checkpoints. |
| `--init_from` | str | `scratch` | Either `scratch` to start fresh or `resume` to continue from `out_dir`. |

### Evaluation
To evaluate on downstream clinical tasks:
```bash
python eval.py \
    --out_dir "out_ntp_duration" \
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
| `--condition_indices_path` | str | (required) | Path to JSON mapping condition names to their clinical code indices. |
| `--condition_name` | str | (required) | The specific clinical condition to evaluate (e.g., `acute_mi`). |
| `--mode` | str | `val` | Evaluation mode: `val` or `test`. |
| `--horizon_months` | int | 24 | The time horizon (in months) for prediction. |
| `--R` | int | 100 | Number of Monte Carlo rollouts per patient window. |
| `--batch_size` | int | 16 | Number of patient windows to process in parallel. |
| `--batch_r` | int | 32 | Parallel rollouts per window per chunk. |
| `--topk` | int | 50 | Top-K sampling parameter. |
| `--temperature` | float | 1.0 | Sampling temperature. |

## Configuration

Configuration is managed via `configurator.py`, which allows you to override any global variable in `train.py` via CLI (e.g., `--n_layer=12`) or by providing a config script.
