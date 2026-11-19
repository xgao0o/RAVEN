# Scaling Recurrence-aware Foundation Models for Clinical Records via Next-Visit Prediction

RAVEN is a generative pretraining approach for sequential electronic health record (EHR) data using next-visit event prediction. Trained on data from over one million patients, RAVEN learns to autoregressively generate tokenized clinical events for future visits based on patient history. The model achieves competitive performance in zero-shot disease incidence forecasting and generalizes to external cohorts without additional fine-tuning.

**Paper:** [arxiv.org/abs/2603.24562](https://arxiv.org/abs/2603.24562)

**Model:** [huggingface.co/NYUMedML/RAVEN](https://huggingface.co/NYUMedML/RAVEN)

## Repository Structure

```
RAVEN/
  raven/                    # Core model code
    model.py                #   GPT architecture with EHR-specific loss functions
    dataset.py              #   UnifiedSeqEHRDataset for sequence preprocessing
    focal.py                #   Focal loss implementation
    rotary_embedding_torch.py  # Rotary positional embeddings (RoPE/xpos)
    configurator.py         #   CLI config override utility
    bootstrap.py            #   Shared patient-level confidence interval helpers
  scripts/                  # Training and evaluation entry points
    train.py                #   Distributed training (single/multi-GPU via torchrun)
    calc_loss.py            #   Compute test-set loss
    eval_condition_mp.py    #   Zero-shot condition evaluation (single condition)
    eval_conditions_agg.py  #   Batch condition evaluation (all conditions)
    eval_script.py          #   BERT-based downstream evaluation
  data_processing/          # Data loading and evaluation utilities
    condition_helper.py     #   Condition dataset and logit aggregation
    condition_helper_ehrshot.py  # EHRSHOT-specific condition handling
    cls_dataloader.py       #   Classification dataloader
    cls_dataloader_ehrshot.py   # EHRSHOT classification dataloader
  config/                   # Hyperparameter configuration files
    train_raven.py
  baselines/                # Baseline model implementations
    EGE/                    #   Enhanced Generative EHR (duration tokens)
    Multiclass/             #   Standard next-token prediction
    SeqLoss/                #   Multiset loss variant
    LLM_inference/          #   Zero-shot LLM evaluation
  notebooks/                # Analysis notebooks
    EHRSHOT.ipynb           #   EHRSHOT data translation and mapping
    bootstrap_ci.ipynb      #   Patient-level bootstrap confidence intervals
    on_time.ipynb           #   On-time prediction analysis
  environment_gpt.yml       # Conda environment specification
  setup.py                  # Package installation
```

## Setup

### Environment

```bash
conda env create -f environment_gpt.yml
conda activate gpt
pip install -e .
```

### Data Preparation

RAVEN expects data organized in the following structure:

```
data/
  csvs/
    train.csv                      # Patient split: columns [patient_id, part]
    val.csv
    test.csv
  data_files/
    delete_inds.npy                # Indices of tokens to remove
    loss_mask_0_perc.npy            # Eligible prediction targets
    loss_mask_{40,50,60,75,90}_perc.npy  # Frequency-band loss masks
    pos_weights_icf.npy            # Inverse condition frequency weights
    pos_weights_idf.npy            # Inverse document frequency weights
    pos_weights_ccf.npy            # Cumulative condition frequency weights
    pos_weights_cdf.npy            # Cumulative document frequency weights
    filtered_headers_token_merge.json  # Token header mappings
  conditions2inds.json   # Condition-to-token-index mapping
  processed_part_0.shelve          # Shelve databases with sparse EHR matrices
  processed_part_1.shelve
  ...
  processed_part_39.shelve
```

Each shelve database stores sparse matrices indexed by patient ID, where rows represent visits and columns represent clinical tokens.

## Training

### Single GPU

```bash
python scripts/train.py config/train_raven.py --csv_folder=./data
```

### Multi-GPU (Distributed Data Parallel)

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
    config/train_raven.py --csv_folder=./data
```

### With Config File

```bash
python scripts/train.py config/train_raven.py --csv_folder=./data
```

Key hyperparameters can be overridden via `--key=value` CLI arguments. Checkpoints are saved periodically to the output directory.

Training uses compact labels and temporal weights at `[SEP]` prediction positions
by default (`loss_only=True`). The transformer still processes the full context;
only the output projection and loss skip unsupervised positions. Both single-window
and packed multi-window training support this path. The model's default forward
API still returns dense logits for evaluation, and existing model checkpoints
remain compatible. Set `--loss_only=False` to use dense labels and outputs.

The loader uses eight persistent workers per process and prefetches four batches
per worker. Adjust `--train_workers` and `--loader_prefetch` to fit the available
CPUs and memory; `--train_workers=0` runs loading in the main process. CUDA batches
use pinned memory and nonblocking transfers. Compilation is disabled by default.

Use `--out_dir=./runs/raven` to choose a checkpoint directory, then use the same
directory with `--init_from=resume` to continue training. `max_iters`, validation
intervals, and checkpoint intervals count microbatches; an optimizer update occurs
after `gradient_accumulation_steps / world_size` microbatches per process. Under
DDP, validation retains the existing rank-zero shard and `eval_iters` batch limit.

## Evaluation

### Test Loss

```bash
torchrun --standalone --nproc_per_node=2 scripts/calc_loss.py \
    --out_dir=<checkpoint_dir> \
    --folder=<data_dir> \
    --csv_folder=<data_dir>/csvs \
    --split=test \
    --iteration=600000
```

Loss and aggregate-condition evaluation accept `--metadata_folder=/path/to/metadata`
when the checkpoint's loss masks live separately from `--folder`.

### Batch Condition Evaluation (All Conditions)

```bash
torchrun --standalone --nproc_per_node=2 scripts/eval_conditions_agg.py \
    --out_dir=<checkpoint_dir> \
    --folder=<data_dir> \
    --csv_folder=<data_dir>/csvs \
    --all_conditions \
    --threshold_method=f1 \
    --agg_method=sum
```


### Zero-Shot Condition Evaluation (Single Condition)

```bash
torchrun --standalone --nproc_per_node=2 scripts/eval_condition_mp.py \
    --out_dir=<checkpoint_dir> \
    --data_path=<condition_data_dir> \
    --condition_indices_path=<data_dir>/conditions2inds_ICLR_plus_one.json \
    --condition_name=COPD \
    --time_horizon=730 \
    --agg_method=sum \
    --approach=direct \
    --num_patients=-1 \
    --multiwin
```

## Baselines

See [`baselines/README.md`](baselines/README.md) for instructions on running baseline models (EGE, Multiclass NTP, SeqLoss, LLM inference).

## Citation

```bibtex
@article{gao2026raven,
  title={Scaling Recurrence-aware Foundation Models for Clinical Records via Next-Visit Prediction},
  author={Gao, Xiang and Rajamohan, Haresh Rengaraj and Zhu, Weicheng and Huang, Shih-Lun and Chen, Long and Schulman, Gabe and Jin, Huizhen and Li, Shengduo and Wang, Yixuan and Yang, Huidi and others},
  journal={arXiv preprint arXiv:2603.24562},
  year={2026}
}
```

## Acknowledgements

The training code is based on [nanoGPT](https://github.com/karpathy/nanoGPT).
