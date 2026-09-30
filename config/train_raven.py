# Settings of the v5 0.5-decay run with targets observed in >100 patients.
# Iterations count microbatches. The reference used two GPUs (global batch 64).
# $ torchrun --standalone --nproc_per_node=2 scripts/train.py config/train_raven.py

wandb_log = True
wandb_project = 'GPT-EHR'
wandb_run_name = 'raven'

# model
n_layer = 8
n_head = 8
n_embd = 1024
block_size = 512
dropout = 0.0
bias = False
rotary = True
use_xpos = False

# loss
loss_type = "FL"  # focal loss
loss_weighting = "standard"
gamma = 2.0
alpha = 0.995
pos_weight = 1.0

# training
loss_only = True  # store labels and project outputs only at SEP positions
train_workers = 8  # per process; set to 0 for loading in the main process
loader_prefetch = 4  # batches prefetched per worker
compile = False
dtype = 'bfloat16'
batch_size = 8
gradient_accumulation_steps = 8
max_iters = 610000
learning_rate = 2.2e-4
min_lr = 2.2e-5
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
decay_lr = True
warmup_iters = 20000
lr_decay_iters = 800000

# data
csv_folder = "./data/"
# Optional separate snapshot of data_files/; empty uses csv_folder.
# Reproduction requires the corrected >100-patient loss masks in that metadata.
loss_metadata_folder = ''
data_subset = 1.0
temporal_decay = 0.5
multiwin = True
agg_labels = False
perc = 0

# eval
eval_interval = 500
eval_iters = 8
log_interval = 1
save_interval = 150000
required_checkpoint_iter = 600000
