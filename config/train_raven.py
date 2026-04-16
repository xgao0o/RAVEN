# Config for training RAVEN on EHR data.
# Launch with DDP on 4 GPUs (example):
# $ torchrun --standalone --nproc_per_node=4 scripts/train.py config/train_raven.py

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
batch_size = 16
gradient_accumulation_steps = 8
max_iters = 600000
learning_rate = 2.2e-4
min_lr = 2.2e-5
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
decay_lr = True
warmup_iters = 20000
lr_decay_iters = 600000

# data
csv_folder = "./data/"
data_subset = 1.0
temporal_decay = 0.5
multiwin = False
agg_labels = False

# eval
eval_interval = 500
eval_iters = 8
log_interval = 1
save_interval = 5000
