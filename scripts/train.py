"""
Training script for RAVEN. Supports single-GPU and distributed data parallel (DDP).

To run on a single GPU:
$ python scripts/train.py --batch_size=32 --compile=False

To run with DDP on 4 GPUs on 1 node:
$ torchrun --standalone --nproc_per_node=4 scripts/train.py

To override defaults with a config file:
$ torchrun --standalone --nproc_per_node=4 scripts/train.py config/train_raven.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import glob

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from raven.model import GPTConfig, GPT
from torch.utils.data import DataLoader
import pandas as pd
import json
from raven.dataset import UnifiedSeqEHRDataset, custom_collate



os.environ['TORCH_USE_CUDA_DSA'] = '1'

# -----------------------------------------------------------------------------
# default config values
# I/O
n_tokens = 42337

eval_interval = 500 # how often to evaluate the validation set, in iterations
log_interval = 1
save_interval = 5000
eval_iters = 8
eval_only = False # if True, script exits right after the first eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'GPT-EHR'
wandb_run_name = 'raven'
# data
csv_folder = "./data/"

gradient_accumulation_steps = 8 # used to simulate larger batch sizes

batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512
# model
loss_weighting = "standard"
n_layer = 8
n_head = 8
n_embd = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
perc  = 0
# adamw optimizer
rotary = True # rotary embedding
use_xpos = False
gamma = 2.0
alpha = 0.995
pos_weight = 1.0
loss_type = "FL" # BCE or Focal Loss
learning_rate = 2.2e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
agg_labels = False
temporal_decay = 0.5
multiwin = False
data_subset = 1.0


grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 20000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 2.2e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
from raven.configurator import apply_overrides
apply_overrides(globals()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
if loss_type=="BCE":
    out_dir = f"scaling_{data_subset}/out_multi_win_{multiwin}_{loss_weighting}_grad_clip_{grad_clip}_lr_{learning_rate}_min_lr_{min_lr}_top_perc_{perc}_n_layer_{n_layer}_n_head_{n_head}_n_embd_{n_embd}_rotary_embedding_{rotary}_use_xpos_{use_xpos}_max_iters_{max_iters}_agg_labels_{agg_labels}_reverse_temporal_decay_{temporal_decay}_block_size_{block_size}_data_{data_subset}"
    wandb_run_name = wandb_run_name + f"_{loss_weighting}_grad_clip_{grad_clip}_lr_{learning_rate}_min_lr_{min_lr}_top_perc_{perc}_n_layer_{n_layer}_n_head_{n_head}_n_embd_{n_embd}_rotary_embedding_{rotary}_use_xpos_{use_xpos}_max_iters_{max_iters}_agg_labels_{agg_labels}_reverse_temporal_decay_{temporal_decay}_block_size_{block_size}_data_{data_subset}"
else:
    out_dir = f"scaling_{data_subset}/out_multi_win_{multiwin}_{loss_weighting}_grad_clip_{grad_clip}_lr_{learning_rate}_min_lr_{min_lr}_top_perc_{perc}_n_layer_{n_layer}_n_head_{n_head}_n_embd_{n_embd}_rotary_embedding_{rotary}_use_xpos_{use_xpos}_agg_labels_{agg_labels}_reverse_temporal_decay_{temporal_decay}_block_size_{block_size}_data_{data_subset}"
    wandb_run_name = wandb_run_name + f"_{loss_weighting}_grad_clip_{grad_clip}_lr_{learning_rate}_min_lr_{min_lr}_top_perc_{perc}_n_layer_{n_layer}_n_head_{n_head}_n_embd_{n_embd}_rotary_embedding_{rotary}_use_xpos_{use_xpos}_agg_labels_{agg_labels}_reverse_temporal_decay_{temporal_decay}_block_size_{block_size}_data_{data_subset}"


def load_latest_checkpoint(out_dir, device):
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(out_dir, 'ckpt-*.pt'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {out_dir}")
    
    # Sort checkpoint files by step number
    checkpoint_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    
    # Get the latest checkpoint file
    latest_checkpoint = checkpoint_files[-1]
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    return checkpoint, checkpoint_model_args

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    config_file_path = os.path.join(out_dir, 'config.json')

    # Serialize config to a JSON file
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
train_df = pd.read_csv(os.path.join(csv_folder, "csvs", "train.csv"))
if data_subset<1.0:
    train_df = train_df.sample(frac=data_subset, random_state=1337)
print("total training size:", train_df.shape[0])
val_df = pd.read_csv(os.path.join(csv_folder, "csvs", "val.csv"))

train_dataset = UnifiedSeqEHRDataset(train_df, max_length=block_size, model='GPT', folder = csv_folder, age=None, train = True, temporal_decay = temporal_decay, multiwin = multiwin)
val_dataset = UnifiedSeqEHRDataset(val_df, max_length=block_size, model='GPT', folder = csv_folder, age=None, train = False, temporal_decay = None, multiwin = multiwin)

sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=custom_collate)
loaders = {"train":train_dataloader, "val":val_dataloader}


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, gamma = gamma,loss_type=loss_type, alpha=alpha, pos_weight = pos_weight, loss_weighting = loss_weighting, rotary = rotary, use_xpos = use_xpos, loss_top_perc= perc, data_dir=csv_folder) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
   
    model_args['vocab_size'] = n_tokens + 6
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    checkpoint, checkpoint_model_args = load_latest_checkpoint(out_dir, device)
    print("Latest checkpoint loaded successfully")

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

print(f"device : {device}")
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) #,find_unused_parameters=True)

# helps estimate an arbitrarily accurate loss over either split using many batches
def is_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def log_nan_info(name, tensor):
    if is_nan_or_inf(tensor):
        print(f"NaN or Inf detected in {name}")
        print(f"Shape: {tensor.shape}")
        print(f"Min: {tensor.min().item()}, Max: {tensor.max().item()}")
        print(f"Mean: {tensor.mean().item()}, Std: {tensor.std().item()}")


@torch.inference_mode()
def estimate_loss():
    out = {}
    
    model.eval()
    for split in ['val']:

        data = loaders[split]
        k = 0
        losses = 0
        total = 0
        for batch in data:
            token_arr, days_embed_arr, labels, pred_mask, pad_mask, attention_bias, patient_ids = batch["token_arr"], batch["days_embed_arr"],batch["labels"],batch["pred_mask"],batch["pad_mask"],batch["attention_bias"], batch['patient_ids']
            token_arr,days_embed_arr,labels,pred_mask,pad_mask,attention_bias,patient_ids = token_arr.to(device),days_embed_arr.to(device),labels.to(device),pred_mask.to(device),pad_mask.to(device),attention_bias.to(device),patient_ids.to(device)
            bs = token_arr.shape[0]
            total+=bs
            with ctx:
                _, loss, _, _ = model(token_arr,days_embed_arr,labels,pred_mask,pad_mask,attention_bias,patient_ids=patient_ids)
            losses += loss.item() * bs

            k+=1
            if k>=eval_iters:
                break
        losses = losses/total
        out[split] = losses

    model.train()
    return out

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, save_code=True,config=config)

# training loop



t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
all_losses = {"train":[], "val":[]}

do_train = True

epoch = 0
micro_step = 0
while do_train:
    sampler.set_epoch(epoch)
    for batch in loaders["train"]:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num == 0 and eval_only:
            break

        token_arr, days_embed_arr, labels, pred_mask, pad_mask, attention_bias = (
            batch["token_arr"].to(device), batch["days_embed_arr"].to(device),
            batch["labels"].to(device), batch["pred_mask"].to(device),
            batch["pad_mask"].to(device), batch["attention_bias"].to(device),
        )
        temporal_weights = None
        patient_ids = None
        if 'temporal_weights' in batch and temporal_decay is not None:
            temporal_weights = batch['temporal_weights'].to(device)
        if 'patient_ids' in batch:
            patient_ids = batch['patient_ids'].to(device)

        if ddp:
            # Only sync on last micro step of the macro step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss, loss_components, _ = model(token_arr, days_embed_arr, labels, pred_mask, pad_mask, attention_bias, temporal_weights, patient_ids)
            loss = loss / gradient_accumulation_steps  # Scale the loss based on number of sub-batches
            
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        micro_step += 1

        total_norm = 0.0
        if micro_step == gradient_accumulation_steps:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if master_process and (iter_num % log_interval == 0):
                    total_norm = grad_norm_tensor.item()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps

            all_losses["train"].append(lossf)

            if wandb_log:
                log_data = {
                    "iter": iter_num,
                    "epoch": epoch,
                    "lr": lr,
                    "grad_norm": total_norm,
                    "step_train/loss": lossf,
                    "batch_size": len(batch["token_arr"]),
                }
                log_data.update(loss_components)
                wandb.log(log_data)

            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, batch_size {len(batch['token_arr'])}")
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(len(batch["token_arr"]), dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu


        # ——— run validation every eval_interval iters ———
        if iter_num % eval_interval == 0 and master_process:
            val_stats = estimate_loss()           # runs on eval_iters batches
            val_loss = val_stats["val"]
            all_losses["val"].append(val_loss)
            print(f"iter {iter_num}: val loss {val_loss:.4f}")

            if wandb_log:
                log_data = {
                    "iter": iter_num,
                    "epoch": epoch,
                    "step_val/loss": val_loss,
                }
                wandb.log(log_data)


        iter_num += 1
        local_iter_num += 1
        if iter_num>0 and iter_num % save_interval == 0 and master_process:
            print(f"saving checkpoint to {out_dir}")
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt-{iter_num}.pt'))
            checkpoint = None  # free memory
            with open(os.path.join(out_dir, 'loss_logs.pkl'), 'wb') as f:
                pickle.dump(all_losses, f)
        elif master_process and iter_num % (log_interval * 100) == 0:
            # periodic loss-log snapshot, much less frequent than every iter
            with open(os.path.join(out_dir, 'loss_logs.pkl'), 'wb') as f:
                pickle.dump(all_losses, f)

        # termination conditions
        if iter_num > max_iters:
            do_train = False
            break
    epoch += 1

if ddp:
    destroy_process_group()
