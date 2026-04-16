# train.py
# Next-Token Prediction training script (DDP-friendly)
# - Duration tokens between visits; EOW between stitched windows
# - RoPE over absolute positions (no day/patient embeddings)
# - CE loss with ignore_index=-1
# - NO validation loop; logs train loss; saves checkpoints at save_interval

import os
import math
import time
import json
import glob
from contextlib import nullcontext

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from dataset import UnifiedSeqEHRDataset, custom_collate

# =======================
# Env (keep keys out of code in practice)
# =======================
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# ------------- User-editable paths -------------
csv_folder = "./data/"
train_csv = os.path.join(csv_folder, "csvs/train.csv")

# ----------------------- Duration vocab knobs -----------------------
# Must match dataset constants
DURATION_UNITS_DAYS = [180, 60, 30, 14, 7, 3, 1]  # descending coins
NUM_DUR = len(DURATION_UNITS_DAYS)

# ----------------------- Default config ------------------------------
out_dir = "out_ntp_duration_lr_1e-4"
save_interval = 25000
log_interval = 1
init_from = "scratch"   # 'scratch' | 'resume'

# wandb (off by default)
wandb_log = False
wandb_project = "gpt-ntp-ehrshot"
wandb_run_name = "run-duration"

# data / loader
gradient_accumulation_steps = 8     # effective batch size = this * batch_size
batch_size = 8
block_size = 512
num_workers = 0          # shelve safety
pin_memory = True
multiwin = True          # use multi-patient training windows

# model
n_tokens_base = 42337                 # base code vocab (without specials)
special_tokens = 6                    # reserve first 6 (PAD, <old SEP slot>, CLS, MASK, UNK, EOW)
vocab_size = n_tokens_base + special_tokens + NUM_DUR
n_layer = 8
n_head = 8
n_embd = 1024
dropout = 0.0
bias = False

# rotary embeddings (absolute positions)
rotary = True
use_xpos = False

# adamw optimizer
learning_rate = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule (cosine)
decay_lr = True
warmup_iters = 10_000
lr_decay_iters = 500000
min_lr = 1e-6
max_iters = 500000

# system
backend = "nccl"
device = "cuda"
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# allow CLI override via configurator.py (optional)
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
if os.path.exists('configurator.py'):
    exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

print("**************************")
print(out_dir)

# ----------------------- DDP / device setup --------------------------
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ----------------------- Dataset / Dataloader ------------------------
train_df = pd.read_csv(train_csv)[["part", "patient_id"]]

train_dataset = UnifiedSeqEHRDataset(
    df=train_df, max_length=block_size, model='GPT',
    folder=csv_folder, train=True, multiwin=multiwin
)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if ddp else None

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler,
    shuffle=(train_sampler is None), collate_fn=custom_collate,
    num_workers=num_workers, pin_memory=pin_memory, drop_last=False
)

# ----------------------- Model init / resume -------------------------
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, vocab_size=vocab_size,
    dropout=dropout, rotary=rotary, use_xpos=use_xpos
)

def load_latest_checkpoint(dirname, device):
    ckpts = glob.glob(os.path.join(dirname, "ckpt-*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found")
    ckpts.sort(key=lambda p: int(os.path.basename(p).split("-")[-1].split(".")[0]))
    path = ckpts[-1]
    print("Loading", path)
    return torch.load(path, map_location=device)

iter_num = 0

if init_from == "scratch":
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    checkpoint = load_latest_checkpoint(out_dir, device)
    chk_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "rotary", "use_xpos"]:
        model_args[k] = chk_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
else:
    raise ValueError("init_from must be 'scratch' or 'resume'")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size

model.to(device)

# scaler for fp16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

# compile (PyTorch 2.0+)
if compile:
    print("Compiling model...")
    model = torch.compile(model)

# DDP wrap
if ddp:
    model = DDP(model, device_ids=[int(device.split(":")[-1])])

raw_model = model.module if ddp else model

# ----------------------- LR schedule --------------------------------
def get_lr(it):
    if not decay_lr: return learning_rate
    if it < warmup_iters: return learning_rate * (it / max(1, warmup_iters))
    if it > lr_decay_iters: return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ----------------------- wandb --------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ----------------------- Training loop -------------------------------
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    if ddp and train_sampler is not None:
        train_sampler.set_epoch(local_iter_num)

    for batch in train_loader:
        # set lr
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        tokens  = batch["token_arr"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        # forward + loss (with grad accumulation)
        with ctx:
            logits, loss = model(tokens, targets=targets)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        # optimizer step when we complete one macro-batch
        if (iter_num + 1) % gradient_accumulation_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # logging
        if iter_num % log_interval == 0 and master_process:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item() * gradient_accumulation_steps  # undo micro scaling
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.6f}, time {dt*1000:.1f}ms, mfu {running_mfu*100:.2f}%")
            if wandb_log:
                import wandb
                wandb.log({"iter": iter_num, "train/loss": lossf, "lr": lr, "mfu": running_mfu * 100})

        # ---------- periodic checkpoint ----------
        if (iter_num > 0) and (iter_num % save_interval == 0) and master_process:
            print(f"saving checkpoint to {out_dir}")
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": None,
                "config": config,
            }
            torch.save(checkpoint, os.path.join(out_dir, f"ckpt-{iter_num}.pt"))

        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
