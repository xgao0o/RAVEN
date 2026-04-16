# test_loss.py
import os
import glob
import json
import argparse
from typing import Tuple
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from tqdm import tqdm

from raven.model import GPTConfig, GPT
from raven.dataset import UnifiedSeqEHRDataset, custom_collate

GPTCONFIG_PARAMS = [
    'block_size','vocab_size','n_layer','n_head','n_embd','dropout','bias',
    'gamma','alpha','loss_type','pos_weight','loss_weighting','loss_top_perc',
    'rotary','use_xpos'
]

def parse_args():
    p = argparse.ArgumentParser("Compute test-set loss for a trained model.")
    p.add_argument('--out_dir', type=str, required=True, help='Directory with checkpoints and config.json')
    p.add_argument('--folder', type=str, required=True, help='Data folder used by UnifiedSeqEHRDataset')
    p.add_argument('--csv_folder', type=str, required=True, help='Folder containing split CSVs (e.g., test.csv)')
    p.add_argument('--split', type=str, default='test', help='CSV split name: test/val/train (default: test)')
    p.add_argument('--iteration', type=int, default=None, help='Evaluate specific ckpt-<iter>.pt; if omitted uses latest')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--limit_batches', type=int, default=0, help='Eval at most this many batches (0 = all)')
    p.add_argument('--test_patients', type=int, default=-1, help='Subsample N patients for quick eval (-1 = all)')
    p.add_argument('--seed', type=int, default=1037, help='Seed for subsampling test_patients')
    return p.parse_args()

def ddp_setup():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank, world_size, rank = 0, 1, 0
        master = True
    return ddp, device, local_rank, world_size, rank, master

def load_latest_checkpoint(out_dir, device, iteration=None):
    if iteration is not None:
        ckpts = glob.glob(os.path.join(out_dir, f'ckpt-{iteration}.pt'))
        if not ckpts:
            raise FileNotFoundError(f"Checkpoint ckpt-{iteration}.pt not found in {out_dir}")
    else:
        ckpts = glob.glob(os.path.join(out_dir, 'ckpt-*.pt'))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint files found in {out_dir}")
        ckpts.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    ckpt_path = ckpts[-1]
    with open(os.path.join(out_dir, 'config.json'), 'r') as f:
        cfg = json.load(f)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt, ckpt['model_args'], cfg

@torch.inference_mode()
def evaluate(model: torch.nn.Module,
             dataloader: DataLoader,
             device: torch.device,
             ctx,
             ddp: bool,
             master: bool,
             limit_batches: int = 0,
             empty_cache_every: int = 200) -> Tuple[float, int]:
    model.eval()
    # total_loss = torch.tensor(0.0, device=device)
    # total_count = torch.tensor(0, device=device, dtype=torch.long)
    total_loss = 0.0
    total_count = 0

    seen = 0
    for i, batch in enumerate(tqdm(dataloader, disable=not master)):

        token_arr = batch["token_arr"].to(device)
        days_embed_arr = batch["days_embed_arr"].to(device)
        labels = batch["labels"].to(device)
        pred_mask = batch["pred_mask"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        attention_bias = batch["attention_bias"].to(device)
        patient_ids = batch.get("patient_ids", None)
        if patient_ids is not None:
            patient_ids = patient_ids.to(device)

        bs = token_arr.shape[0]
        with ctx:
            out = model(token_arr, days_embed_arr, labels,
                                  pred_mask, pad_mask, attention_bias,
                                  patient_ids=patient_ids)
        total_loss += out[1].item() * bs
        total_count += bs

        seen += 1
        if limit_batches and seen >= limit_batches:
            break

        # free up
        del token_arr, days_embed_arr, labels, pred_mask, pad_mask, attention_bias, patient_ids, out
        if empty_cache_every and ((i + 1) % empty_cache_every == 0):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # After the loop, create tensors on the device for DDP reduction.
    total_loss = torch.tensor(total_loss, device=device)
    total_count = torch.tensor(total_count, device=device, dtype=torch.long)

    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

    n = int(total_count.item())
    avg_loss = (total_loss / total_count.clamp(min=1)).item()
    if master:
        print(f"[eval] Samples: {n}")
    return avg_loss, n

def main():
    args = parse_args()
    ddp, device, local_rank, world_size, rank, master = ddp_setup()

    # TF32 on by default (matches training script spirit)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt, ckpt_model_args, train_cfg = load_latest_checkpoint(args.out_dir, device, args.iteration)
    # dtype & autocast context as in training
    dtype_str = train_cfg.get('dtype', 'float16')
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    ptdtype = dtype_map.get(dtype_str, torch.float16)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Build model config strictly from saved args
    model_args = {k: ckpt_model_args[k] for k in GPTCONFIG_PARAMS}
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Load weights (handle _orig_mod. keys)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Dataset
    agg_labels = train_cfg.get('agg_labels', False)
    multiwin = train_cfg.get('multiwin', False)

    csv_path = os.path.join(args.csv_folder, f"{args.split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.test_patients != -1:
        df = df.sample(n=args.test_patients, replace=False, random_state=args.seed).reset_index(drop=True)

    dataset = UnifiedSeqEHRDataset(
        df, max_length=model_args["block_size"], model='GPT',
        folder=args.folder, age=None, train=False,
        agg_labels=agg_labels, multiwin=multiwin
    )

    if ddp:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        collate_fn=custom_collate,
        num_workers=args.num_workers
    )

    if ddp:
        dist.barrier()

    avg_loss, n = evaluate(
        model=model, dataloader=dataloader, device=device, ctx=ctx,
        ddp=ddp, master=master, limit_batches=args.limit_batches
    )

    if master:
        ckpt_name = f"ckpt-{args.iteration}.pt" if args.iteration else "latest"
        # write out to file
        with open(os.path.join(args.out_dir, f'test_loss_{ckpt_name}.txt'), 'w') as f:
            f.write(f"loss: {avg_loss:.6f}\n")
            f.write(f"samples: {n}\n")
        print(f"\n=== Test Loss ({ckpt_name}) ===")
        print(f"loss: {avg_loss:.6f}  |  samples: {n}")

    if ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()