# eval_ntp_seqrollout_auroc_cached_duration.py
import os, json, math, argparse, time, datetime, csv
from glob import glob
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import GPTConfig, GPT, KVState
from eval_dataset import SeqCLSDataset, GPTCONFIG_PARAMS  # assumes your existing helper still provides labels

# ---- Special tokens (no SEP; durations + codes) ----
PAD_ID, CLS_ID, MASK_ID, UNK_ID, EOW_ID = 0, 2, 3, 4, 5

# ---- Duration band (hard-coded for eval remap & horizon logic) ----
DUR_BASE = 6
DURATION_UNITS_DAYS = [180, 60, 30, 14, 7, 3, 1]  # descending "coins"
NUM_DUR  = len(DURATION_UNITS_DAYS)               # e.g., 7
CODES_START = DUR_BASE + NUM_DUR                  # start of clinical code ids
EXTRA_OFFSET_FROM_OLD = NUM_DUR                   # shift old full token ids by +NUM_DUR (old layout had +6)

def parse_args():
    ap = argparse.ArgumentParser("NTP Sequential Rollouts → AUROC/AUPRC (KV cache, duration horizon stop)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--condition_indices_path", type=str, required=True)
    ap.add_argument("--condition_name", type=str, required=True)
    ap.add_argument("--mode", type=str, default="val", choices=["val","test"])

    # DataLoader
    ap.add_argument("--batch_size", type=int, default=16, help="windows per step (per rank)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_windows", type=int, default=-1, help="-1=all (per job, split across ranks)")

    # Horizon (we stop when accumulated duration > H)
    ap.add_argument("--horizon_months", type=int, default=24)
    ap.add_argument("--month_to_days", type=float, default=30.4375)

    # Rollouts
    ap.add_argument("--R", type=int, default=100, help="rollouts per window")
    ap.add_argument("--batch_r", type=int, default=32, help="parallel rollouts per window per chunk")
    ap.add_argument("--max_H", type=int, default=512, help="optional hard cap on H=B*br (0=off)")

    # Decoding
    ap.add_argument("--max_gen_tokens", type=int, default=128)
    ap.add_argument("--topk", type=int, default=50, help="if >0, use top-k (fast)")
    ap.add_argument("--top_p", type=float, default=1.0, help="if topk==0, truncated nucleus cutoff")
    ap.add_argument("--topk_cap", type=int, default=512, help="band size for truncated nucleus")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=0.0)

    # Perf toggles
    ap.add_argument("--compile", action="store_true", help="torch.compile (PyTorch 2+)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--probe_every", type=int, default=0)

    # DDP robustness
    ap.add_argument("--backend", type=str, default="nccl", choices=["nccl","gloo"])
    ap.add_argument("--init_timeout_min", type=int, default=30)
    ap.add_argument("--gather_object_fallback", action="store_true",
                    help="Use all_gather_object for varlen (slower but safest)")
    ap.add_argument("--probe_comm", action="store_true",
                    help="Print comm tensor meta just before gathers")
    return ap.parse_args()

def load_latest_checkpoint(out_dir, device):
    ckpts = glob(os.path.join(out_dir, "ckpt-*.pt"))
    if not ckpts: raise FileNotFoundError(f"No checkpoints in {out_dir}")
    ckpts.sort(key=lambda p: int(os.path.basename(p).split("-")[-1].split(".")[0]))
    path = ckpts[-1]
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    with open(os.path.join(out_dir, "config.json"), "r") as f:
        config = json.load(f)
    return checkpoint, checkpoint.get("model_args", {}), config

def build_allowed_mask(vocab_size, device):
    m = torch.ones(vocab_size, dtype=torch.bool, device=device)
    m[PAD_ID] = False
    m[EOW_ID] = False
    # durations are allowed; we need them to move time forward
    return m

# ---- Metrics (numpy) ----
def roc_auc_score_np(y_true, y_score):
    order = np.argsort(-y_score); y = y_true[order]; s = y_score[order]
    P, N = np.sum(y==1), np.sum(y==0)
    if P==0 or N==0: return float("nan")
    tp=fp=0; prev_s=None; prev_tp=prev_fp=0; auc=0.0
    for yi,si in zip(y,s):
        if prev_s is None: prev_s=si
        elif si!=prev_s:
            auc += (fp-prev_fp)*(tp+prev_tp)/2.0
            prev_s,prev_tp,prev_fp = si,tp,fp
        tp += (yi==1); fp += (yi==0)
    auc += (fp-prev_fp)*(tp+prev_tp)/2.0
    return float(auc/(P*N))

def average_precision_np(y_true, y_score):
    order = np.argsort(-y_score); y = y_true[order]; P = np.sum(y==1)
    if P==0: return float("nan")
    tp=fp=0; prev_rec=0.0; ap=0.0
    for yi in y:
        tp += (yi==1); fp += (yi==0)
        prec = tp/(tp+fp); rec = tp/P
        ap += prec*(rec-prev_rec); prev_rec = rec
    return float(ap)

# ---- Varlen all_gather (backend-aware, robust) ----
def _all_gather_varlen_1d_backend_aware(t: torch.Tensor, group, gather_object_fallback: bool, probe_comm: bool):
    if gather_object_fallback:
        arr_local = t.detach().cpu().numpy()
        obj_list = [None for _ in range(dist.get_world_size(group))]
        if probe_comm:
            print(f"[rank{dist.get_rank()}] object_gather len={arr_local.size}")
        dist.all_gather_object(obj_list, arr_local, group=group)
        if dist.get_rank(group) == 0:
            import numpy as _np
            cat = _np.concatenate(obj_list) if len(obj_list) else _np.array([], dtype=arr_local.dtype)
            return torch.from_numpy(cat)
        return None

    backend = dist.get_backend(group)
    use_cuda_comm = (backend == "nccl")
    t = t.contiguous().view(-1)
    t_comm = t if use_cuda_comm else t.cpu()

    n_local = torch.tensor([t_comm.numel()], dtype=torch.long, device=t_comm.device)
    world = dist.get_world_size(group)
    lens = [torch.empty(1, dtype=torch.long, device=t_comm.device) for _ in range(world)]
    if probe_comm:
        print(f"[rank{dist.get_rank()}] backend={backend} use_cuda_comm={use_cuda_comm} n_local={int(n_local.item())} dtype={t_comm.dtype} dev={t_comm.device}")
    dist.all_gather(lens, n_local, group=group)
    sizes = [int(x.item()) for x in lens]
    maxn = max(sizes) if sizes else 0

    if t_comm.numel() < maxn:
        pad = torch.empty(maxn - t_comm.numel(), dtype=t_comm.dtype, device=t_comm.device)
        t_comm = torch.cat([t_comm, pad], dim=0)

    recvs = [torch.empty(maxn, dtype=t_comm.dtype, device=t_comm.device) for _ in range(world)]
    dist.all_gather(recvs, t_comm, group=group)

    if dist.get_rank(group) == 0:
        chunks = [rc[:s] for rc, s in zip(recvs, sizes)]
        out = torch.cat(chunks, dim=0)
        return out
    return None

def all_gather_varlen_1d(t: torch.Tensor, ddp: bool, gather_object_fallback: bool=False, probe_comm: bool=False):
    if not ddp: return t
    group = dist.group.WORLD
    out0 = _all_gather_varlen_1d_backend_aware(t, group, gather_object_fallback, probe_comm)
    return out0  # None on non-zero ranks

# ---- Samplers ----
@torch.no_grad()
def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    if temperature>0: logits = logits/temperature
    vals, idx = torch.topk(logits, min(topk, logits.shape[-1]), dim=-1)
    probs = torch.softmax(vals, dim=-1)
    sel = torch.multinomial(probs, 1).squeeze(-1)
    return idx.gather(1, sel.unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def sample_truncated_nucleus(logits: torch.Tensor, top_p: float, topk_cap: int, temperature: float):
    if temperature > 0:
        logits = logits / temperature
    k = min(topk_cap, logits.size(-1))
    vals, idx = torch.topk(logits, k, dim=-1)                 # [B, k]
    row_all_neginf = torch.isneginf(vals).all(dim=-1)         # [B]
    if row_all_neginf.any():
        vals[row_all_neginf, 0] = 0
    probs = torch.softmax(vals, dim=-1)                       # [B, k]
    cdf = torch.cumsum(probs, dim=-1)                         # [B, k]
    B = cdf.size(0)
    if top_p >= 1.0:
        keep_counts = torch.full((B,), k, device=cdf.device, dtype=torch.long)
    else:
        counts_le = (cdf <= top_p).sum(dim=-1)
        keep_counts = torch.clamp(counts_le + 1, min=1, max=k)
    ar = torch.arange(k, device=logits.device).unsqueeze(0)    # [1, k]
    keep = ar < keep_counts.unsqueeze(1)                       # [B, k]
    kept = vals.masked_fill(~keep, float("-inf"))
    kept_probs = torch.softmax(kept, dim=-1)
    sel = torch.multinomial(kept_probs, 1).squeeze(-1)
    return idx.gather(1, sel.unsqueeze(-1)).squeeze(-1)

# ---- Duration helpers ----
@torch.no_grad()
def is_duration(tok_ids: torch.Tensor) -> torch.Tensor:
    return (tok_ids >= DUR_BASE) & (tok_ids < DUR_BASE + NUM_DUR)

@torch.no_grad()
def dur_days_from_ids(tok_ids: torch.Tensor) -> torch.Tensor:
    # maps duration ids to day values; non-duration → 0
    idx = (tok_ids - DUR_BASE).clamp(min=0, max=NUM_DUR-1)
    table = torch.tensor(DURATION_UNITS_DAYS, device=tok_ids.device, dtype=torch.long)
    vals = torch.where(is_duration(tok_ids), table[idx], torch.zeros_like(tok_ids))
    return vals

# ---- Context extraction (right-trim to last non-PAD) ----
def extract_ctx_batch(tokens_b: torch.Tensor):
    B, T = tokens_b.size()
    lens = (tokens_b != PAD_ID).sum(dim=1).clamp(min=1)        # at least 1
    Lmax = int(lens.max().item())
    ctx_tok = torch.full((B, Lmax), PAD_ID, dtype=torch.long, device=tokens_b.device)
    for i in range(B):
        L = int(lens[i].item())
        ctx_tok[i, :L] = tokens_b[i, :L]
    return ctx_tok, lens

# ---- KV allocation ----
def alloc_kv_states(model, batch_H: int, t_max: int, device):
    nh = model.config.n_head
    hd = model.config.n_embd // nh
    param_dtype = next(model.parameters()).dtype
    kv_states = []
    for _ in range(model.config.n_layer):
        k = torch.empty(batch_H, nh, t_max, hd, device=device, dtype=param_dtype)
        v = torch.empty(batch_H, nh, t_max, hd, device=device, dtype=param_dtype)
        kv_states.append(KVState(k, v, t0=0))
    return kv_states

# ---- Core: KV-cached sequential rollouts with horizon stop (duration accumulation) ----
@torch.no_grad()
def batch_seqrollout_probs_cached_duration(
    model, allowed_mask,
    ctx_tok_b,                                  # (B, Lctx) tokens only (right-padded)
    cond_token_set,                             # set[int] code token ids to detect
    horizon_days: int,                          # stop when accumulated duration > H
    R, batch_r, max_H,
    max_gen_tokens, topk, top_p, topk_cap, temperature,
    repetition_penalty
):
    device = ctx_tok_b.device
    B, Lctx = ctx_tok_b.size()
    BS = model.config.block_size
    V = model.config.vocab_size

    # Truncate to block size once
    Lctx = min(Lctx, BS)
    ctx_tok_b = ctx_tok_b[:, :Lctx]

    # Condition mask
    cond_mask = torch.zeros(V, dtype=torch.bool, device=device)
    if cond_token_set: cond_mask[list(cond_token_set)] = True

    hits_per_window = torch.zeros(B, dtype=torch.int32, device=device)

    # Prefill prefix up to last-1
    Tctxm1 = max(1, Lctx - 1)
    prefill_tokens = ctx_tok_b[:, :Tctxm1]
    last_ctx_token_B = ctx_tok_b[:, Tctxm1:Tctxm1+1]

    remain = R
    while remain > 0:
        br = min(batch_r, remain); remain -= br
        H = B * br
        if max_H and H > max_H:
            br = max(1, max_H // max(1, B)); H = B * br

        # Replicate across rollouts
        prefill_tok_H = prefill_tokens.repeat_interleave(br, 0)
        last_ctx_token = last_ctx_token_B.repeat_interleave(br, 0)

        # Allocate KV
        t_max = min(BS, prefill_tok_H.size(1) + 1 + max_gen_tokens)
        kv_states = alloc_kv_states(model, batch_H=H, t_max=t_max, device=device)

        # Prefill
        _ = model(prefill_tok_H, kv_states=kv_states)

        # Per-sample rollout state
        acc_days = torch.zeros(H, dtype=torch.long, device=device)
        hit = torch.zeros(H, dtype=torch.bool, device=device)
        finished = torch.zeros(H, dtype=torch.bool, device=device)

        cur = last_ctx_token
        steps = 0
        while steps < max_gen_tokens and (~finished).any():
            logits, _ = model(cur, kv_states=kv_states)   # T=1 step; appends to cache
            next_logits = logits[:, -1, :]

            # Allowed mask (forbid PAD/EOW)
            next_logits = next_logits.masked_fill(~allowed_mask, float("-inf"))

            # (Optional) repetition penalty on the last token only (lightweight)
            if repetition_penalty and repetition_penalty > 1.0:
                # discourage repeating the same token just emitted
                rp_idx = cur.squeeze(-1)
                gather = next_logits.gather(1, rp_idx)
                penalized = torch.where(gather > 0, gather / repetition_penalty, gather * repetition_penalty)
                next_logits.scatter_(1, rp_idx.unsqueeze(1), penalized)

            # Sample
            if topk and topk > 0:
                next_id = sample_topk(next_logits, topk, temperature)
            else:
                next_id = sample_truncated_nucleus(next_logits, top_p, topk_cap, temperature)

            # Update states only for active rows
            active = ~finished
            if active.any():
                ids_a = next_id[active]

                # durations advance time
                dur_mask = is_duration(ids_a)
                if dur_mask.any():
                    acc_days[active] += dur_days_from_ids(ids_a)

                # code hits counted only if still within horizon
                code_mask = ~dur_mask
                if code_mask.any():
                    ids_codes = ids_a[code_mask]
                    within = acc_days[active][code_mask] <= horizon_days
                    if within.any():
                        idx_keep = torch.nonzero(within, as_tuple=False).squeeze(-1)
                        if idx_keep.numel() > 0:
                            ids_codes_keep = ids_codes[idx_keep]
                            hit_sub = cond_mask[ids_codes_keep]
                            # mark hit for those rows
                            hit_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)[code_mask][idx_keep]
                            hit[hit_idx] |= hit_sub

                # finish condition: crossed horizon, EOW, or we already hit (early-exit for speed)
                crossed = acc_days > horizon_days
                end_tok = (next_id == EOW_ID)
                finished |= crossed | end_tok | hit

            # next input
            cur = next_id.view(-1, 1)
            steps += 1

        # Aggregate back to B windows (sum over br)
        hits_chunk = hit.view(B, br).sum(dim=1)
        hits_per_window += hits_chunk

    return (hits_per_window.float() / float(R)).detach()

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(
            backend=args.backend,
            timeout=datetime.timedelta(minutes=int(args.init_timeout_min))
        )
        rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda": torch.cuda.set_device(device)
        master = (rank == 0)
        if args.backend == "nccl" and device.type != "cuda":
            raise RuntimeError("NCCL backend requires CUDA device; use --backend gloo for CPU.")
    else:
        rank = 0; world = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        master = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        from torch.nn.attention import sdpa_kernel
        sdpa_ctx = sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    except Exception:
        from contextlib import nullcontext
        sdpa_ctx = nullcontext()

    # Model
    checkpoint, ckpt_args, _ = load_latest_checkpoint(args.out_dir, device)
    margs = {k: ckpt_args[k] for k in GPTCONFIG_PARAMS if k in ckpt_args}
    for k in ["block_size","vocab_size","n_layer","n_head","n_embd","dropout","bias","rotary","use_xpos"]:
        if k not in margs: margs[k] = ckpt_args[k]
    model = GPT(GPTConfig(**margs)).to(device)
    state = checkpoint["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {(k[len("_orig_mod."): ] if k.startswith("_orig_mod.") else k): v for k,v in state.items()}
    model.load_state_dict(state, strict=True); model.eval()
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            if master: print("[compile] enabled")
        except Exception as e:
            if master: print(f"[compile] disabled: {e}")
    if ddp and device.type == "cuda":
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    elif ddp:
        model = DDP(model)
    raw_model = model.module if ddp else model

    # Condition tokens (self-contained hard remap: +NUM_DUR)
    with open(args.condition_indices_path, "r") as f:
        cond_map = json.load(f)
    if args.condition_name not in cond_map:
        raise KeyError(f"{args.condition_name} not in {args.condition_indices_path}")
    raw_ids = [int(i) for i in cond_map[args.condition_name]]
    cond_token_set = set(int(i) + EXTRA_OFFSET_FROM_OLD for i in raw_ids)
    # sanity: all condition ids should be in code band (>= CODES_START)
    assert all(i >= CODES_START for i in cond_token_set), \
        f"Some condition ids < CODES_START={CODES_START}; check JSON or offset."
    
    # after loading checkpoint + building model
    print("[ckpt] vocab_size from ckpt_args:", ckpt_args.get("vocab_size"))
    print("[model] model.config.vocab_size   :", raw_model.config.vocab_size)

    # Data (expects SeqCLSDataset to return tokens + labels; ignore any days/pids if present)
    ds = SeqCLSDataset(mode=args.mode, folder=args.data_path, max_length=raw_model.config.block_size, num_patients=-1)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if ddp else None
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=args.num_workers, pin_memory=(device.type=="cuda"), drop_last=False)

    # Per-rank cap (keeps collectives aligned when max_windows is used)
    local_cap = None if (args.max_windows is None or args.max_windows < 0) else int(math.ceil(args.max_windows / world))

    # Horizon in days
    horizon_days = int(round(args.horizon_months * args.month_to_days))

    allowed_mask = build_allowed_mask(raw_model.config.vocab_size, device)

    print("[mask] allowed_mask size          :", allowed_mask.numel())

    t0 = time.time()
    local_scores, local_labels, local_pids = [], [], []
    processed_local = 0

    for batch in loader:
        if local_cap is not None and processed_local >= local_cap: break
        tokens_t = batch["token_arr"]
        labels_t = batch["labels"].view(-1) if "labels" in batch else batch["label"].view(-1)

        # Safely extract patient IDs
        if "patient_ids" in batch:
            pids_t = batch["patient_ids"].view(-1)
        elif "patient_id" in batch:
            pids_t = batch["patient_id"].view(-1)
        else:
            pids_t = torch.zeros_like(labels_t) # Fallback if missing

        if not torch.is_tensor(tokens_t): tokens_t = torch.from_numpy(np.array(tokens_t))
        if not torch.is_tensor(pids_t):   pids_t   = torch.from_numpy(np.array(pids_t))
        
        tokens_t = tokens_t.to(device, torch.long)
        labels_t = labels_t.to(device, torch.long)
        pids_t   = pids_t.to(device, torch.long)

        B = tokens_t.size(0)
        remaining = (local_cap - processed_local) if local_cap is not None else None
        if remaining is not None:
            remaining = max(0, int(remaining))
            if remaining < B:
                tokens_t = tokens_t[:remaining] 
                labels_t = labels_t[:remaining]
                pids_t   = pids_t[:remaining]
                B = remaining
            if B == 0: break

        # build context up to last non-PAD
        ctx_tok_b, lens_b = extract_ctx_batch(tokens_t)

        with sdpa_ctx:
            probs_b = batch_seqrollout_probs_cached_duration(
                model=raw_model, allowed_mask=allowed_mask,
                ctx_tok_b=ctx_tok_b,
                cond_token_set=cond_token_set,
                horizon_days=horizon_days,
                R=args.R, batch_r=args.batch_r, max_H=args.max_H,
                max_gen_tokens=args.max_gen_tokens,
                topk=args.topk, top_p=args.top_p, topk_cap=args.topk_cap, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )

        local_scores.extend(probs_b.tolist())
        local_labels.extend(labels_t.tolist())
        local_pids.extend(pids_t.tolist())
        processed_local += B

        if args.probe_every and (processed_local % max(args.probe_every,1) == 0) and (not ddp or dist.get_rank()==0):
            print(f"[probe] windows={processed_local} avg_prob={float(probs_b.mean()):.4f}")

    # Ensure all ranks reach gathers even if they processed zero windows
    if len(local_scores) == 0:
        local_scores_t = torch.empty(0, device=device, dtype=torch.float32)
        local_labels_t = torch.empty(0, device=device, dtype=torch.int64)
        local_pids_t   = torch.empty(0, device=device, dtype=torch.int64)
    else:
        local_scores_t = torch.tensor(local_scores, device=device, dtype=torch.float32)
        local_labels_t = torch.tensor(local_labels, device=device, dtype=torch.int64)
        local_pids_t   = torch.tensor(local_pids, device=device, dtype=torch.int64)

    # Gather to rank0 (backend-aware); always call on all ranks in the same order
    gathered_scores = all_gather_varlen_1d(local_scores_t, ddp,
                                           gather_object_fallback=args.gather_object_fallback,
                                           probe_comm=args.probe_comm)
    gathered_labels = all_gather_varlen_1d(local_labels_t, ddp,
                                           gather_object_fallback=args.gather_object_fallback,
                                           probe_comm=args.probe_comm)
    gathered_pids   = all_gather_varlen_1d(local_pids_t, ddp,
                                           gather_object_fallback=args.gather_object_fallback,
                                           probe_comm=args.probe_comm)

    if ddp:
        dist.barrier()

    if (not ddp) or (dist.get_rank()==0):
        if ddp:
            scores = (gathered_scores if gathered_scores is not None else torch.empty(0)).detach().cpu().numpy()
            labels = (gathered_labels if gathered_labels is not None else torch.empty(0, dtype=torch.int64)).detach().cpu().numpy().astype(int)
            pids   = (gathered_pids if gathered_pids is not None else torch.empty(0, dtype=torch.int64)).detach().cpu().numpy().astype(int)
        else:
            scores = local_scores_t.detach().cpu().numpy()
            labels = local_labels_t.detach().cpu().numpy().astype(int)
            pids   = local_pids_t.detach().cpu().numpy().astype(int)

        wids = np.arange(len(scores))

        auroc = roc_auc_score_np(labels, scores) if scores.size else float("nan")
        auprc = average_precision_np(labels, scores) if scores.size else float("nan")
        elapsed = time.time() - t0
        print("=== NTP Sequential Rollouts (KV-cached, duration horizon stop) ===")
        print(f"condition         : {args.condition_name}")
        print(f"mode              : {args.mode}")
        print(f"windows_run       : {scores.size}")
        print(f"R per window      : {args.R}   batch_r={args.batch_r}   batch_size={args.batch_size}   max_H={args.max_H}")
        print(f"horizon_months    : {args.horizon_months}  (~days={horizon_days})")
        print(f"sampling          : temp={args.temperature} top_p={args.top_p} topk={args.topk} topk_cap={args.topk_cap} rep_pen={args.repetition_penalty}")
        print(f"max_gen_tokens    : {args.max_gen_tokens}")
        print(f"time_elapsed_sec  : {elapsed:.2f}")
        print(f"AUROC             : {auroc:.4f}")
        print(f"AUPRC             : {auprc:.4f}")

        # JSON save
        out = {
            "condition": args.condition_name,
            "settings": {
                "mode": args.mode,
                "R": args.R, "batch_r": args.batch_r, "batch_size": args.batch_size, "max_H": args.max_H,
                "horizon_months": args.horizon_months, "month_to_days": args.month_to_days,
                "temperature": args.temperature, "top_p": args.top_p, "topk": args.topk, "topk_cap": args.topk_cap,
                "max_gen_tokens": args.max_gen_tokens, "repetition_penalty": args.repetition_penalty
            },
            "windows": int(scores.size),
            "metrics": {"auroc": auroc, "auprc": auprc}
        }
        base_name = f"cached_seqrollout_duration_{args.condition_name.replace(' ','_')}_{args.mode}_R{args.R}_B{args.batch_size}_br{args.batch_r}_H{args.horizon_months}m"
        out_path = os.path.join(args.out_dir, f"{base_name}.json")
        with open(out_path, "w") as f: json.dump(out, f, indent=2)
        print(f"Saved JSON metrics: {out_path}")

        # CSV save for bootstrapping
        csv_path = os.path.join(args.out_dir, f"{base_name}_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", "total_months", "window_id", "patient_id", "label", "pred"])
            for w, p, l, pr in zip(wids, pids, labels, scores):
                # Note: keeping header "total_months" to match previous formatting, passing horizon_months
                writer.writerow([args.condition_name, args.horizon_months, w, p, l, pr])
        print(f"Saved detailed predictions for bootstrapping: {csv_path}")

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()