# inference_utils.py
import torch
from model import KVState

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

@torch.no_grad()
def generate_rollouts_grpo(
    model, 
    ctx_tok_b, ctx_day_b, lvd_b, 
    horizon_days,   
    group_size,     
    max_gen_tokens, 
    temperature=1.0, 
    top_k=50
):
    """
    Generates G rollouts per patient using KV-caching.
    Returns: 
       generated_seqs: [B * G, max_gen_tokens] (Only the NEW tokens)
    """
    device = ctx_tok_b.device
    B, Lctx = ctx_tok_b.shape
    H = B * group_size
    BS = model.config.block_size
    SEP_ID = 1 

    # 1. Expand Inputs for Group Size
    prefill_tokens = ctx_tok_b.repeat_interleave(group_size, dim=0)
    prefill_days   = ctx_day_b.repeat_interleave(group_size, dim=0)
    prefill_pids   = torch.zeros_like(prefill_days) 
    
    # 2. Setup KV Cache
    # t_max = context + gen + 2 buffer
    t_max = min(BS, Lctx + max_gen_tokens + 2) 
    kv_states = alloc_kv_states(model, batch_H=H, t_max=t_max, device=device)

    # 3. Prefill (Process History)
    _ = model(prefill_tokens, targets=None, days=prefill_days, patient_ids=prefill_pids, kv_states=kv_states)

    # 4. Setup for Generation (Jump to Horizon)
    # Day = Last Visit Date + Horizon
    lvd_rep = lvd_b.repeat_interleave(group_size)
    day_next = (lvd_rep + horizon_days).view(H, 1)
    
    # Prompt with the last token of history to trigger the next step
    curr_token = prefill_tokens[:, -1].view(H, 1) 
    
    generated_ids = []
    
    # 5. Autoregressive Loop
    for _ in range(max_gen_tokens):
        # Forward pass 
        logits, _ = model(curr_token, targets=None, days=day_next, patient_ids=None, kv_states=kv_states)
        next_logits = logits[:, -1, :] 
        
        # Sampling
        if temperature > 0:
            next_logits = next_logits / temperature
        
        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        next_logits[next_logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # [H, 1]
        
        generated_ids.append(next_token)
        curr_token = next_token
        # Note: Day stays constant for the rollout (single visit assumption)
    
    gen_seq = torch.cat(generated_ids, dim=1) # [H, max_gen]
    return gen_seq