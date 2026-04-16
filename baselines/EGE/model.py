# model.py
# GPT Language Model with KV-caching support, Absolute RoPE, and Training Utilities

import math
import inspect
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use the RotaryEmbedding from rotary.py
try:
    from rotary import RotaryEmbedding
except Exception:
    RotaryEmbedding = None


# ----------------------------
# KV cache holder
# ----------------------------
class KVState:
    """Per-layer KV cache with in-place append and a write pointer t."""
    def __init__(self, k: torch.Tensor, v: torch.Tensor, t0: int = 0):
        self.k, self.v, self.t = k, v, t0
    @property
    def slice(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k[:, :, :self.t, :], self.v[:, :, :self.t, :]
    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        _, _, Tn, _ = k_new.shape
        t = self.t
        self.k[:, :, t:t+Tn, :].copy_(k_new)
        self.v[:, :, t:t+Tn, :].copy_(v_new)
        self.t += Tn


# ----------------------------
# Layers
# ----------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x): return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

def rope_apply(rope: RotaryEmbedding, q: torch.Tensor, k: torch.Tensor, *, offset: int):
    """Unified RoPE: same single-tensor rotate for full (offset=0) & cached (offset=t).
    Q/K are (B, H, T, D) so seq_dim=-2."""
    qf, kf = q.float(), k.float()
    qf = rope.rotate_queries_or_keys(qf, seq_dim=-2, offset=offset)
    kf = rope.rotate_queries_or_keys(kf, seq_dim=-2, offset=offset)
    return qf.to(q.dtype), kf.to(k.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.dropout, self.bias = config.dropout, config.bias
        self.use_rope = bool(getattr(config, "rotary", True))

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            if RotaryEmbedding is None:
                raise ImportError("rotary.py not found, but rotary=True was set.")
            self.rope = RotaryEmbedding(dim=config.n_embd // config.n_head, use_xpos=config.use_xpos)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias_mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )

    def forward(self, x, kv_state: Optional[KVState] = None):
        B, T, C = x.size()
        nh, hs = self.n_head, C // self.n_head

        q_lin, k_lin, v_lin = self.c_attn(x).split(C, dim=2)
        q = q_lin.view(B, T, nh, hs).transpose(1, 2)  # (B,H,T,hs)
        k = k_lin.view(B, T, nh, hs).transpose(1, 2)
        v = v_lin.view(B, T, nh, hs).transpose(1, 2)

        if kv_state is None:
            # Full pass
            if self.use_rope: q, k = rope_apply(self.rope, q, k, offset=0)
            if self.flash:
                y_heads = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None, is_causal=True,
                    dropout_p=self.dropout if self.training else 0.0
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
                causal = self.bias_mask[:, :, :T, :T]
                att = att.masked_fill(~causal.bool(), float("-inf"))
                att = torch.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y_heads = att @ v
            y = y_heads.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.c_proj(y))

        # Incremental decode
        offset = kv_state.t
        if self.use_rope: q, k = rope_apply(self.rope, q, k, offset=offset)
        kv_state.append(k.to(kv_state.k.dtype), v.to(kv_state.v.dtype))

        k_ctx, v_ctx = kv_state.slice
        Tq, S = q.size(-2), k_ctx.size(-2)
        if self.flash:
            causal = torch.ones((Tq, S), device=q.device, dtype=torch.bool).tril(S - Tq)
            attn_bias = torch.full((B, nh, Tq, S), float("-inf"), device=q.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(causal.expand(B, nh, -1, -1), 0.0)
            y_heads = F.scaled_dot_product_attention(
                q, k_ctx, v_ctx,
                attn_mask=attn_bias, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            att = (q @ k_ctx.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            causal = torch.ones((1, 1, Tq, S), device=att.device, dtype=torch.bool).tril(S - Tq)
            att = att.masked_fill(~causal, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y_heads = att @ v_ctx

        y = y_heads.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x): return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x, kv: Optional[KVState] = None):
        x = x + self.attn(self.ln_1(x), kv_state=kv)
        x = x + self.mlp(self.ln_2(x))
        return x


# ----------------------------
# Config & Model
# ----------------------------
@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    rotary: bool = True
    use_xpos: bool = False

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                kv_states: Optional[List[KVState]] = None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"
        x = self.transformer.drop(self.transformer.wte(idx))

        for li, blk in enumerate(self.transformer.h):
            kv = kv_states[li] if kv_states is not None else None
            x = blk(x, kv=kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def crop_block_size(self, block_size: int):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, "bias_mask"):
                block.attn.bias_mask = block.attn.bias_mask[:, :, :block_size, :block_size]

    # ----- Optimizer -----
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra = dict(fused=True) if use_fused else {}
        opt = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)
        return opt

    # ----- Metrics -----
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    # ----- Generation (simple) -----
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def alloc_kv(self, B: int, T: int, device=None, dtype=None) -> List[KVState]:
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        H = self.config.n_head; hd = self.config.n_embd // H
        kvs: List[KVState] = []
        for _ in self.transformer.h:
            k = torch.empty((B, H, T, hd), device=device, dtype=dtype)
            v = torch.empty((B, H, T, hd), device=device, dtype=dtype)
            kvs.append(KVState(k, v, t0=0))
        return kvs
