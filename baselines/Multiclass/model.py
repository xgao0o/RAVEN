# model.py
# GPT Language Model with KV-caching support, RoPE, and Training Utilities

import math
import inspect
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional RoPE dependency
try:
    from rotary import RotaryEmbedding
except Exception:
    RotaryEmbedding = None


# ----------------------------
# KV cache (in-place) holder
# ----------------------------
class KVState:
    """Per-layer KV cache with in-place append and a write pointer."""
    def __init__(self, k: torch.Tensor, v: torch.Tensor, t0: int = 0):
        """
        k, v: preallocated tensors [B, n_head, T_max, head_dim]
        t0:   current valid length (0 for empty; <= T_max)
        """
        self.k = k
        self.v = v
        self.t = t0

    @property
    def slice(self):
        return self.k[:, :, :self.t, :], self.v[:, :, :self.t, :]

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        k_new, v_new: [B, n_head, T_new, head_dim]
        Appends in-place and advances the write pointer.
        """
        B, H, Tn, D = k_new.shape
        self.k[:, :, self.t:self.t + Tn, :].copy_(k_new)
        self.v[:, :, self.t:self.t + Tn, :].copy_(v_new)
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


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head, self.n_embd, self.dropout, self.bias = config.n_head, config.n_embd, config.dropout, config.bias

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Rotary
        self.use_rope = bool(getattr(config, "rotary", False))
        self.use_xpos = bool(getattr(config, "use_xpos", False))
        if self.use_rope:
            if RotaryEmbedding is None:
                raise ImportError("rotary_embedding not found, but rotary=True was set.")
            self.rope = RotaryEmbedding(dim=config.n_embd // config.n_head, use_xpos=self.use_xpos)

        # Flash SDPA path
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias_mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )

    def forward(self, x, days: Optional[torch.Tensor] = None, kv_state: Optional[KVState] = None):
        """
        x: (B, T, C)
        days: (B, T) integer days (used for RoPE/XPos)
        kv_state: if provided, appends K/V in-place and attends over prefix [0:kv_state.t]
        """
        B, T, C = x.size()
        nh, hs = self.n_head, C // self.n_head

        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # RoPE rotation for current chunk
        if self.use_rope:
            if self.use_xpos:
                q, k = self.rope.rotate_queries_and_keys(q, k, days)
            else:
                q = self.rope.rotate_queries_or_keys(q, days)
                k = self.rope.rotate_queries_or_keys(k, days)

        if kv_state is None:
            # full prefill over the current chunk only
            k_ctx, v_ctx = k, v
            t_total = T
        else:
            # append chunk (often T=1) and attend over entire prefix via slicing (no cat)
            cache_dtype = kv_state.k.dtype
            if q.dtype != cache_dtype:
                q = q.to(cache_dtype)
                k = k.to(cache_dtype)
                v = v.to(cache_dtype)

            # append new kv in-place; then attend over the whole prefix (slice, no cat)
            kv_state.append(k, v)
            k_ctx, v_ctx = kv_state.slice
            t_total = kv_state.t
        
        if self.flash:
            # Flash SDPA path.
            T  = q.size(-2)          # query length for this chunk
            Tp = k_ctx.size(-2)      # total keys in context (prefix [+ current])
            if kv_state is None:
                # Full square causal attention: built-in causal is ok.
                y = F.scaled_dot_product_attention(
                    q, k_ctx, v_ctx,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                # Build keep matrix (T, Tp) and convert to additive mask shape (B, nh, T, Tp)
                causal_keep_2d = torch.ones((T, Tp), device=q.device, dtype=torch.bool).tril(diagonal=Tp - T)
                B, nh = q.size(0), q.size(1)
                attn_bias = torch.full((B, nh, T, Tp), float("-inf"), device=q.device, dtype=q.dtype)
                attn_bias[causal_keep_2d.expand(B, nh, -1, -1)] = 0.0  # 0 where allowed, -inf where masked

                y = F.scaled_dot_product_attention(
                    q, k_ctx, v_ctx,
                    attn_mask=attn_bias,                     # additive mask
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,                         # mask encodes causality
                )

        else:
            att = (q @ k_ctx.transpose(-2, -1)) * (1.0 / math.sqrt(k_ctx.size(-1)))
            Tp = t_total
            causal = torch.ones((1, 1, T, Tp), device=att.device, dtype=torch.bool).tril(diagonal=Tp - T)
            att = att.masked_fill(~causal, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v_ctx

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


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
    def forward(self, x, days: Optional[torch.Tensor] = None, kv_state: Optional[KVState] = None):
        x = x + self.attn(self.ln_1(x), days=days, kv_state=kv_state)
        x = x + self.mlp(self.ln_2(x))
        return x


# ----------------------------
# Config
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
    rotary: bool = False
    use_xpos: bool = False


# ----------------------------
# Model
# ----------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie weights
        self.transformer.wte.weight = self.lm_head.weight

        # init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    # ----- Utilities -----
    def get_num_params(self, non_embedding: bool = True):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Additive sinusoid for days/patient_ids when not using RoPE
    def get_batch_positional_embeddings(self, batch_days: torch.Tensor):
        B, T = batch_days.shape
        C = self.config.n_embd
        pe = torch.zeros(B, T, C, device=batch_days.device, dtype=torch.float32)
        div = torch.exp(torch.arange(0, C, 2, device=batch_days.device).float() * -(np.log(10000.0) / C))
        pe[:, :, 0::2] = torch.sin(batch_days[:, :, None] * div)
        pe[:, :, 1::2] = torch.cos(batch_days[:, :, None] * div)
        return pe

    # ----- Forward -----
    def forward(
        self,
        idx: torch.Tensor,                          # (B, T)
        targets: Optional[torch.Tensor] = None,     # (B, T) or None
        days: Optional[torch.Tensor] = None,        # (B, T) int
        patient_ids: Optional[torch.Tensor] = None, # (B, T) int (zeros if unused)
        kv_states: Optional[List[KVState]] = None   # list[KVState] or None
    ):
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        tok = self.transformer.wte(idx)  # (B,T,C)

        pos_add = 0
        if days is not None and not self.config.rotary:
            pos_add = self.get_batch_positional_embeddings(days)

        pat_add = 0
        if patient_ids is not None:
            pat_add = self.get_batch_positional_embeddings(patient_ids)

        x = self.transformer.drop(tok + pos_add + pat_add)

        for li, block in enumerate(self.transformer.h):
            kv = None if kv_states is None else kv_states[li]
            x = block(x, days=days if self.config.rotary else None, kv_state=kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    # ----- Surgery -----
    def crop_block_size(self, block_size: int):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.attn, "bias_mask"):
                block.attn.bias_mask = block.attn.bias_mask[:, :, :block_size, :block_size]

    # ----- HF Interop -----
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args), "Only 'dropout' can be overridden."
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")
        config_args = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args: config_args["dropout"] = override_args["dropout"]
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias_mask")]
        hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad(): sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad(): sd[k].copy_(sd_hf[k])
        return model

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

    # ----- Simple generation -----
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
