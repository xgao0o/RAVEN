# rotary_embedding_torch.py
# Lightweight RoPE / XPos utilities tailored for (B, H, T, D_head) tensors
# Compatible with the GPT model provided (rotate by 'days' integers per token)

from math import pi
from typing import Optional, Union, Literal

import torch
from torch import einsum
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat


# -----------------------
# helpers
# -----------------------

def exists(x):
    return x is not None


def rotate_half(x):
    """
    Split last dim into pairs and rotate (x1, x2) -> (-x2, x1)
    Shapes:
      x: (..., 2*k)
      out: (..., 2*k)
    """
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


@autocast(enabled=False)
def apply_rotary_emb(freqs: torch.Tensor,
                     t: torch.Tensor,
                     start_index: int = 0,
                     scale: Union[float, torch.Tensor] = 1.,
                     seq_dim: int = -2):
    """
    Apply rotary embedding to contiguous block of features in `t`.

    Args:
      freqs: (..., T, rot_dim) or (T, rot_dim) cosine/sine phases (interleaved) to apply
      t:     (..., T, C) where the last dim has a sub-block [start_index : start_index+rot_dim]
      scale: scalar or (..., T, rot_dim) multiplicative scale (for XPos)
      seq_dim: dimension for sequence length (default -2 for (B, H, T, D_head))

    Returns:
      tensor of same shape as `t`
    """
    if t.ndim == 3:
        # (B, T, D) case
        T = t.shape[seq_dim]
        freqs = freqs[..., -T:, :].to(t)
    else:
        # (B, H, T, D_head) case
        T = t.shape[seq_dim]
        freqs = freqs[..., -T:, :].to(t)

    # expand freqs to heads if needed
    # expected shapes after this:
    #   t: (B, H, T, D_head)
    #   freqs: (B, 1, T, rot_dim)  or (1, 1, T, rot_dim)
    if t.ndim == 4:
        # freqs may be (B, T, rot_dim) or (T, rot_dim)
        if freqs.ndim == 2:
            freqs = freqs.unsqueeze(0).unsqueeze(0)        # (1,1,T,rot_dim)
        elif freqs.ndim == 3:
            freqs = freqs.unsqueeze(1)                     # (B,1,T,rot_dim)

        if isinstance(scale, torch.Tensor):
            if scale.ndim == 3:
                # (B, T, rot_dim) -> (B,1,T,rot_dim)
                scale = scale.unsqueeze(1)
        elif isinstance(scale, (int, float)):
            pass

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], (
        f"feature dim {t.shape[-1]} too small to rotate {rot_dim} dims "
        f"[start={start_index}, end={end_index})"
    )

    t_left, t_mid, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

    if isinstance(scale, torch.Tensor):
        t_mid = (t_mid * freqs.cos() * scale) + (rotate_half(t_mid) * freqs.sin() * scale)
    else:
        t_mid = (t_mid * freqs.cos() * scale) + (rotate_half(t_mid) * freqs.sin() * scale)

    return torch.cat((t_left, t_mid, t_right), dim=-1)


# -----------------------
# main module
# -----------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary embeddings over per-token 'days' scalars.

    - freqs_for='lang' yields the standard 1 / theta^(2i/d) spectrum
    - use_xpos=True enables XPos scaling (Chen et al.) to improve length extrapolation
    - seq_before_head_dim=False assumes tensors shaped (B, H, T, D_head)

    API used by the GPT:
      - rotate_queries_or_keys(q, days)      # when use_xpos=False
      - rotate_queries_and_keys(q, k, days)  # when use_xpos=True (needs scale asymmetry)
    """

    def __init__(
        self,
        dim: int,
        custom_freqs: Optional[torch.Tensor] = None,
        freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        theta: float = 10000.0,
        max_freq: float = 10.0,
        num_freqs: int = 1,
        learned_freq: bool = False,
        use_xpos: bool = False,
        xpos_scale_base: float = 512.0,
        interpolate_factor: float = 1.0,
        theta_rescale_factor: float = 1.0,
        seq_before_head_dim: bool = False,
        cache_if_possible: bool = False
    ):
        super().__init__()

        # NTK-style rescale
        theta *= (theta_rescale_factor ** (dim / max(dim - 2, 1)))

        # construct base frequencies
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # standard RoPE spectrum
            freqs = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"Unknown freqs_for {freqs_for}")

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.learned_freq = learned_freq

        self.use_xpos = use_xpos
        self.scale_base = xpos_scale_base

        # optional XPos base scale per feature pair
        if use_xpos:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
            self.register_buffer('xpos_scale', scale, persistent=False)
        else:
            self.register_buffer('xpos_scale', None, persistent=False)

        self.interpolate_factor = max(1.0, float(interpolate_factor))
        self.seq_before_head_dim = seq_before_head_dim
        self.cache_if_possible = cache_if_possible

        # caches (non-persistent)
        self.register_buffer('cached_freqs', None, persistent=False)
        self.register_buffer('cached_scales', None, persistent=False)
        self.register_buffer('dummy', torch.tensor(0), persistent=False)

    # ------------- utilities -------------

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len: int, device, dtype, offset: int = 0):
        # used when generating “positions”; unused here because we feed real `days`
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    # ------------- core ops -------------

    @autocast(enabled=False)
    def forward(self, t: torch.Tensor, seq_len: Optional[int] = None, offset: int = 0):
        """
        Convert token-wise scalar inputs t (e.g., days) to interleaved cos/sin phases.

        Args:
          t: (B, T) or (T,)
          seq_len / offset: kept for API compatibility; enables caching

        Returns:
          phases: same shape as t with last dim expanded to 2*(dim//2)
        """
        should_cache = (self.cache_if_possible and not self.learned_freq and exists(seq_len) and t.ndim == 1)

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        base = self.freqs  # (dim//2,)

        phases = einsum('... t, f -> ... t f', t.to(base.dtype), base)  # (..., T, dim//2)
        phases = repeat(phases, '... t f -> ... t (f r)', r=2)          # (..., T, dim)

        if should_cache:
            self.cached_freqs = phases.detach()
        return phases

    @autocast(enabled=False)
    def get_scale(self, t: torch.Tensor, seq_len: Optional[int] = None, offset: int = 0):
        """
        XPos multiplicative scale per token and feature pair.
        t: (B, T) or (T,)
        returns: (..., T, dim) duplicated over pairs
        """
        assert self.use_xpos, "get_scale called but use_xpos=False"

        should_cache = (self.cache_if_possible and exists(seq_len))
        if should_cache and exists(self.cached_scales) and (offset + seq_len) <= self.cached_scales.shape[0]:
            return self.cached_scales[offset:(offset + seq_len)]

        # Center around midpoint per sequence to avoid overflow
        # For (B,T): midpoint per batch row
        if t.ndim == 2:
            midpoint, _ = t.max(dim=1, keepdim=True)
            midpoint = midpoint // 2
            power = (t.float() - midpoint.float()) / self.scale_base  # (B,T)
        else:
            # (T,)
            midpoint = t.max().item() // 2
            power = (t.float() - float(midpoint)) / self.scale_base   # (T,)

        scale = (self.xpos_scale ** power.unsqueeze(-1))   # (..., T, dim//2)
        scale = torch.cat((scale, scale), dim=-1)          # (..., T, dim)

        if should_cache:
            self.cached_scales = scale
        return scale

    def rotate_queries_or_keys(self, t: torch.Tensor, days: torch.Tensor, start_index: int = 0, seq_dim: int = -2):
        """
        Apply RoPE to a single tensor (queries or keys). Use when use_xpos=False.

        t:    (B, H, T, D_head)
        days: (B, T) ints
        """
        assert not self.use_xpos, "Use rotate_queries_and_keys when use_xpos=True"
        # construct phases from days
        freqs = self.forward(days, seq_len=days.shape[-1])
        # expand to (B,1,T,rot_dim)
        if freqs.ndim == 2:
            freqs = freqs.unsqueeze(1)
        return apply_rotary_emb(freqs, t, start_index=start_index, seq_dim=seq_dim)

    def rotate_queries_and_keys(self, q: torch.Tensor, k: torch.Tensor, days: torch.Tensor, start_index: int = 0, seq_dim: int = -2):
        """
        Apply RoPE with XPos asymmetric scaling to both q and k.

        q, k: (B, H, T, D_head)
        days: (B, T)
        """
        assert self.use_xpos, "rotate_queries_and_keys requires use_xpos=True"
        T = days.shape[-1]
        freqs = self.forward(days, seq_len=T)
        scale = self.get_scale(days, seq_len=T).to(q.dtype)

        # expand phases and scales to heads
        if freqs.ndim == 2:
            freqs = freqs.unsqueeze(1)            # (B,1,T,rot_dim)
            scale = scale.unsqueeze(1)            # (B,1,T,rot_dim)

        q = apply_rotary_emb(freqs, q, start_index=start_index, scale=scale, seq_dim=seq_dim)
        k = apply_rotary_emb(freqs, k, start_index=start_index, scale=(scale ** -1), seq_dim=seq_dim)
        return q.type_as(q), k.type_as(k)
