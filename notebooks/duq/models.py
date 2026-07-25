"""A faithful *small* DiT with TREAD-style token routing.

Design follows Peebles & Xie, "Scalable Diffusion Models with Transformers":
patchify -> sequence of transformer blocks with **AdaLN-Zero** conditioning on
(timestep, class) -> linear unpatchify.  Two additions matter for Task 1:

  1. Every attention block can *return its attention map* A^(l) (softmax weights),
     which we need for the attention-entropy proxy U_attn.
  2. TREAD routing: a contiguous block range [i, j] can be bypassed (identity)
     for a random subset of tokens.  Training under a distribution over these
     sub-networks is what makes routing variance a *legitimate* epistemic signal
     (proposal, "TREAD as a stochastic-depth training procedure").

The model predicts epsilon (the noise).  x0 is recovered analytically in
`duq.diffusion`.  All proxy code in `duq.proxies` treats the model as a black box
`(x_t, t, y, route) -> eps`, so it transfers unchanged to the real DiT-XL/2.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelCfg


# --------------------------------------------------------------- embeddings ---
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal timestep embedding (as in DiT)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.hidden = hidden

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.hidden))


# ---------------------------------------------------------------- attention ---
class Attention(nn.Module):
    """Multi-head self-attention that can return its (softmax) attention map."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attn: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, H, N, hd)
        attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        attn = attn.softmax(dim=-1)                    # A^(l): (B, H, N, N)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        if return_attn:
            return out, attn
        return out, None


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning (DiT)."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"),
                                 nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, return_attn=False):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.ada(c).chunk(6, dim=1)
        a_out, attn = self.attn(modulate(self.norm1(x), shift_a, scale_a),
                                return_attn=return_attn)
        x = x + gate_a.unsqueeze(1) * a_out
        x = x + gate_m.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_m, scale_m))
        return x, attn


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ------------------------------------------------------------- routing spec ---
@dataclass
class Route:
    """A TREAD route: bypass blocks [i..j] for the tokens in `routed_mask`."""
    i: int
    j: int
    routed_mask: torch.Tensor    # bool (N,) or (B, N); True => token is routed out

    @staticmethod
    def sample(cfg: ModelCfg, batch: int, n_patches: int,
               device, generator: torch.Generator | None = None) -> "Route":
        k = int(round(cfg.selection_rate * n_patches))
        scores = torch.rand(batch, n_patches, device=device, generator=generator)
        idx = scores.argsort(dim=1)[:, :k]            # k lowest scores => routed
        mask = torch.zeros(batch, n_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, True)
        return Route(cfg.route_i, cfg.route_j, mask)

    @staticmethod
    def full_path(n_patches: int, device) -> "Route":
        """No routing: standard full-path inference (routed_mask all False)."""
        return Route(0, -1, torch.zeros(1, n_patches, dtype=torch.bool, device=device))


# ----------------------------------------------------------------- the DiT ----
class TinyDiT(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.patch = cfg.patch
        self.out_ch = cfg.in_ch
        gh = cfg.img_size // cfg.patch
        self.grid = gh
        self.n_patches = gh * gh

        self.x_embed = nn.Conv2d(cfg.in_ch, cfg.hidden, cfg.patch, cfg.patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, cfg.hidden))
        self.t_embed = TimestepEmbedder(cfg.hidden)
        self.y_embed = nn.Embedding(cfg.num_classes + 1, cfg.hidden)  # +1 = null
        self.null_class = cfg.num_classes
        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden, cfg.heads) for _ in range(cfg.depth)])
        self.final = FinalLayer(cfg.hidden, cfg.patch, self.out_ch)
        self._init()

    def _init(self):
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.y_embed.weight, std=0.02)
        for b in self.blocks:                       # AdaLN-Zero: zero the gates
            nn.init.zeros_(b.ada[-1].weight); nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight); nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight); nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.grid, self.grid, self.patch, self.patch, self.out_ch)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, self.out_ch, self.grid * self.patch, self.grid * self.patch)

    def forward(self, x, t, y, route: Route | None = None, return_attn: bool = False):
        """x:(B,C,H,W) t:(B,) y:(B,) -> eps:(B,C,H,W).

        If `route` is given, tokens flagged in route.routed_mask bypass blocks
        [i..j] (identity).  If `return_attn`, also returns stacked attention maps
        of shape (depth, B, heads, N, N).
        """
        B = x.shape[0]
        h = self.x_embed(x).flatten(2).transpose(1, 2) + self.pos   # (B, N, C)
        c = self.t_embed(t) + self.y_embed(y)

        rmask = None
        if route is not None and route.routed_mask.any():
            rmask = route.routed_mask
            if rmask.dim() == 1:
                rmask = rmask.unsqueeze(0)
            if rmask.shape[0] == 1 and B > 1:
                rmask = rmask.expand(B, -1)
            rmask = rmask.to(h.device)

        attns = [] if return_attn else None
        for bi, blk in enumerate(self.blocks):
            routed_here = (route is not None and rmask is not None
                           and route.i <= bi <= route.j)
            if routed_here:
                # routed tokens bypass this block (identity); others updated
                new_h, attn = blk(h, c, return_attn=return_attn)
                keep = (~rmask).unsqueeze(-1)                 # (B, N, 1)
                h = torch.where(keep, new_h, h)
            else:
                h, attn = blk(h, c, return_attn=return_attn)
            if return_attn:
                attns.append(attn)

        eps = self.unpatchify(self.final(h, c))
        if return_attn:
            return eps, torch.stack(attns)   # (depth, B, H, N, N)
        return eps

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------- EMA -------
class EMA:
    """Exponential moving average of model weights (better samples)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
