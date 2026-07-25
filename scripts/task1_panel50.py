"""Task 1.4 — log U_ens / U_attn (/ U_route) over 50 ImageNet classes x 20 timestep bins.

Pretrained DiT-XL/2 (ImageNet-256), inference only, no weight modification.

One pass per class:
  * K seeds are batched, so a single p_sample_loop_progressive gives all of them;
  * at every step we take Var_k[x0_hat] and pool it onto the 16x16 patch grid
    (DiT-XL/2: patch=2 on a 32x32 latent -> 256 tokens);
  * attention entropy is captured during that same pass, so it is averaged over
    the K seeds for free instead of resting on one trajectory.

U_route is left as NaN: it requires a TREAD-trained DiT-XL/2 checkpoint, which is
not publicly released. The column exists so the schema is the Task-1 schema.

Usage:
    python scripts/task1_panel50.py                 # full 50-class run
    python scripts/task1_panel50.py --classes 951 207 --tag smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "DiT"))

from diffusion import create_diffusion  # noqa: E402
from models import DiT_XL_2  # noqa: E402

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
K_SEEDS = 16        # proposal: K = 16
STEPS = 250         # sampling steps along the reverse path
N_BINS = 20         # timestep bins (t in [0, 1000) -> centres 25, 75, ... 975)
CFG_SCALE = 1.0     # 1.0 = pure conditional denoiser, no classifier-free guidance.
                    # Guidance is an inference-time hack that inflates Var_k by a
                    # class-independent factor; at 1.0 U_ens reflects the model's
                    # own prediction. Set >1 only if the rest of the team did.
LATENT, PATCH = 32, 2
GRID = LATENT // PATCH  # 16x16 patch grid

# 50 ImageNet-1k classes, stratified easy / medium / hard.
# Criterion (stated on the slide): expected difficulty of the class-conditional
# distribution p(x|c) — how multi-modal and how cluttered it is.
#   easy   : one canonical object, plain background, essentially unimodal
#   medium : one object but real pose / lighting / background variation
#   hard   : polysemous labels, cluttered multi-object scenes, open-ended layouts
# Includes 951 (lemon) — the unimodal reference used elsewhere in the deck — and
# BOTH senses of "crane": 134 (bird) and 517 (machine), the ambiguity the
# proposal itself singles out.
CLASS_PANEL: list[tuple[int, str, str]] = [
    # ---- easy (17) ----
    (951, "lemon", "easy"),
    (950, "orange", "easy"),
    (949, "strawberry", "easy"),
    (948, "Granny Smith", "easy"),
    (954, "banana", "easy"),
    (953, "pineapple", "easy"),
    (957, "pomegranate", "easy"),
    (945, "bell pepper", "easy"),
    (937, "broccoli", "easy"),
    (988, "acorn", "easy"),
    (985, "daisy", "easy"),
    (574, "golf ball", "easy"),
    (852, "tennis ball", "easy"),
    (429, "baseball", "easy"),
    (430, "basketball", "easy"),
    (971, "bubble", "easy"),
    (892, "wall clock", "easy"),
    # ---- medium (17) ----
    (207, "golden retriever", "medium"),
    (235, "German shepherd", "medium"),
    (250, "Siberian husky", "medium"),
    (281, "tabby cat", "medium"),
    (285, "Egyptian cat", "medium"),
    (291, "lion", "medium"),
    (292, "tiger", "medium"),
    (340, "zebra", "medium"),
    (360, "otter", "medium"),
    (387, "lesser panda", "medium"),
    (388, "giant panda", "medium"),
    (88, "macaw", "medium"),
    (130, "flamingo", "medium"),
    (145, "king penguin", "medium"),
    (850, "teddy bear", "medium"),
    (817, "sports car", "medium"),
    (779, "school bus", "medium"),
    # ---- hard (16) ----
    (134, "crane (bird)", "hard"),
    (517, "crane (machine)", "hard"),
    (762, "restaurant", "hard"),
    (454, "bookshop", "hard"),
    (624, "library", "hard"),
    (582, "grocery store", "hard"),
    (865, "toyshop", "hard"),
    (973, "coral reef", "hard"),
    (974, "geyser", "hard"),
    (975, "lakeside", "hard"),
    (979, "valley", "hard"),
    (980, "volcano", "hard"),
    (972, "cliff", "hard"),
    (978, "seashore", "hard"),
    (933, "cheeseburger", "hard"),
    (963, "pizza", "hard"),
]

# --------------------------------------------------------------------------- #
# Attention-entropy capture
#
# DiT's blocks use timm's Attention, whose fused SDPA kernel never materialises
# the softmax weights. We temporarily swap in an eager forward that exposes A,
# accumulate the per-query Shannon entropy, and restore the original afterwards.
#
# We report the MEAN per-query entropy (averaged over queries, heads, seeds)
# rather than the proposal's literal sum over (i, j) — the two differ by a factor
# of N. The mean is bounded by ln(N) = ln(256) = 5.545, which gives the curve an
# interpretable ceiling ("attention is uniform / the block is not selecting").
# --------------------------------------------------------------------------- #
_ENTROPY: dict[int, float] = {}


def _eager_forward_with_entropy(layer_idx: int):
    def forward(self, x):
        B, N, C = x.shape
        nh = self.num_heads
        hd = C // nh
        qkv = self.qkv(x).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
        if hasattr(self, "k_norm"):
            k = self.k_norm(k)
        attn = (q * getattr(self, "scale", hd ** -0.5)) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)                                  # (B, nh, N, N)

        ent = -(attn * attn.clamp_min(1e-12).log()).sum(-1)          # (B, nh, N)
        _ENTROPY[layer_idx] = float(ent.mean().item())               # over seeds/heads/queries

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if hasattr(self, "proj_drop"):
            x = self.proj_drop(x)
        return x

    return forward


def _enable_capture(model):
    import types
    for i, blk in enumerate(model.blocks):
        if not hasattr(blk.attn, "_orig_forward"):
            blk.attn._orig_forward = blk.attn.forward
        blk.attn.forward = types.MethodType(_eager_forward_with_entropy(i), blk.attn)


def _disable_capture(model):
    for blk in model.blocks:
        if hasattr(blk.attn, "_orig_forward"):
            blk.attn.forward = blk.attn._orig_forward
            del blk.attn._orig_forward


# --------------------------------------------------------------------------- #
# Proxies
# --------------------------------------------------------------------------- #
def patch_variance(x0: torch.Tensor) -> torch.Tensor:
    """(K, C, H, W) x0-predictions -> (GRID, GRID) per-patch variance across seeds.

    Variance is taken across seeds FIRST, then pooled within each 2x2 latent patch
    (the DiT-XL/2 token footprint) and across channels.
    """
    var = x0.float().var(dim=0, unbiased=False)                      # (C, H, W)
    c, h, w = var.shape
    return var.reshape(c, h // PATCH, PATCH, w // PATCH, PATCH).mean(dim=(0, 2, 4))


@torch.no_grad()
def run_class(model, cls: int, device, k: int, steps: int, cfg: float):
    """One batched reverse pass; returns timesteps, per-patch U_ens maps, per-layer U_attn."""
    diff = create_diffusion(str(steps))
    depth = len(model.blocks)
    ts = [diff.timestep_map[i] for i in range(diff.num_timesteps - 1, -1, -1)]

    torch.manual_seed(1234 + cls)
    z = torch.randn(k, 4, LATENT, LATENT, device=device)

    if cfg == 1.0:
        fn = model.forward
        model_kwargs = dict(y=torch.full((k,), cls, device=device, dtype=torch.long))
        shape, z_in = z.shape, z
    else:
        fn = model.forward_with_cfg
        z_in = torch.cat([z, z], 0)
        y = torch.cat([torch.full((k,), cls, device=device, dtype=torch.long),
                       torch.full((k,), 1000, device=device, dtype=torch.long)])
        model_kwargs = dict(y=y, cfg_scale=cfg)
        shape = z_in.shape

    maps, per_layer = [], []
    _enable_capture(model)
    try:
        for out in diff.p_sample_loop_progressive(
            fn, shape, z_in, clip_denoised=False,
            model_kwargs=model_kwargs, device=device, progress=False,
        ):
            x0 = out["pred_xstart"][:k]                              # drop the null half if CFG
            maps.append(patch_variance(x0).cpu().numpy())
            per_layer.append([_ENTROPY[i] for i in range(depth)])
    finally:
        _disable_capture(model)
        _ENTROPY.clear()

    return np.array(ts), np.stack(maps), np.array(per_layer)         # (T,), (T,G,G), (T,depth)


def bin_curve(t_vals, y_vals, n_bins=N_BINS, t_max=1000):
    edges = np.linspace(0, t_max, n_bins + 1)
    idx = np.clip(np.digitize(np.asarray(t_vals, float), edges) - 1, 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = np.asarray(y_vals, float)[m].mean()
    return out, ((edges[:-1] + edges[1:]) / 2).astype(int)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, nargs="*", default=None)
    ap.add_argument("--k", type=int, default=K_SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cfg", type=float, default=CFG_SCALE)
    ap.add_argument("--tag", type=str, default="panel50")
    args = ap.parse_args()

    panel = CLASS_PANEL
    if args.classes:
        keep = set(args.classes)
        panel = [c for c in CLASS_PANEL if c[0] in keep]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiT_XL_2(input_size=LATENT).to(device)
    state = torch.load(os.path.join(REPO, "DiT", "DiT-XL-2-256x256.pt"), map_location=device)
    state = state.get("ema", state) if isinstance(state, dict) and "ema" in state else state
    model.load_state_dict(state)
    model.eval()
    depth = len(model.blocks)
    print(f"DiT-XL/2: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
          f"{depth} blocks | device={device} | K={args.k} steps={args.steps} cfg={args.cfg}")
    print(f"panel: {len(panel)} classes x {N_BINS} bins\n")

    records, heatmaps = [], {}
    t_start = time.time()
    for n, (cls, name, diff_tier) in enumerate(panel, 1):
        t0 = time.time()
        ts, maps, per_layer = run_class(model, cls, device, args.k, args.steps, args.cfg)

        u_ens_curve = maps.mean(axis=(1, 2))
        b_ens, centres = bin_curve(ts, u_ens_curve)
        b_attn, _ = bin_curve(ts, per_layer.mean(1))
        b_b0, _ = bin_curve(ts, per_layer[:, 0])
        b_bmid, _ = bin_curve(ts, per_layer[:, depth // 2])
        b_blast, _ = bin_curve(ts, per_layer[:, -1])

        for b in range(N_BINS):
            records.append({
                "class_id": cls, "class_name": name, "difficulty": diff_tier,
                "bin": b, "t_centre": int(centres[b]),
                "U_ens": b_ens[b],
                "U_route": np.nan,          # needs a TREAD DiT-XL/2 checkpoint
                "U_attn": b_attn[b],
                "U_attn_block0": b_b0[b],
                "U_attn_block14": b_bmid[b],
                "U_attn_block27": b_blast[b],
            })

        # per-patch maps at early / mid / late t, for the spatial-heatmap slide
        order = np.argsort(ts)  # ascending t
        early, mid, late = order[-1], order[len(order) // 2], order[0]
        heatmaps[f"{cls}"] = np.stack([maps[early], maps[mid], maps[late]])
        heatmaps[f"{cls}_t"] = np.array([ts[early], ts[mid], ts[late]])

        eta = (time.time() - t_start) / n * (len(panel) - n)
        print(f"[{n:2d}/{len(panel)}] {cls:4d} {name:18s} {diff_tier:6s} "
              f"U_ens {b_ens[-1]:.4f} -> {b_ens[0]:.4f} | "
              f"U_attn {b_attn[-1]:.2f} -> {b_attn[0]:.2f} "
              f"[{time.time() - t0:.0f}s, ETA {eta / 60:.0f}m]")

    out_dir = os.path.join(REPO, "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"task1_{args.tag}.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    np.savez_compressed(os.path.join(out_dir, f"task1_{args.tag}_patchmaps.npz"), **heatmaps)
    print(f"\nsaved -> {csv_path}  ({len(records)} rows, {time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
