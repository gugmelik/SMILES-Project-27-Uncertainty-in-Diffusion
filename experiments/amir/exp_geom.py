"""
Geometric perturbations of the latent (companion to Experiment 1).

Instead of adding *noise* to a saved latent, we apply a controlled *geometric*
transform — rotation, translation, scaling, horizontal flip, shear — then
denoise to the end and see how the final image responds. A geometric transform
is deterministic given its parameter, so we probe it two ways:

  sweep     For a fixed checkpoint latent x_t, apply the transform at a range of
            magnitudes with the denoising noise held fixed, and measure the
            DEVIATION of each final image from the untransformed reference
            (CLIP / DINOv2 / LPIPS / latent-MSE). Also measures an EQUIVARIANCE
            residual: how close final(T(x_t)) is to T(final(x_t)) — i.e. does a
            geometric move in latent space produce the same geometric move in
            image space?

  ensemble  Draw M random transform parameters within a bound, denoise all with
            a shared denoising seed, and measure the pairwise SPREAD of the
            ensemble — the same uncertainty framing as Experiment 1, so the
            results drop straight into spread-vs-checkpoint plots.

Model/sampling/metric infrastructure is imported from exp1_renoise.py, which
must sit next to this file. Runs across the difficulty tiers in tiers.py.

Typical run (server):
    python experiments/amir/exp_geom.py --classes-per-tier 3
    python experiments/amir/plot_geom.py
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# exp1_renoise sets up the DiT import path as a side effect of import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1_renoise import (  # noqa: E402
    load_dit, load_vae, generate_trajectory, denoise_from, MetricSuite,
    steps_for_fracs,
)
from diffusion import create_diffusion  # noqa: E402
import tiers as T  # noqa: E402

LATENT_W = 32  # DiT-XL/2 @ 256px -> 32x32 latent; translation normalised by this


# --------------------------------------------------------------------------- #
# Geometric transforms (work identically on a 32px latent or a 256px image)
# --------------------------------------------------------------------------- #
def _affine_theta(kind, param, device, dtype):
    """2x3 affine matrix for F.affine_grid. Coordinates are normalised to
    [-1, 1], so the SAME matrix applied to a latent and to an image is the same
    geometric move — which is what makes the equivariance check meaningful."""
    if kind == "rotation":
        a = math.radians(param)
        c, s = math.cos(a), math.sin(a)
        m = [[c, -s, 0.0], [s, c, 0.0]]
    elif kind == "translation":
        dx, dy = param if isinstance(param, (tuple, list)) else (param, 0.0)
        m = [[1.0, 0.0, 2.0 * dx / LATENT_W], [0.0, 1.0, 2.0 * dy / LATENT_W]]
    elif kind == "scaling":              # param > 1 => zoom in
        s = 1.0 / float(param)
        m = [[s, 0.0, 0.0], [0.0, s, 0.0]]
    elif kind == "flip":                 # param in {0, 1}: horizontal flip
        f = -1.0 if param else 1.0
        m = [[f, 0.0, 0.0], [0.0, 1.0, 0.0]]
    elif kind == "shear":                # x-direction shear by param
        m = [[1.0, float(param), 0.0], [0.0, 1.0, 0.0]]
    else:
        raise ValueError(f"unknown transform: {kind}")
    return torch.tensor(m, device=device, dtype=dtype)


def geometric_transform(x, kind, param, padding_mode="reflection"):
    n = x.shape[0]
    theta = _affine_theta(kind, param, x.device, x.dtype).unsqueeze(0).expand(n, -1, -1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode,
                         align_corners=False)


def is_identity(kind, param):
    return (kind in ("rotation", "translation", "shear") and param == 0) \
        or (kind == "scaling" and param == 1.0) \
        or (kind == "flip" and not param)


# sweep magnitudes per transform (identity value included as the anchor)
SWEEP_PARAMS = {
    "rotation":    [0, 5, 10, 20, 30, 45],       # degrees
    "translation": [0, 1, 2, 4, 6, 8],           # latent pixels, along x
    "scaling":     [0.8, 0.9, 1.0, 1.1, 1.25, 1.5],
    "flip":        [0, 1],
    "shear":       [0.0, 0.05, 0.1, 0.2, 0.3],
}
# ensemble bounds per transform (random param drawn in +/- bound); flip excluded
# because it is binary and has no magnitude
ENSEMBLE_LEVELS = {
    "rotation":    [10, 20, 30],                 # +/- degrees
    "translation": [2, 4, 6],                    # +/- latent pixels (dx, dy)
    "scaling":     [0.1, 0.2, 0.3],              # +/- fraction around 1.0
    "shear":       [0.1, 0.2, 0.3],              # +/- shear
}


def random_param(kind, level, gen, device):
    def u(a, b):
        return (torch.rand((), generator=gen, device=device).item() * (b - a)) + a
    if kind == "rotation":
        return u(-level, level)
    if kind == "translation":
        return (u(-level, level), u(-level, level))
    if kind == "scaling":
        return u(1.0 - level, 1.0 + level)
    if kind == "shear":
        return u(-level, level)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Deviation metrics for a batch of finals against one reference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def deviation_row(metrics, final_latents, ref_latent, ref_img, kind, params, device):
    """Per-member distance to the untransformed reference, plus equivariance."""
    imgs = metrics.decode(final_latents)
    clip = metrics.clip_feats(imgs)
    dino = metrics.dino_feats(imgs)
    ref_c = metrics.clip_feats(ref_img)
    ref_d = metrics.dino_feats(ref_img)

    clip_dev = (1 - clip @ ref_c.T).squeeze(1)          # [n]
    dino_dev = (1 - dino @ ref_d.T).squeeze(1)
    x = imgs * 2 - 1
    r = (ref_img * 2 - 1).expand_as(x).contiguous()
    lpips_dev = torch.stack([metrics.lpips(x[i:i + 1], r[i:i + 1]).flatten()
                             for i in range(x.shape[0])]).flatten()
    lat_dev = ((final_latents.to(device) - ref_latent.to(device)) ** 2)
    lat_dev = lat_dev.flatten(1).mean(1)

    # equivariance: apply the SAME transform to the reference image and compare
    equi = []
    for i, p in enumerate(params):
        t_ref = geometric_transform(ref_img, kind, p)      # [1,3,256,256]
        equi.append(metrics.lpips((imgs[i:i + 1] * 2 - 1),
                                  (t_ref * 2 - 1)).item())
    return dict(clip=clip_dev.tolist(), dino=dino_dev.tolist(),
                lpips=lpips_dev.tolist(), latmse=lat_dev.tolist(),
                equi=equi), imgs


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--classes", type=int, nargs="+",
                     help="explicit class ids (overrides tier sampling)")
    grp.add_argument("--classes-per-tier", type=int, default=3,
                     help="sample this many classes from each difficulty tier")
    ap.add_argument("--all-classes", action="store_true",
                    help="use all 45 tiered classes")
    ap.add_argument("--mode", choices=["sweep", "ensemble", "both"], default="both")
    ap.add_argument("--transforms", nargs="+",
                    default=["rotation", "translation", "scaling", "flip", "shear"])
    ap.add_argument("--num-steps", type=int, default=250)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--ensemble", type=int, default=8)
    ap.add_argument("--t-fracs", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                    help="fractions of denoising completed at the probed latent")
    ap.add_argument("--padding", choices=["reflection", "border", "zeros"],
                    default="reflection")
    ap.add_argument("--traj-seed", type=int, default=0)
    ap.add_argument("--denoise-seed", type=int, default=1000)
    ap.add_argument("--perturb-seed", type=int, default=3000)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-grids", action="store_true", help="skip saving image strips")
    ap.add_argument("--outdir", type=str,
                    default=str(Path(__file__).resolve().parent / "outputs" / "geom"))
    args = ap.parse_args()

    if args.classes:
        classes = args.classes
    elif args.all_classes:
        classes = T.ALL_CLASSES
    else:
        classes = T.sample_per_tier(args.classes_per_tier)

    device = args.device
    torch.set_grad_enabled(False)
    outdir = Path(args.outdir)
    (outdir / "grids").mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps({**vars(args), "classes": classes}, indent=2))

    print(f"[geom] device={device}; {len(classes)} classes; loading models ...", flush=True)
    model, latent_size = load_dit(device, ckpt=args.ckpt)
    vae = load_vae(device)
    diffusion = create_diffusion(str(args.num_steps))
    metrics = MetricSuite(vae, device)
    n_steps = diffusion.num_timesteps
    t_map = steps_for_fracs(n_steps, args.t_fracs)
    print(f"[geom] {n_steps} steps; probed latents: {t_map}", flush=True)

    def pmag(kind, param):
        """A scalar 'how big is this transform' for the x-axis of sweep plots."""
        if kind == "scaling":
            return abs(math.log(param))
        if kind == "flip":
            return float(bool(param))
        return abs(param)

    rows = []
    for cls in classes:
        tier, kid = T.tier_of(cls), T.kid_of(cls)
        print(f"[geom] class {cls} ({tier}) ...", flush=True)
        latents, x0_ref = generate_trajectory(
            diffusion, model, class_label=cls, traj_seed=args.traj_seed,
            cfg_scale=args.cfg_scale, latent_size=latent_size, device=device)

        for frac, t_idx in t_map.items():
            x_t = latents[t_idx].to(device)
            # reference: identity transform denoised with the fixed seed
            ref_final = denoise_from(diffusion, model, x_t, t_idx,
                                     denoise_seed=args.denoise_seed, shared_noise=True,
                                     class_label=cls, cfg_scale=args.cfg_scale, device=device)
            ref_img = metrics.decode(ref_final)

            for kind in args.transforms:
                # ---------------- sweep ----------------
                if args.mode in ("sweep", "both"):
                    params = SWEEP_PARAMS[kind]
                    batch = torch.cat([geometric_transform(x_t, kind, p, args.padding)
                                       for p in params], 0)
                    finals = denoise_from(diffusion, model, batch, t_idx,
                                          denoise_seed=args.denoise_seed, shared_noise=True,
                                          class_label=cls, cfg_scale=args.cfg_scale, device=device)
                    dev, imgs = deviation_row(metrics, finals, ref_final, ref_img,
                                              kind, params, device)
                    for i, p in enumerate(params):
                        rows.append(dict(
                            class_id=cls, tier=tier, kid=kid, mode="sweep", kind=kind,
                            param=float(p if not isinstance(p, tuple) else p[0]),
                            magnitude=pmag(kind, p), level=np.nan,
                            t_frac=frac, t_idx=t_idx, s=n_steps - 1 - t_idx,
                            clip_dev=dev["clip"][i], dino_dev=dev["dino"][i],
                            lpips_dev=dev["lpips"][i], latmse_dev=dev["latmse"][i],
                            equivariance_lpips=dev["equi"][i]))
                    if not args.no_grids:
                        save_image(torch.cat([ref_img, imgs]),
                                   outdir / "grids" / f"sweep_cls{cls}_t{frac:.1f}_{kind}.png",
                                   nrow=len(params) + 1, normalize=False)

                # -------------- ensemble ---------------
                if args.mode in ("ensemble", "both") and kind in ENSEMBLE_LEVELS:
                    for level in ENSEMBLE_LEVELS[kind]:
                        gen = torch.Generator(device=device).manual_seed(args.perturb_seed)
                        batch = []
                        for m in range(args.ensemble):
                            p = random_param(kind, level, gen, device)
                            batch.append(geometric_transform(x_t, kind, p, args.padding))
                        finals = denoise_from(diffusion, model, torch.cat(batch, 0), t_idx,
                                              denoise_seed=args.denoise_seed, shared_noise=True,
                                              class_label=cls, cfg_scale=args.cfg_scale, device=device)
                        row, imgs, _ = metrics.ensemble_row(finals, ref_final)
                        rows.append(dict(
                            class_id=cls, tier=tier, kid=kid, mode="ensemble", kind=kind,
                            param=np.nan, magnitude=float(level), level=float(level),
                            t_frac=frac, t_idx=t_idx, s=n_steps - 1 - t_idx,
                            clip_pair_cosdist=row["clip_pair_cosdist"],
                            dino_pair_cosdist=row["dino_pair_cosdist"],
                            lpips_pair=row["lpips_pair"],
                            latent_mse_pair=row["latent_mse_pair"]))
                        if not args.no_grids and level == ENSEMBLE_LEVELS[kind][-1]:
                            save_image(torch.cat([ref_img, imgs]),
                                       outdir / "grids" /
                                       f"ens_cls{cls}_t{frac:.1f}_{kind}_L{level}.png",
                                       nrow=args.ensemble + 1, normalize=False)

        pd.DataFrame(rows).to_csv(outdir / "geom_results.csv", index=False)

    pd.DataFrame(rows).to_csv(outdir / "geom_results.csv", index=False)
    print(f"[geom] done -> {outdir / 'geom_results.csv'}", flush=True)


if __name__ == "__main__":
    main()