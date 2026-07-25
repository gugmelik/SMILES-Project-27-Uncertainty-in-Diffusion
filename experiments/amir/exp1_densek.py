import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torchvision.utils import save_image

import exp1_renoise as E
import tiers as T


@torch.no_grad()
def get_trajectory(diffusion, model, cls, *, traj_dir, traj_seed, cfg_scale,
                   latent_size, device):
    pt = Path(traj_dir) / f"latents_class{cls}.pt"
    if pt.is_file():
        latents = torch.load(pt, map_location="cpu")
        x0_ref = E.denoise_from(diffusion, model, latents[0].to(device), 0,
                                denoise_seed=0, shared_noise=True, class_label=cls,
                                cfg_scale=cfg_scale, device=device)
        print(f"[densek] class {cls}: trajectory reused from cache", flush=True)
        return latents, x0_ref
    latents, x0_ref = E.generate_trajectory(
        diffusion, model, class_label=cls, traj_seed=traj_seed,
        cfg_scale=cfg_scale, latent_size=latent_size, device=device)
    Path(traj_dir).mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in latents.items()}, pt)
    print(f"[densek] class {cls}: trajectory regenerated + cached", flush=True)
    return latents, x0_ref


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--classes", type=int, nargs="+", default=T.sample_per_tier(5, seed=0),
                    help="default: 5 classes per difficulty tier (15 total)")
    ap.add_argument("--num-steps", type=int, default=250)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--ensemble", type=int, default=8, help="M — must match the tier run")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7, 10, 15, 20, 30],
                    help="dense k grid (steps re-noised backward)")
    ap.add_argument("--t-fracs", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--noise-type", type=str, default="gaussian",
                    choices=["gaussian", "uniform", "laplace"])
    ap.add_argument("--traj-seed", type=int, default=0)
    ap.add_argument("--denoise-seed", type=int, default=1000)
    ap.add_argument("--perturb-seed", type=int, default=2000,
                    help="base; member m uses perturb_seed + m (same as tier run)")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--traj-dir", type=str, default="outputs/exp1_tiers/trajectories",
                    help="cached latents_class{cls}.pt live here (reused if present)")
    ap.add_argument("--outdir", type=str, default="outputs/exp1_densek")
    args = ap.parse_args()

    device = args.device
    torch.set_grad_enabled(False)
    outdir = Path(args.outdir)
    (outdir / "grids").mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2))

    print(f"[densek] device={device}; loading DiT-XL/2 + VAE + metrics ...", flush=True)
    model, latent_size = E.load_dit(device, ckpt=args.ckpt)
    vae = E.load_vae(device)
    diffusion = E.create_diffusion(str(args.num_steps))
    metrics = E.MetricSuite(vae, device)
    n_steps = diffusion.num_timesteps
    t_map = E.steps_for_fracs(n_steps, args.t_fracs)
    print(f"[densek] classes={args.classes}\n[densek] ks={args.ks}  t_map={t_map}")

    rows = []
    for cls in args.classes:
        latents, x0_ref = get_trajectory(
            diffusion, model, cls, traj_dir=args.traj_dir, traj_seed=args.traj_seed,
            cfg_scale=args.cfg_scale, latent_size=latent_size, device=device)
        ref_img = metrics.decode(x0_ref)

        for frac, t_idx in t_map.items():
            x_t = latents[t_idx].to(device)
            for k in args.ks:
                perturbed, start = [], None
                for m in range(args.ensemble):
                    x_j, j = E.renoise(diffusion, x_t, t_idx, k,
                                       perturb_seed=args.perturb_seed + m,
                                       noise_type=args.noise_type, device=device)
                    perturbed.append(x_j)
                    start = j
                x_batch = torch.cat(perturbed, 0)
                finals = E.denoise_from(diffusion, model, x_batch, start,
                                        denoise_seed=args.denoise_seed, shared_noise=True,
                                        class_label=cls, cfg_scale=args.cfg_scale,
                                        device=device)
                row, imgs, _ = metrics.ensemble_row(finals, x0_ref)
                row.update(dict(class_id=cls, mode="perturb", k=k, noise_type=args.noise_type,
                                t_frac=frac, t_idx=t_idx, start_idx=start))
                rows.append(row)
                save_image(torch.cat([ref_img, imgs]),
                           outdir / "grids" /
                           f"cls{cls}_t{frac:.1f}_perturb_k{k}_{args.noise_type}.png",
                           nrow=args.ensemble + 1)
                print(f"  t={frac:.1f} (idx {t_idx})  perturb k={k}  "
                      f"lpips={row['lpips_pair']:.4f} clip={row['clip_pair_cosdist']:.4f}",
                      flush=True)
            pd.DataFrame(rows).to_csv(outdir / "exp1_densek.csv", index=False)

    pd.DataFrame(rows).to_csv(outdir / "exp1_densek.csv", index=False)
    print(f"[densek] done -> {outdir / 'exp1_densek.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
