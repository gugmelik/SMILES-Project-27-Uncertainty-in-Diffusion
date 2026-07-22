import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import pandas as pd
import torch
from torchvision.utils import save_image

DIT_ZIP_URL = "https://github.com/facebookresearch/DiT/archive/refs/heads/main.zip"


def _is_dit_dir(p):
    return (Path(p) / "diffusion" / "__init__.py").is_file()


def _find_dit_dir():
    env = os.environ.get("DIT_DIR")
    if env and _is_dit_dir(env):
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        if _is_dit_dir(base / "DiT"):
            return (base / "DiT").resolve()
    if _is_dit_dir(Path.cwd() / "DiT"):
        return (Path.cwd() / "DiT").resolve()

    target = here.parent / "DiT"
    print(f"[exp1] DiT code not found — downloading source into {target} ...", flush=True)
    tmp = here.parent / "_dit_download"
    shutil.rmtree(tmp, ignore_errors=True)
    with urllib.request.urlopen(DIT_ZIP_URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(tmp)
    root = next((p for p in tmp.iterdir() if _is_dit_dir(p)), None)
    if root is None:
        raise RuntimeError(
            "Downloaded archive did not contain the DiT sources. Copy the repo's "
            "DiT/ folder next to this script, or set $DIT_DIR to point at it.")
    shutil.move(str(root), str(target))
    shutil.rmtree(tmp, ignore_errors=True)
    return target.resolve()


DIT_DIR = _find_dit_dir()
sys.path.insert(0, str(DIT_DIR))

from diffusion import create_diffusion  
from download import find_model  
from models import DiT_models 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

NULL_CLASS = 1000  
LATENT_SCALE = 0.18215


def sample_unit_noise(noise_type, shape, generator, device):
    if noise_type == "gaussian":
        return torch.randn(shape, generator=generator, device=device)
    if noise_type == "uniform":
        u = torch.rand(shape, generator=generator, device=device)
        return (u - 0.5) * (12.0 ** 0.5)
    if noise_type == "laplace":
        u = torch.rand(shape, generator=generator, device=device) - 0.5
        b = 1.0 / (2.0 ** 0.5)  # variance = 2 b^2 = 1
        return -b * torch.sign(u) * torch.log1p(-2.0 * u.abs() + 1e-12)
    raise ValueError(f"unknown noise type: {noise_type}")


@torch.no_grad()
def generate_trajectory(diffusion, model, *, class_label, traj_seed, cfg_scale,
                        latent_size, device):
    
    n_steps = diffusion.num_timesteps
    gen = torch.Generator(device=device).manual_seed(traj_seed)
    x = torch.randn(1, 4, latent_size, latent_size, device=device, generator=gen)
    y = torch.tensor([class_label], device=device)

    z = torch.cat([x, x], 0)
    y_full = torch.cat([y, torch.full_like(y, NULL_CLASS)], 0)
    model_kwargs = dict(y=y_full, cfg_scale=cfg_scale)

    latents = {n_steps - 1: x.cpu().clone()}
    for i in range(n_steps - 1, -1, -1):
        t = torch.full((2,), i, device=device, dtype=torch.long)
        out = diffusion.p_mean_variance(model.forward_with_cfg, z, t,
                                        clip_denoised=False, model_kwargs=model_kwargs)
        if i > 0:
            eps = torch.randn(1, *z.shape[1:], device=device, generator=gen).expand_as(z)
            z = out["mean"] + torch.exp(0.5 * out["log_variance"]) * eps
            latents[i - 1] = z[:1].cpu().clone()
        else:
            z = out["mean"]
    x0 = z[:1].clone()
    return latents, x0


@torch.no_grad()
def denoise_from(diffusion, model, x, start_idx, *, denoise_seed, shared_noise,
                 class_label, cfg_scale, device):
    n = x.shape[0]
    y = torch.full((n,), class_label, device=device, dtype=torch.long)
    z = torch.cat([x, x], 0)
    y_full = torch.cat([y, torch.full_like(y, NULL_CLASS)], 0)
    model_kwargs = dict(y=y_full, cfg_scale=cfg_scale)

    gen = torch.Generator(device=device).manual_seed(denoise_seed)
    for i in range(start_idx, -1, -1):
        t = torch.full((2 * n,), i, device=device, dtype=torch.long)
        out = diffusion.p_mean_variance(model.forward_with_cfg, z, t,
                                        clip_denoised=False, model_kwargs=model_kwargs)
        if i > 0:
            if shared_noise:
                eps = torch.randn(1, *z.shape[1:], device=device, generator=gen).expand_as(z)
            else:
                eps = torch.randn(n, *z.shape[1:], device=device, generator=gen)
                eps = torch.cat([eps, eps], 0)  # second (CFG dup) half is unused
            z = out["mean"] + torch.exp(0.5 * out["log_variance"]) * eps
        else:
            z = out["mean"]
    return z[:n]


def renoise(diffusion, x_t, t_idx, k, *, perturb_seed, noise_type, device):
    """Push x_t k steps backward with the forward kernel q(x_{t+k} | x_t)."""
    j = min(t_idx + k, diffusion.num_timesteps - 1)
    abar = diffusion.alphas_cumprod  # numpy, respaced
    ratio = float(abar[j] / abar[t_idx])
    gen = torch.Generator(device=device).manual_seed(perturb_seed)
    eps = sample_unit_noise(noise_type, x_t.shape, gen, device)
    x_j = (ratio ** 0.5) * x_t.to(device) + ((1.0 - ratio) ** 0.5) * eps
    return x_j, j


class MetricSuite:


    def __init__(self, vae, device):
        import lpips
        import timm
        from transformers import CLIPModel

        self.device = device
        self.vae = vae
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.dino = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                      pretrained=True, num_classes=0).to(device).eval()
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self._inet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._inet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def decode(self, latents):
        imgs = self.vae.decode(latents.to(self.device) / LATENT_SCALE).sample
        return ((imgs + 1) / 2).clamp(0, 1)

    @torch.no_grad()
    def clip_feats(self, imgs01):
        x = torch.nn.functional.interpolate(imgs01, size=224, mode="bicubic", align_corners=False)
        x = (x - self._clip_mean) / self._clip_std
        f = self.clip.get_image_features(pixel_values=x)
        return torch.nn.functional.normalize(f, dim=-1)

    @torch.no_grad()
    def dino_feats(self, imgs01):
        x = torch.nn.functional.interpolate(imgs01, size=518, mode="bicubic", align_corners=False)
        x = (x - self._inet_mean) / self._inet_std
        f = self.dino(x)
        return torch.nn.functional.normalize(f, dim=-1)

    @torch.no_grad()
    def _lpips_batched(self, a, b, chunk=32):
        """Mean LPIPS over aligned pairs, evaluated in chunks."""
        vals = []
        for s in range(0, a.shape[0], chunk):
            vals.append(self.lpips(a[s:s + chunk], b[s:s + chunk]).flatten())
        return float(torch.cat(vals).mean().item())

    @torch.no_grad()
    def pairwise_lpips(self, imgs01):
        n = imgs01.shape[0]
        x = imgs01 * 2 - 1
        iu = torch.triu_indices(n, n, offset=1)
        return self._lpips_batched(x[iu[0]], x[iu[1]])

    @torch.no_grad()
    def lpips_to_ref(self, imgs01, ref01):
        x = imgs01 * 2 - 1
        r = (ref01 * 2 - 1).expand_as(x).contiguous()
        return self._lpips_batched(x, r)

    @staticmethod
    def pairwise_dist_stats(feats):
        n = feats.shape[0]
        sims = feats @ feats.T
        iu = torch.triu_indices(n, n, offset=1)
        pair_cos_dist = (1 - sims[iu[0], iu[1]]).mean().item()
        centroid = feats.mean(0, keepdim=True)
        var = ((feats - centroid) ** 2).sum(-1).mean().item()
        return float(pair_cos_dist), float(var)

    @staticmethod
    def pairwise_mse(x):
        n = x.shape[0]
        flat = x.reshape(n, -1).float()
        sq = torch.cdist(flat, flat) ** 2 / flat.shape[1]
        iu = torch.triu_indices(n, n, offset=1)
        return float(sq[iu[0], iu[1]].mean().item())

    @torch.no_grad()
    def ensemble_row(self, final_latents, ref_latent):
        imgs = self.decode(final_latents)
        ref_img = self.decode(ref_latent)
        clip_f, dino_f = self.clip_feats(imgs), self.dino_feats(imgs)
        clip_ref, dino_ref = self.clip_feats(ref_img), self.dino_feats(ref_img)
        clip_pd, clip_var = self.pairwise_dist_stats(clip_f)
        dino_pd, dino_var = self.pairwise_dist_stats(dino_f)
        row = {
            "clip_pair_cosdist": clip_pd,
            "clip_var": clip_var,
            "dino_pair_cosdist": dino_pd,
            "dino_var": dino_var,
            "lpips_pair": self.pairwise_lpips(imgs),
            "latent_mse_pair": self.pairwise_mse(final_latents),
            "pixel_mse_pair": self.pairwise_mse(imgs),
            "clip_dist_to_ref": float((1 - clip_f @ clip_ref.T).mean().item()),
            "dino_dist_to_ref": float((1 - dino_f @ dino_ref.T).mean().item()),
            "lpips_to_ref": self.lpips_to_ref(imgs, ref_img),
        }
        return row, imgs, ref_img



def load_dit(device, image_size=256, ckpt=None):
    latent_size = image_size // 8
    model = DiT_models["DiT-XL/2"](input_size=latent_size, num_classes=1000).to(device)
    name = f"DiT-XL-2-{image_size}x{image_size}.pt"
    
    # Check for checkpoint in multiple locations
    if ckpt and os.path.exists(ckpt):
        ckpt_path = ckpt
    else:
        local = DIT_DIR / name
        if local.is_file():
            ckpt_path = str(local)
        else:
            pretrained = Path("./pretrained_models") / name
            if pretrained.is_file():
                ckpt_path = str(pretrained)
            else:
                ckpt_path = name
    
    model.load_state_dict(find_model(ckpt_path))
    model.eval()
    return model, latent_size

def load_vae(device):
    from diffusers.models import AutoencoderKL
    return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()


def steps_for_fracs(n_steps, fracs):
    """Fraction of denoising completed -> respaced step index (input latent)."""
    return {f: max(0, min(n_steps - 1, round((n_steps - 1) * (1 - f)))) for f in fracs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--classes", type=int, nargs="+", default=[207, 88])
    ap.add_argument("--num-steps", type=int, default=250, help="respaced sampling steps")
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--ensemble", type=int, default=16, help="ensemble size M")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5],
                    help="sampler steps backward along the trajectory")
    ap.add_argument("--t-fracs", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9],
                    help="fractions of denoising completed at the saved latent")
    ap.add_argument("--noise-types", type=str, nargs="+", default=["gaussian"],
                    choices=["gaussian", "uniform", "laplace"])
    ap.add_argument("--traj-seed", type=int, default=0)
    ap.add_argument("--denoise-seed", type=int, default=1000)
    ap.add_argument("--perturb-seed", type=int, default=2000,
                    help="base; member m uses perturb_seed + m")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="path to a DiT checkpoint (default: repo copy or auto-download)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=str,
                    default=str(Path(__file__).resolve().parent / "outputs" / "exp1"))
    args = ap.parse_args()

    device = args.device
    torch.set_grad_enabled(False)
    outdir = Path(args.outdir)
    (outdir / "grids").mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2))

    print(f"[exp1] device={device}, loading DiT-XL/2 + VAE + metric models ...", flush=True)
    model, latent_size = load_dit(device, ckpt=args.ckpt)
    vae = load_vae(device)
    diffusion = create_diffusion(str(args.num_steps))
    metrics = MetricSuite(vae, device)
    n_steps = diffusion.num_timesteps
    t_map = steps_for_fracs(n_steps, args.t_fracs)
    print(f"[exp1] {n_steps} respaced steps; saved-latent steps: {t_map}")

    rows = []
    for cls in args.classes:
        print(f"[exp1] class {cls}: generating reference trajectory ...", flush=True)
        latents, x0_ref = generate_trajectory(
            diffusion, model, class_label=cls, traj_seed=args.traj_seed,
            cfg_scale=args.cfg_scale, latent_size=latent_size, device=device)
        ref_img = metrics.decode(x0_ref)
        save_image(ref_img, outdir / "trajectories" / f"ref_class{cls}.png")
        torch.save({k: v for k, v in latents.items()},
                   outdir / "trajectories" / f"latents_class{cls}.pt")

        for frac, t_idx in t_map.items():
            x_t = latents[t_idx].to(device)

            x_batch = x_t.expand(args.ensemble, -1, -1, -1).contiguous()
            finals = denoise_from(diffusion, model, x_batch, t_idx,
                                  denoise_seed=args.denoise_seed, shared_noise=False,
                                  class_label=cls, cfg_scale=args.cfg_scale, device=device)
            row, imgs, _ = metrics.ensemble_row(finals, x0_ref)
            row.update(dict(class_id=cls, mode="denoise", k=0, noise_type="gaussian",
                            t_frac=frac, t_idx=t_idx, start_idx=t_idx))
            rows.append(row)
            save_image(torch.cat([ref_img, imgs]),
                       outdir / "grids" / f"cls{cls}_t{frac:.1f}_denoise_k0.png",
                       nrow=args.ensemble + 1)
            print(f"  t={frac:.1f} (idx {t_idx})  denoise k=0   "
                  f"lpips={row['lpips_pair']:.4f} clip={row['clip_pair_cosdist']:.4f}",
                  flush=True)

            for noise_type in args.noise_types:
                for k in args.ks:
                    perturbed, start = [], None
                    for m in range(args.ensemble):
                        x_j, j = renoise(diffusion, x_t, t_idx, k,
                                         perturb_seed=args.perturb_seed + m,
                                         noise_type=noise_type, device=device)
                        perturbed.append(x_j)
                        start = j
                    x_batch = torch.cat(perturbed, 0)
                    finals = denoise_from(diffusion, model, x_batch, start,
                                          denoise_seed=args.denoise_seed, shared_noise=True,
                                          class_label=cls, cfg_scale=args.cfg_scale,
                                          device=device)
                    row, imgs, _ = metrics.ensemble_row(finals, x0_ref)
                    row.update(dict(class_id=cls, mode="perturb", k=k, noise_type=noise_type,
                                    t_frac=frac, t_idx=t_idx, start_idx=start))
                    rows.append(row)
                    save_image(torch.cat([ref_img, imgs]),
                               outdir / "grids" /
                               f"cls{cls}_t{frac:.1f}_perturb_k{k}_{noise_type}.png",
                               nrow=args.ensemble + 1)
                    print(f"  t={frac:.1f} (idx {t_idx})  perturb k={k} ({noise_type})  "
                          f"lpips={row['lpips_pair']:.4f} clip={row['clip_pair_cosdist']:.4f}",
                          flush=True)

            pd.DataFrame(rows).to_csv(outdir / "exp1_results.csv", index=False)

    pd.DataFrame(rows).to_csv(outdir / "exp1_results.csv", index=False)
    print(f"[exp1] done -> {outdir / 'exp1_results.csv'}")


if __name__ == "__main__":
    main()
