#!/usr/bin/env python
"""Measure Bayesian uncertainty for DiT-XL/2 generations.

Runs any combination of the three measurements against a fitted posterior:

``--profile``
    Per-timestep local epistemic uncertainty along a reference trajectory.
``--predictive``
    Posterior-predictive generations with the diffusion noise pinned, scored in
    pixel and feature space, plus a spatial heatmap.
``--decompose``
    Nested weight x noise sampling split into epistemic and aleatoric parts.

Example::

    python experiments/bayesian_uncertainty/measure_uncertainty.py \\
        --posterior results/posteriors/laplace_last_layer.pt \\
        --classes 207 88 980 --predictive --profile \\
        --num-weight-samples 16 --encoder vae \\
        --out results/bayesian_uncertainty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.uncertainty import (  # noqa: E402
    FixedNoiseSampler,
    GuidanceConfig,
    build_encoder,
    decompose_uncertainty,
    load_dit,
    local_epistemic_profile,
    posterior_predictive,
    trajectory_uncertainty,
)
from src.uncertainty.lora import load_lora_posterior  # noqa: E402
from src.uncertainty.posteriors import MAPPosterior, load_posterior  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--posterior", type=Path, default=None, help="Required unless --posterior-kind map")
    parser.add_argument(
        "--posterior-kind",
        choices=["auto", "laplace", "lora", "map"],
        default="auto",
        help="'map' is a point-mass control: every epistemic number must come out zero",
    )
    parser.add_argument("--posterior-scale", type=float, default=None, help="Override the Gaussian posterior temperature")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "bayesian_uncertainty")

    parser.add_argument("--classes", type=int, nargs="+", default=[207])
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--method", choices=["ddpm", "ddim"], default="ddpm")

    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--predictive", action="store_true")
    parser.add_argument("--decompose", action="store_true")

    parser.add_argument("--num-weight-samples", type=int, default=16)
    parser.add_argument("--num-noise-samples", type=int, default=4, help="K for --decompose")
    parser.add_argument("--profile-samples", type=int, default=8)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--encoder", type=str, default="vae", help="vae | pixel | dino | clip | timm:<name>")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_any_posterior(args, bundle):
    if args.posterior_kind == "map":
        from src.uncertainty.dit import select_parameters  # noqa: PLC0415

        return MAPPosterior(bundle.model, select_parameters(bundle.model, "last_layer"))

    state = torch.load(args.posterior, map_location="cpu", weights_only=False)
    kind = state.get("kind", "")
    if kind == "BayesianLoRAPosterior" or args.posterior_kind == "lora":
        return load_lora_posterior(bundle.model, args.posterior, device=bundle.device)

    posterior = load_posterior(args.posterior, device=bundle.device)
    if hasattr(posterior, "to"):
        posterior = posterior.to(bundle.device)
    if args.posterior_scale is not None and hasattr(posterior, "scale"):
        posterior.scale = args.posterior_scale
    return posterior


def save_image_grid(images: torch.Tensor, path: Path) -> None:
    """``images`` in [-1, 1], shape (M, 3, H, W)."""
    from torchvision.utils import save_image  # noqa: PLC0415

    save_image(
        (images.clamp(-1, 1) + 1) / 2,
        path,
        nrow=min(8, images.shape[0]),
    )


def save_heatmap(heatmap: torch.Tensor, path: Path, title: str) -> None:
    """``heatmap`` shape (1, H, W)."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return

    figure, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(heatmap.squeeze(0).cpu().numpy(), cmap="inferno")
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_profile_plot(profile, path: Path) -> None:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(profile.timesteps, profile.scalar.mean(dim=1).numpy(), marker="o", ms=3)
    axis.invert_xaxis()  # denoising runs from t=999 down to t=0
    axis.set_xlabel("diffusion timestep t (1000-step schedule)")
    axis.set_ylabel(r"epistemic Var$[\hat{x}_0]$")
    axis.set_title(f"Per-timestep epistemic uncertainty (M={profile.num_samples})")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    if not (args.profile or args.predictive or args.decompose):
        raise SystemExit("Pick at least one of --profile / --predictive / --decompose")
    if args.posterior is None and args.posterior_kind != "map":
        raise SystemExit("--posterior is required unless --posterior-kind map")

    torch.set_grad_enabled(False)
    bundle = load_dit(
        image_size=args.image_size,
        num_sampling_steps=args.num_sampling_steps,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    posterior = load_any_posterior(args, bundle)
    sampler = FixedNoiseSampler(
        bundle, GuidanceConfig(cfg_scale=args.cfg_scale), method=args.method
    )
    encoder = build_encoder(args.encoder, vae=bundle.vae, device=bundle.device)

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "posterior": str(args.posterior) if args.posterior else args.posterior_kind,
        "num_bayesian_parameters": posterior.num_parameters,
        "classes": args.classes,
        "cfg_scale": args.cfg_scale,
        "num_sampling_steps": bundle.num_sampling_steps,
        "method": args.method,
        "encoder": getattr(encoder, "name", args.encoder),
        "seed": args.seed,
        "results": {},
    }

    for class_label in args.classes:
        tag = f"class_{class_label:04d}"
        class_dir = args.out / tag
        class_dir.mkdir(parents=True, exist_ok=True)
        entry: dict = {}
        print(f"\n=== {tag} ===")

        z = sampler.initial_latent(1, seed=args.seed)
        step_noise = (
            sampler.make_step_noise(1, seed=args.seed)
            if args.method == "ddpm"
            else None
        )

        if args.predictive:
            predictive = posterior_predictive(
                sampler,
                posterior,
                z=z,
                class_labels=[class_label],
                step_noise=step_noise,
                num_weight_samples=args.num_weight_samples,
                seed=args.seed,
            )
            score = trajectory_uncertainty(predictive, encoder)
            entry["predictive"] = {
                "pixel_uncertainty": float(score.pixel_uncertainty[0]),
                "semantic_uncertainty": float(score.semantic_uncertainty[0]),
                "semantic_entropy": float(score.semantic_entropy[0]),
                "feature_dim": score.feature_dim,
                "num_weight_samples": score.num_weight_samples,
            }
            print(
                f"  pixel={entry['predictive']['pixel_uncertainty']:.6f}  "
                f"semantic={entry['predictive']['semantic_uncertainty']:.6e}  "
                f"entropy={entry['predictive']['semantic_entropy']:.3f}"
            )
            save_heatmap(
                score.pixel_map[0], class_dir / "pixel_variance.png", f"{tag} pixel variance"
            )
            torch.save(
                {"pixel_map": score.pixel_map, "latents": predictive.latents},
                class_dir / "predictive.pt",
            )
            if args.save_images and predictive.images is not None:
                save_image_grid(predictive.images[:, 0], class_dir / "posterior_draws.png")

        if args.profile:
            profile = local_epistemic_profile(
                sampler,
                posterior,
                z=z,
                class_labels=[class_label],
                step_noise=step_noise,
                num_samples=args.profile_samples,
                record_every=args.record_every,
                seed=args.seed,
            )
            entry["profile"] = {
                "timesteps": list(profile.timesteps),
                "uncertainty": profile.scalar.mean(dim=1).tolist(),
                "num_samples": profile.num_samples,
            }
            peak = int(torch.argmax(profile.scalar.mean(dim=1)))
            print(f"  profile peak at t={profile.timesteps[peak]}")
            save_profile_plot(profile, class_dir / "timestep_profile.png")
            torch.save(profile.scalar, class_dir / "timestep_profile.pt")

        if args.decompose:
            decomposition = decompose_uncertainty(
                sampler,
                posterior,
                encoder,
                class_labels=[class_label],
                num_weight_samples=args.num_weight_samples,
                num_noise_samples=args.num_noise_samples,
                weight_seed=args.seed,
                noise_seed=args.seed + 1000,
            )
            entry["decomposition"] = {
                "epistemic": float(decomposition.epistemic[0]),
                "aleatoric": float(decomposition.aleatoric[0]),
                "total": float(decomposition.total[0]),
                "epistemic_ratio": float(decomposition.ratio()[0]),
                "M": decomposition.num_weight_samples,
                "K": decomposition.num_noise_samples,
            }
            print(
                f"  epistemic={entry['decomposition']['epistemic']:.6e}  "
                f"aleatoric={entry['decomposition']['aleatoric']:.6e}  "
                f"ratio={entry['decomposition']['epistemic_ratio']:.3f}"
            )

        summary["results"][tag] = entry

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
