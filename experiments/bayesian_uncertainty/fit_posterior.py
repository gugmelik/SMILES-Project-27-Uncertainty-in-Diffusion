#!/usr/bin/env python
"""Fit an approximate posterior over a subset of DiT-XL/2's weights.

Two methods, both leaving the pretrained backbone frozen:

``laplace``
    Post-hoc diagonal Laplace. Training-free, minutes on one GPU for the last
    layer. Start here.
``lora``
    Variational Bayesian LoRA on late blocks (Bayes-by-Backprop). Slower, but
    it can express uncertainty that the final projection cannot.

Examples::

    python experiments/bayesian_uncertainty/fit_posterior.py laplace \\
        --params last_layer --num-samples 512 \\
        --out results/posteriors/laplace_last_layer.pt

    python experiments/bayesian_uncertainty/fit_posterior.py lora \\
        --targets late_blocks:4 --rank 4 --steps 2000 \\
        --out results/posteriors/lora_late4.pt
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

from src.uncertainty import cache_latents, load_dit  # noqa: E402
from src.uncertainty.data import (  # noqa: E402
    DEFAULT_IMAGENET_VAL,
    cached_latent_loader,
    imagenet_latent_loader,
)
from src.uncertainty.dit import bootstrap_dit, count_parameters, select_parameters  # noqa: E402
from src.uncertainty.laplace import LaplaceConfig, fit_diagonal_laplace  # noqa: E402
from src.uncertainty.lora import LoRAFitConfig, fit_variational_lora  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=["laplace", "lora"])
    parser.add_argument("--out", type=Path, required=True, help="Where to save the posterior (.pt)")

    data = parser.add_argument_group("data")
    data.add_argument("--data-path", type=Path, default=DEFAULT_IMAGENET_VAL)
    data.add_argument("--latent-cache", type=Path, default=None, help="Reuse/write a .pt latent cache")
    data.add_argument("--cache-size", type=int, default=2048, help="Images to encode when building the cache")
    data.add_argument("--num-workers", type=int, default=4)

    model = parser.add_argument_group("model")
    model.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    model.add_argument("--checkpoint", type=str, default=None)
    model.add_argument("--device", type=str, default=None)

    laplace = parser.add_argument_group("laplace")
    laplace.add_argument("--params", type=str, default="last_layer", help="last_layer | last_layer+adaln | late_blocks:N | regex:...")
    laplace.add_argument("--num-samples", type=int, default=512)
    laplace.add_argument("--timesteps-per-image", type=int, default=1)
    laplace.add_argument("--prior-precision", type=float, default=1.0)
    laplace.add_argument("--fisher-scale", type=float, default=1.0)
    laplace.add_argument("--posterior-scale", type=float, default=1.0)
    laplace.add_argument("--no-optimize-prior", action="store_true")

    lora = parser.add_argument_group("lora")
    lora.add_argument("--targets", type=str, default="late_blocks:4")
    lora.add_argument("--rank", type=int, default=4)
    lora.add_argument("--steps", type=int, default=2000)
    lora.add_argument("--batch-size", type=int, default=8)
    lora.add_argument("--learning-rate", type=float, default=1e-3)
    lora.add_argument("--prior-std", type=float, default=1e-2)
    lora.add_argument("--kl-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    return parser


def make_data(args, bundle, *, repeat: bool, batch_size: int):
    """Latent stream, preferring a cache so repeated fits skip the VAE."""
    if args.latent_cache is not None:
        if not args.latent_cache.exists():
            print(f"Building latent cache at {args.latent_cache} ...")
            cache_latents(
                bundle,
                args.latent_cache,
                root=args.data_path,
                num_images=args.cache_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
        return cached_latent_loader(
            args.latent_cache, batch_size=batch_size, repeat=repeat, seed=args.seed
        )
    return imagenet_latent_loader(
        bundle,
        args.data_path,
        batch_size=batch_size,
        repeat=repeat,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def main() -> None:
    args = build_parser().parse_args()
    bootstrap_dit()
    from diffusion import create_diffusion  # noqa: PLC0415

    torch.manual_seed(args.seed)
    bundle = load_dit(
        image_size=args.image_size,
        num_sampling_steps=250,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    # Curvature/ELBO must reflect the full noise schedule, not a respaced sampler.
    train_diffusion = create_diffusion("")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "laplace":
        names = select_parameters(bundle.model, args.params)
        print(
            f"Bayesianising {len(names)} tensors "
            f"({count_parameters(bundle.model, names):,} parameters) via {args.params}"
        )
        config = LaplaceConfig(
            parameter_spec=args.params,
            num_samples=args.num_samples,
            timesteps_per_image=args.timesteps_per_image,
            prior_precision=args.prior_precision,
            fisher_scale=args.fisher_scale,
            posterior_scale=args.posterior_scale,
            optimize_prior=not args.no_optimize_prior,
            seed=args.seed,
        )
        data = make_data(args, bundle, repeat=False, batch_size=8)
        torch.set_grad_enabled(True)
        result = fit_diagonal_laplace(bundle.model, train_diffusion, data, config)
        torch.set_grad_enabled(False)
        result.posterior.save(args.out)
        metadata = {
            "method": "laplace",
            "config": config.as_dict(),
            "map_loss_sum": result.map_loss_sum,
            "num_samples": result.num_samples,
        }
    else:
        config = LoRAFitConfig(
            target_spec=args.targets,
            rank=args.rank,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            prior_std=args.prior_std,
            kl_weight=args.kl_weight,
            seed=args.seed,
        )
        data = make_data(args, bundle, repeat=True, batch_size=args.batch_size)
        posterior = fit_variational_lora(bundle.model, train_diffusion, data, config)
        posterior.save(args.out)
        metadata = {
            "method": "lora",
            "config": config.__dict__,
            "history": getattr(posterior, "history", []),
        }

    metadata_path = args.out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Saved posterior -> {args.out}")
    print(f"Saved metadata  -> {metadata_path}")


if __name__ == "__main__":
    main()
