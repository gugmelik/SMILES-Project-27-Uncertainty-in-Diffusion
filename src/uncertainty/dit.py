"""Loading and wiring helpers for the vendored DiT codebase.

The DiT submodule is not an installable package: its modules import each other by
top-level name (``from diffusion import create_diffusion``) and ``find_model``
downloads checkpoints into a *relative* ``pretrained_models/`` directory. This
module hides both quirks so the rest of ``src/uncertainty`` can stay clean.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
DIT_ROOT = REPO_ROOT / "DiT"

DEFAULT_VAE = "stabilityai/sd-vae-ft-ema"
LATENT_SCALE = 0.18215


def unconditional_class(model: nn.Module) -> int:
    """The label index DiT uses for the unconditional (dropped-label) embedding.

    ``LabelEmbedder`` appends one extra row to its embedding table and drops
    labels to ``num_classes``, so this is 1000 for ImageNet DiT-XL/2 — but
    reading it off the model keeps smaller test configurations working instead of
    indexing past the end of the table.
    """
    embedder = getattr(model, "y_embedder", None)
    if embedder is None or not hasattr(embedder, "num_classes"):
        raise AttributeError(
            "Model has no y_embedder.num_classes; pass an explicit uncond_class."
        )
    return int(embedder.num_classes)


def bootstrap_dit() -> Path:
    """Put the vendored DiT checkout on ``sys.path``. Idempotent."""
    if not DIT_ROOT.is_dir():
        raise FileNotFoundError(
            f"DiT checkout not found at {DIT_ROOT}. Clone facebookresearch/DiT there "
            "(see README) before using src.uncertainty."
        )
    path = str(DIT_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return DIT_ROOT


@contextlib.contextmanager
def _chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass
class DiTBundle:
    """Everything needed to run and measure a DiT sampler."""

    model: nn.Module
    vae: Any
    diffusion: Any
    device: torch.device
    latent_size: int
    image_size: int
    num_sampling_steps: int

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.model.in_channels, self.latent_size, self.latent_size)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents -> images in [-1, 1], shape (N, 3, H, W)."""
        with torch.no_grad():
            return self.vae.decode(latents / LATENT_SCALE).sample

    def encode(self, images: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        """Images in [-1, 1] -> scaled latents, shape (N, 4, h, w)."""
        with torch.no_grad():
            dist = self.vae.encode(images).latent_dist
            latents = dist.sample() if sample else dist.mean
        return latents * LATENT_SCALE


def load_dit(
    *,
    image_size: int = 256,
    num_sampling_steps: int = 250,
    vae_model: str = DEFAULT_VAE,
    checkpoint: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> DiTBundle:
    """Load DiT-XL/2, the SD VAE and a respaced diffusion process.

    ``checkpoint`` may be a local ``.pt`` path; when omitted the official
    ``DiT-XL-2-{image_size}x{image_size}.pt`` checkpoint is used (downloaded into
    ``DiT/pretrained_models/`` on first use, matching the notebooks).
    """
    bootstrap_dit()
    from diffusers.models import AutoencoderKL  # noqa: PLC0415
    from diffusion import create_diffusion  # noqa: PLC0415
    from download import find_model  # noqa: PLC0415
    from models import DiT_XL_2  # noqa: PLC0415

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    latent_size = image_size // 8
    model = DiT_XL_2(input_size=latent_size).to(device=device, dtype=dtype)

    # Resolve a user-supplied path against the *caller's* cwd before chdir'ing.
    name = (
        os.path.abspath(checkpoint)
        if checkpoint is not None
        else f"DiT-XL-2-{image_size}x{image_size}.pt"
    )
    # find_model writes/reads a relative pretrained_models/ directory.
    with _chdir(DIT_ROOT):
        state_dict = find_model(name)
    model.load_state_dict(state_dict)
    model.eval()

    vae = AutoencoderKL.from_pretrained(vae_model).to(device=device, dtype=dtype).eval()
    diffusion = create_diffusion(str(num_sampling_steps))

    return DiTBundle(
        model=model,
        vae=vae,
        diffusion=diffusion,
        device=device,
        latent_size=latent_size,
        image_size=image_size,
        num_sampling_steps=diffusion.num_timesteps,
    )


def freeze(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def select_parameters(model: nn.Module, spec: str) -> tuple[str, ...]:
    """Resolve a parameter-subset spec into fully qualified parameter names.

    Supported specs (mirroring the cheap-to-expensive ladder for Bayesianising a
    large denoiser):

    ``"last_layer"``
        Only ``final_layer.linear`` — the BayesDiff-style last-layer Laplace.
    ``"last_layer+adaln"``
        Adds the final adaLN modulation projection.
    ``"late_blocks:N"``
        ``attn.proj`` and ``mlp.fc2`` of the last ``N`` DiT blocks, plus
        ``last_layer+adaln``.
    ``"regex:<pattern>"``
        Every parameter whose name matches ``<pattern>``.
    """
    names = [name for name, _ in model.named_parameters()]

    if spec.startswith("regex:"):
        import re  # noqa: PLC0415

        pattern = re.compile(spec[len("regex:") :])
        selected = [name for name in names if pattern.search(name)]
    elif spec == "last_layer":
        selected = [n for n in names if n.startswith("final_layer.linear.")]
    elif spec == "last_layer+adaln":
        selected = [n for n in names if n.startswith("final_layer.")]
    elif spec.startswith("late_blocks:"):
        num_blocks = int(spec.split(":", 1)[1])
        depth = len(model.blocks)
        if not 0 < num_blocks <= depth:
            raise ValueError(f"late_blocks:{num_blocks} out of range for depth {depth}")
        late = {depth - 1 - i for i in range(num_blocks)}
        suffixes = ("attn.proj.", "mlp.fc2.")
        selected = [
            n
            for n in names
            if n.startswith("final_layer.")
            or any(n.startswith(f"blocks.{i}.{s}") for i in late for s in suffixes)
        ]
    else:
        raise ValueError(f"Unknown parameter spec {spec!r}")

    if not selected:
        raise ValueError(f"Spec {spec!r} matched no parameters")
    return tuple(selected)


def count_parameters(model: nn.Module, names: tuple[str, ...]) -> int:
    lookup = dict(model.named_parameters())
    return sum(lookup[name].numel() for name in names)
