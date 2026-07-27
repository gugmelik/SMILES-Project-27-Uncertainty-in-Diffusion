"""ImageNet data for fitting a posterior over the denoiser's weights.

Both the Laplace fit and the variational LoRA fit need ``(latent, label)``
pairs in exactly the format DiT was trained on: ADM centre-crop, ``[-1, 1]``
scaling, VAE-encoded and multiplied by ``0.18215``.

``torchvision.datasets.ImageFolder`` sorts WordNet ids alphabetically, which is
the standard ImageNet-1k class ordering DiT's label embedding expects, so no
remapping is needed as long as the directory holds all 1000 synset folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image

from .dit import LATENT_SCALE, REPO_ROOT

DEFAULT_IMAGENET_VAL = REPO_ROOT / "imagenet_val" / "val"


def center_crop_arr(pil_image: Image.Image, image_size: int) -> np.ndarray:
    """ADM centre crop, copied from ``DiT/train.py`` so training and fitting match."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def imagenet_image_loader(
    root: str | Path = DEFAULT_IMAGENET_VAL,
    *,
    image_size: int = 256,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 0,
):
    """DataLoader over ImageNet folders yielding ``(images in [-1,1], labels)``."""
    from torchvision import transforms  # noqa: PLC0415
    from torchvision.datasets import ImageFolder  # noqa: PLC0415

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"No ImageNet directory at {root}. Run scripts/download_imagenet_val_kaggle.sh "
            "or pass --data-path."
        )

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True),
        ]
    )
    dataset = ImageFolder(str(root), transform=transform)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def imagenet_latent_loader(
    bundle,
    root: str | Path = DEFAULT_IMAGENET_VAL,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    repeat: bool = False,
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield VAE-encoded, scaled latents and labels.

    Set ``repeat=True`` for the variational fit, which consumes a fixed number
    of optimiser steps rather than a fixed number of epochs.
    """
    loader = imagenet_image_loader(
        root,
        image_size=bundle.image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=seed,
    )

    while True:
        for images, labels in loader:
            images = images.to(bundle.device, non_blocking=True)
            with torch.no_grad():
                latents = bundle.vae.encode(images).latent_dist.sample() * LATENT_SCALE
            yield latents, labels.to(bundle.device)
        if not repeat:
            return


def cached_latent_loader(
    path: str | Path,
    *,
    batch_size: int = 8,
    repeat: bool = False,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Iterate over a ``.pt`` cache of ``{"latents": ..., "labels": ...}``.

    Encoding the same images repeatedly is wasteful when fitting several
    posteriors; :func:`cache_latents` writes the cache once.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    latents, labels = blob["latents"], blob["labels"]
    generator = torch.Generator().manual_seed(seed)

    while True:
        order = (
            torch.randperm(latents.shape[0], generator=generator)
            if shuffle
            else torch.arange(latents.shape[0])
        )
        for start in range(0, len(order) - batch_size + 1, batch_size):
            index = order[start : start + batch_size]
            yield latents[index], labels[index]
        if not repeat:
            return


def cache_latents(
    bundle,
    output: str | Path,
    *,
    root: str | Path = DEFAULT_IMAGENET_VAL,
    num_images: int = 2048,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 0,
) -> Path:
    """Encode ``num_images`` ImageNet images once and save them for reuse."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    latents, labels = [], []
    collected = 0
    for latent, label in imagenet_latent_loader(
        bundle,
        root,
        batch_size=batch_size,
        num_workers=num_workers,
        repeat=False,
        seed=seed,
    ):
        take = min(batch_size, num_images - collected)
        latents.append(latent[:take].cpu())
        labels.append(label[:take].cpu())
        collected += take
        if collected >= num_images:
            break

    if collected < num_images:
        raise ValueError(f"Only found {collected} images, wanted {num_images}")

    torch.save(
        {
            "latents": torch.cat(latents),
            "labels": torch.cat(labels),
            "image_size": bundle.image_size,
            "latent_scale": LATENT_SCALE,
        },
        output,
    )
    return output
