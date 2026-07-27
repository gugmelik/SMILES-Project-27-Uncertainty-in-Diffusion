"""Feature encoders for measuring disagreement in a semantic space.

Pixel variance is a poor uncertainty score for generative models: a one-pixel
translation, a texture reshuffle or a background change all produce large pixel
variance while the semantic content is unchanged. Projecting each sample into a
perceptual feature space first and measuring variance there is much better
behaved.

All encoders take images in ``[-1, 1]`` with shape (N, 3, H, W) and return
``(N, d)`` float32 features on CPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _to_unit_range(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(-1, 1) + 1) / 2


def _normalize(
    images: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]
) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_t) / std_t


class FeatureEncoder(ABC):
    """Maps generated images to a vector space where variance is meaningful."""

    name: str = "encoder"

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) in [-1, 1] -> (N, d) float32 on CPU."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


class PixelEncoder(FeatureEncoder):
    """Flattened pixels, optionally downsampled.

    Included mainly as the baseline the semantic encoders should beat — if your
    uncertainty ranking is the same in pixel space and in DINO space, the score
    is probably driven by low-level variation.
    """

    name = "pixel"

    def __init__(self, size: int | None = 64):
        self.size = size

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.size is not None and images.shape[-1] != self.size:
            images = F.interpolate(
                images, size=(self.size, self.size), mode="bilinear", align_corners=False
            )
        return images.flatten(1).float().cpu()


class VAEEncoder(FeatureEncoder):
    """The diffusion model's own VAE latent space.

    Always available (the VAE is already loaded) and far more semantic than raw
    pixels, so this is a sensible default when CLIP/DINO weights cannot be
    downloaded.
    """

    name = "vae"

    def __init__(self, vae, scale: float = 0.18215):
        self.vae = vae
        self.scale = scale

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        device = next(self.vae.parameters()).device
        latents = self.vae.encode(images.to(device)).latent_dist.mean * self.scale
        return latents.flatten(1).float().cpu()


class TimmEncoder(FeatureEncoder):
    """Any ``timm`` backbone used as a frozen feature extractor (e.g. DINOv2).

    ``timm`` is already a DiT dependency, so this needs no new packages.
    """

    name = "timm"

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        *,
        device: str | torch.device = "cuda",
        normalize_features: bool = True,
    ):
        import timm  # noqa: PLC0415

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        config = timm.data.resolve_data_config({}, model=self.model)
        self.input_size = config["input_size"][-1]
        self.mean = config.get("mean", IMAGENET_MEAN)
        self.std = config.get("std", IMAGENET_STD)
        self.device = torch.device(device)
        self.normalize_features = normalize_features
        self.name = f"timm:{model_name}"

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = _to_unit_range(images.to(self.device).float())
        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
        )
        features = self.model(_normalize(images, self.mean, self.std))
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return features.float().cpu()


class CLIPEncoder(FeatureEncoder):
    """CLIP image embeddings via ``open_clip`` or ``transformers``.

    Optional: raises a clear error if neither package is installed rather than
    failing deep inside a sampling loop.
    """

    name = "clip"

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        device: str | torch.device = "cuda",
        normalize_features: bool = True,
    ):
        self.device = torch.device(device)
        self.normalize_features = normalize_features
        self.input_size = 224
        self.mean, self.std = CLIP_MEAN, CLIP_STD
        self.name = f"clip:{model_name}/{pretrained}"

        try:
            import open_clip  # noqa: PLC0415

            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._encode_fn = model.encode_image
            self._backbone = model
        except ImportError:
            try:
                from transformers import CLIPVisionModelWithProjection  # noqa: PLC0415
            except ImportError as error:
                raise ImportError(
                    "CLIPEncoder needs either `open_clip_torch` or `transformers`. "
                    "Install one, or use TimmEncoder / VAEEncoder instead."
                ) from error
            model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._encode_fn = lambda x: model(pixel_values=x).image_embeds
            self._backbone = model

        self._backbone.eval().to(self.device)
        for parameter in self._backbone.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images = _to_unit_range(images.to(self.device).float())
        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
        )
        features = self._encode_fn(_normalize(images, self.mean, self.std))
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return features.float().cpu()


def build_encoder(
    name: str, *, vae=None, device: str | torch.device = "cuda"
) -> FeatureEncoder:
    """Construct an encoder from a short name used by the CLI scripts."""
    if name == "pixel":
        return PixelEncoder()
    if name == "vae":
        if vae is None:
            raise ValueError("The 'vae' encoder needs the loaded AutoencoderKL")
        return VAEEncoder(vae)
    if name in {"dino", "dinov2"}:
        return TimmEncoder("vit_base_patch14_dinov2.lvd142m", device=device)
    if name == "clip":
        return CLIPEncoder(device=device)
    if name.startswith("timm:"):
        return TimmEncoder(name[len("timm:") :], device=device)
    raise ValueError(f"Unknown encoder {name!r}")
