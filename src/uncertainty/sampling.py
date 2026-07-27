"""Reverse-diffusion sampling with every source of randomness pinned.

To attribute variability in a generated image to the *weights* rather than to
the sampler, everything else has to be held fixed across draws:

* the initial latent ``z_T``,
* the conditioning,
* the timestep schedule,
* and the noise injected at every reverse step.

``FixedNoiseSampler`` precomputes the per-step noise from a seed so the exact
same tensor sequence can be replayed under a different ``theta_m``. With
``method="ddim"`` the reverse map is deterministic and no step noise is used at
all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .dit import DiTBundle, unconditional_class
from .utils import randn

StepCallback = Callable[[int, torch.Tensor, torch.Tensor, dict], None]
"""``(step_index, t, x_t, p_mean_variance_out) -> None``."""


@dataclass(frozen=True)
class GuidanceConfig:
    """Classifier-free guidance settings."""

    cfg_scale: float | None = 4.0
    uncond_class: int | None = None
    """Unconditional label index; derived from the model when ``None``."""

    @property
    def enabled(self) -> bool:
        return self.cfg_scale is not None

    def uncond_index(self, model: nn.Module) -> int:
        if self.uncond_class is not None:
            return self.uncond_class
        return unconditional_class(model)


def build_cfg_inputs(
    z: torch.Tensor,
    class_labels: Sequence[int] | torch.Tensor,
    guidance: GuidanceConfig,
    model: nn.Module,
) -> tuple[torch.Tensor, Callable, dict]:
    """Expand a batch for classifier-free guidance.

    Returns ``(z_in, model_fn, model_kwargs)``. With guidance enabled the batch
    is duplicated (conditional half first, unconditional half second), which is
    the convention ``DiT.forward_with_cfg`` expects; the conditional half is the
    one to read results from.
    """
    device = z.device
    labels = torch.as_tensor(class_labels, device=device, dtype=torch.long)
    if labels.ndim == 0:
        labels = labels[None]
    if labels.shape[0] != z.shape[0]:
        raise ValueError(
            f"Got {labels.shape[0]} labels for a batch of {z.shape[0]} latents"
        )

    if not guidance.enabled:
        return z, model.forward, {"y": labels}

    z_in = torch.cat([z, z], dim=0)
    uncond = torch.full_like(labels, guidance.uncond_index(model))
    y = torch.cat([labels, uncond], dim=0)
    return z_in, model.forward_with_cfg, {"y": y, "cfg_scale": guidance.cfg_scale}


def conditional_half(tensor: torch.Tensor, guidance: GuidanceConfig) -> torch.Tensor:
    """Drop the unconditional half of a CFG batch."""
    if not guidance.enabled:
        return tensor
    return tensor[: tensor.shape[0] // 2]


@dataclass
class SamplingOutput:
    latents: torch.Tensor
    """Final ``x_0`` latents for the conditional half, shape (N, C, H, W)."""
    pred_xstart_trace: torch.Tensor | None = None
    """Optional (S, N, C, H, W) stack of per-step ``x_0`` predictions."""
    latent_trace: torch.Tensor | None = None
    """Optional (S, N_full, C, H, W) stack of the ``x_t`` fed to the model.

    Kept at full CFG width so a recorded ``x_t`` can be replayed through the
    denoiser under a different posterior draw.
    """
    trace_steps: tuple[int, ...] = ()
    """Reverse-loop indices corresponding to the traces."""


class FixedNoiseSampler:
    """Replayable reverse-diffusion sampler over a :class:`DiTBundle`."""

    def __init__(
        self,
        bundle: DiTBundle,
        guidance: GuidanceConfig | None = None,
        *,
        clip_denoised: bool = False,
        method: str = "ddpm",
    ):
        if method not in {"ddpm", "ddim"}:
            raise ValueError(f"method must be 'ddpm' or 'ddim', got {method!r}")
        self.bundle = bundle
        self.guidance = guidance or GuidanceConfig()
        self.clip_denoised = clip_denoised
        self.method = method

    @property
    def num_steps(self) -> int:
        return self.bundle.diffusion.num_timesteps

    def initial_latent(self, batch_size: int, seed: int) -> torch.Tensor:
        """Deterministic ``z_T`` of shape (N, C, H, W)."""
        from .utils import make_generator  # noqa: PLC0415

        generator = make_generator(seed)
        return randn(
            (batch_size, *self.bundle.latent_shape),
            generator=generator,
            device=self.bundle.device,
        )

    def make_step_noise(self, batch_size: int, seed: int) -> torch.Tensor:
        """Per-reverse-step noise of shape (T, N_full, C, H, W).

        ``N_full`` accounts for the doubled CFG batch. Drawing the whole schedule
        up front (rather than lazily inside the loop) is what makes the sequence
        identical across posterior draws.
        """
        from .utils import make_generator  # noqa: PLC0415

        generator = make_generator(seed)
        full = batch_size * (2 if self.guidance.enabled else 1)
        return randn(
            (self.num_steps, full, *self.bundle.latent_shape),
            generator=generator,
            device=self.bundle.device,
        )

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        class_labels: Sequence[int] | torch.Tensor,
        *,
        step_noise: torch.Tensor | None = None,
        record_every: int | None = None,
        record_latents: bool = False,
        callback: StepCallback | None = None,
    ) -> SamplingOutput:
        """Run the full reverse process from ``z`` and return the final latents.

        ``step_noise`` is required for ``method="ddpm"``: passing ``None`` would
        silently reintroduce the sampler randomness this class exists to remove.
        """
        bundle = self.bundle
        diffusion = bundle.diffusion
        model = bundle.model

        z_in, model_fn, model_kwargs = build_cfg_inputs(
            z, class_labels, self.guidance, model
        )

        if self.method == "ddpm":
            if step_noise is None:
                raise ValueError(
                    "method='ddpm' needs step_noise; call make_step_noise(...) so the "
                    "same noise can be replayed for every posterior draw."
                )
            if step_noise.shape[0] != self.num_steps:
                raise ValueError(
                    f"step_noise has {step_noise.shape[0]} steps, sampler has "
                    f"{self.num_steps}"
                )
            if step_noise.shape[1:] != z_in.shape:
                raise ValueError(
                    f"step_noise batch shape {tuple(step_noise.shape[1:])} does not "
                    f"match model input {tuple(z_in.shape)}"
                )

        x = z_in
        trace: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []
        trace_steps: list[int] = []
        indices = list(range(self.num_steps))[::-1]

        for step_index, i in enumerate(indices):
            t = torch.full((x.shape[0],), i, device=bundle.device, dtype=torch.long)

            if self.method == "ddim":
                out = diffusion.ddim_sample(
                    model_fn,
                    x,
                    t,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )
                x_next = out["sample"]
            else:
                out = diffusion.p_mean_variance(
                    model_fn,
                    x,
                    t,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=model_kwargs,
                )
                nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
                x_next = (
                    out["mean"]
                    + nonzero_mask
                    * torch.exp(0.5 * out["log_variance"])
                    * step_noise[step_index]
                )

            if record_every and step_index % record_every == 0:
                trace.append(conditional_half(out["pred_xstart"], self.guidance).cpu())
                if record_latents:
                    latents.append(x.cpu())
                trace_steps.append(i)
            if callback is not None:
                callback(step_index, t, x, out)

            x = x_next

        return SamplingOutput(
            latents=conditional_half(x, self.guidance),
            pred_xstart_trace=torch.stack(trace) if trace else None,
            latent_trace=torch.stack(latents) if latents else None,
            trace_steps=tuple(trace_steps),
        )

    @torch.no_grad()
    def predict_xstart(
        self,
        x_t: torch.Tensor,
        t: int | torch.Tensor,
        class_labels: Sequence[int] | torch.Tensor,
        *,
        already_expanded: bool = False,
    ) -> dict[str, torch.Tensor]:
        """One denoiser evaluation at ``(x_t, t)``, returning the p(x_{t-1}|x_t) terms.

        Set ``already_expanded=True`` when ``x_t`` is already a doubled CFG batch
        (e.g. a latent captured mid-trajectory).
        """
        bundle = self.bundle
        model = bundle.model

        if already_expanded:
            x_in = x_t
            labels = torch.as_tensor(class_labels, device=bundle.device, dtype=torch.long)
            if self.guidance.enabled:
                half = x_t.shape[0] // 2
                uncond = torch.full_like(labels, self.guidance.uncond_index(model))
                y = torch.cat([labels, uncond])
                model_fn = model.forward_with_cfg
                model_kwargs = {"y": y, "cfg_scale": self.guidance.cfg_scale}
                if labels.shape[0] != half:
                    raise ValueError("Expected one label per conditional-half element")
            else:
                model_fn = model.forward
                model_kwargs = {"y": labels}
        else:
            x_in, model_fn, model_kwargs = build_cfg_inputs(
                x_t, class_labels, self.guidance, model
            )

        if not torch.is_tensor(t):
            t = torch.full((x_in.shape[0],), int(t), device=bundle.device, dtype=torch.long)
        elif t.ndim == 0:
            t = t.expand(x_in.shape[0])

        out = bundle.diffusion.p_mean_variance(
            model_fn,
            x_in,
            t,
            clip_denoised=self.clip_denoised,
            model_kwargs=model_kwargs,
        )
        return {
            "pred_xstart": conditional_half(out["pred_xstart"], self.guidance),
            "mean": conditional_half(out["mean"], self.guidance),
            "log_variance": conditional_half(out["log_variance"], self.guidance),
        }

    def noisy_latent(
        self, x_start: torch.Tensor, step_index: int, *, seed: int = 0
    ) -> torch.Tensor:
        """``q(x_t | x_0)`` at a reverse-loop index, for probing a fixed ``x_t``."""
        from .utils import make_generator  # noqa: PLC0415

        generator = make_generator(seed)
        noise = randn(
            tuple(x_start.shape), generator=generator, device=x_start.device
        )
        t = torch.full(
            (x_start.shape[0],), int(step_index), device=x_start.device, dtype=torch.long
        )
        return self.bundle.diffusion.q_sample(x_start, t, noise=noise)
