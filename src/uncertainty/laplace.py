"""Post-hoc diagonal Laplace approximation for a pretrained denoiser.

The cheapest way to turn a fixed ``theta_MAP`` into a posterior is to fit a
Gaussian around it using curvature of the training loss:

    q(theta) = N(theta_MAP, (lambda * I + F)^-1)

where ``F`` is the diagonal empirical Fisher of the diffusion loss. Only the
parameter subset chosen by ``dit.select_parameters`` is treated as random, which
is what makes this tractable for DiT-XL/2 (the last layer is ~37k parameters).

This is the "last-layer Laplace" family used by BayesDiff. It is easy and
training-free, but it can only see uncertainty that is expressible in the
Bayesianised subset — compare it against ``lora.py`` and a deep ensemble before
drawing conclusions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn as nn

from .dit import count_parameters, freeze, select_parameters, unconditional_class
from .posteriors import DiagonalGaussianPosterior

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class LaplaceConfig:
    """Settings for :func:`fit_diagonal_laplace`."""

    parameter_spec: str = "last_layer"
    num_samples: int = 512
    """Number of (latent, timestep) pairs used to accumulate the Fisher."""
    timesteps_per_image: int = 1
    """Independent timestep draws per image; >1 reuses the data loader less."""
    prior_precision: float = 1.0
    fisher_scale: float = 1.0
    """Multiplies the accumulated Fisher, e.g. ``dataset_size / num_samples``."""
    label_dropout_prob: float = 0.1
    """Matches DiT's classifier-free-guidance training dropout."""
    posterior_scale: float = 1.0
    seed: int = 0
    optimize_prior: bool = True
    optimize_prior_steps: int = 300
    optimize_prior_lr: float = 0.1

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LaplaceFitResult:
    posterior: DiagonalGaussianPosterior
    fisher: dict[str, torch.Tensor]
    map_loss_sum: float
    num_samples: int
    config: LaplaceConfig = field(repr=False)


def _iter_fit_batches(
    data: Iterable[Batch], num_samples: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield single examples from a batched iterable until ``num_samples``."""
    produced = 0
    for latents, labels in data:
        for i in range(latents.shape[0]):
            if produced >= num_samples:
                return
            yield latents[i : i + 1], labels[i : i + 1]
            produced += 1
    if produced < num_samples:
        raise ValueError(
            f"Data source exhausted after {produced} examples, need {num_samples}. "
            "Reduce LaplaceConfig.num_samples or make the loader repeat."
        )


def fit_diagonal_laplace(
    model: nn.Module,
    diffusion,
    data: Iterable[Batch],
    config: LaplaceConfig | None = None,
    *,
    progress: bool = True,
) -> LaplaceFitResult:
    """Accumulate the diagonal empirical Fisher and build a Gaussian posterior.

    ``data`` yields ``(latents, labels)`` where ``latents`` are already
    VAE-encoded and scaled (see ``data.imagenet_latent_loader``). Gradients are
    taken one example at a time: the empirical Fisher is a sum of *per-sample*
    squared gradients, and a batch gradient is not a substitute for it.

    ``diffusion`` should be the full 1000-step process, not a respaced sampler,
    so the curvature reflects the whole noise schedule.
    """
    config = config or LaplaceConfig()
    device = next(model.parameters()).device

    names = select_parameters(model, config.parameter_spec)
    freeze(model)
    lookup = dict(model.named_parameters())
    targets = [lookup[name] for name in names]
    for parameter in targets:
        parameter.requires_grad_(True)

    fisher = {name: torch.zeros_like(lookup[name]) for name in names}
    uncond = unconditional_class(model)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    loss_sum = 0.0
    seen = 0

    was_training = model.training
    model.eval()  # keeps LabelEmbedder from applying its own dropout

    stream = _iter_fit_batches(data, config.num_samples)
    if progress:
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415

            stream = tqdm(stream, total=config.num_samples, desc="Laplace Fisher")
        except ImportError:
            pass

    # The notebooks call torch.set_grad_enabled(False) at import; re-enable
    # locally so fitting works regardless of the ambient grad mode.
    with torch.enable_grad():
        for latents, labels in stream:
            latents = latents.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            for _ in range(config.timesteps_per_image):
                t = torch.randint(
                    0, diffusion.num_timesteps, (latents.shape[0],), generator=generator
                ).to(device)

                y = labels.clone()
                if config.label_dropout_prob > 0:
                    drop = (
                        torch.rand(y.shape, generator=generator)
                        < config.label_dropout_prob
                    ).to(device)
                    y = torch.where(drop, torch.full_like(y, uncond), y)

                terms = diffusion.training_losses(
                    model, latents, t, model_kwargs={"y": y}
                )
                # mean_flat gives a per-sample mean over the D latent dims; undo it
                # so the loss is the Gaussian NLL 0.5 * ||eps - eps_theta||^2 and the
                # Fisher is on the same scale as the prior precision.
                num_dims = latents[0].numel()
                loss = 0.5 * num_dims * terms["mse"].sum()

                grads = torch.autograd.grad(loss, targets, allow_unused=False)
                for name, grad in zip(names, grads):
                    fisher[name] += grad.detach() ** 2

                loss_sum += float(loss.detach())
                seen += 1

    if was_training:
        model.train()

    scaled_fisher = {name: config.fisher_scale * value for name, value in fisher.items()}
    means = {name: lookup[name].detach().clone() for name in names}

    prior_precision = config.prior_precision
    if config.optimize_prior:
        prior_precision = optimize_prior_precision(
            means,
            scaled_fisher,
            map_loss_sum=loss_sum,
            init=config.prior_precision,
            steps=config.optimize_prior_steps,
            lr=config.optimize_prior_lr,
        )

    std = {
        name: torch.rsqrt(prior_precision + scaled_fisher[name]) for name in names
    }
    posterior = DiagonalGaussianPosterior(
        names, means, std, scale=config.posterior_scale
    )

    if progress:
        total = count_parameters(model, names)
        mean_std = float(
            torch.cat([value.flatten() for value in std.values()]).mean()
        )
        print(
            f"Laplace fit: {total:,} Bayesian parameters over {seen} samples | "
            f"prior_precision={prior_precision:.4g} | mean posterior std={mean_std:.4g}"
        )

    return LaplaceFitResult(
        posterior=posterior,
        fisher=scaled_fisher,
        map_loss_sum=loss_sum,
        num_samples=seen,
        config=config,
    )


def log_marginal_likelihood(
    means: dict[str, torch.Tensor],
    fisher: dict[str, torch.Tensor],
    *,
    map_loss_sum: float,
    prior_precision: torch.Tensor | float,
) -> torch.Tensor:
    """Diagonal-Laplace evidence, up to constants independent of the precision.

        log Z ~= -L(theta_MAP) - (lambda/2)||theta_MAP||^2
                 + (D/2) log lambda - (1/2) sum_i log(lambda + F_i)
    """
    if not torch.is_tensor(prior_precision):
        prior_precision = torch.tensor(float(prior_precision))
    flat_mean = torch.cat([value.detach().flatten().cpu() for value in means.values()])
    flat_fisher = torch.cat(
        [value.detach().flatten().cpu() for value in fisher.values()]
    )
    num_params = flat_mean.numel()

    return (
        -map_loss_sum
        - 0.5 * prior_precision * flat_mean.pow(2).sum()
        + 0.5 * num_params * torch.log(prior_precision)
        - 0.5 * torch.log(prior_precision + flat_fisher).sum()
    )


def optimize_prior_precision(
    means: dict[str, torch.Tensor],
    fisher: dict[str, torch.Tensor],
    *,
    map_loss_sum: float,
    init: float = 1.0,
    steps: int = 300,
    lr: float = 0.1,
) -> float:
    """Tune the prior precision by maximising the Laplace evidence.

    Cheap: after the Fisher is accumulated this touches no data and no forward
    passes, so it is a strictly better default than an arbitrary ``lambda = 1``.
    """
    with torch.enable_grad():
        log_precision = torch.tensor(math.log(max(init, 1e-8)), requires_grad=True)
        optimizer = torch.optim.Adam([log_precision], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            loss = -log_marginal_likelihood(
                means,
                fisher,
                map_loss_sum=map_loss_sum,
                prior_precision=log_precision.exp(),
            )
            loss.backward()
            optimizer.step()

    return float(log_precision.detach().exp())


def calibrate_posterior_scale(
    posterior: DiagonalGaussianPosterior,
    reference: DiagonalGaussianPosterior | Sequence[float],
    *,
    target_std: float | None = None,
) -> float:
    """Rescale a posterior so its weight-space spread matches a reference.

    A Laplace approximation is usually over-confident relative to a deep
    ensemble. When an ensemble over the same parameter subset is available, this
    matches the mean per-parameter standard deviation and returns the applied
    scale.
    """
    own_std = float(
        torch.cat([value.flatten() for value in posterior.std.values()]).mean()
    )
    if target_std is None:
        if isinstance(reference, DiagonalGaussianPosterior):
            target_std = float(
                torch.cat([v.flatten() for v in reference.std.values()]).mean()
            )
        else:
            target_std = float(torch.as_tensor(list(reference)).mean())

    scale = target_std / max(own_std, 1e-12)
    posterior.scale = scale
    return scale
