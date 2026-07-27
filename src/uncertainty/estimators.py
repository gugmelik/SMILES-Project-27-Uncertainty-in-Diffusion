"""Uncertainty estimators built on a weight posterior.

Three levels of measurement, in increasing cost and increasing relevance to the
final image:

1. :func:`local_epistemic_uncertainty` — disagreement between posterior draws
   about ``x_0`` at a *single* ``(x_t, t)``. Cheap: one denoiser call per draw.
2. :func:`trajectory_uncertainty` — disagreement between complete generations
   with the diffusion randomness pinned, scored in pixel and feature space.
   Cost: one full reverse trajectory per draw.
3. :func:`decompose_uncertainty` — nested weight x noise sampling with common
   random numbers, split into epistemic and aleatoric parts by the law of total
   variance. Cost: ``M * K`` trajectories.

:func:`local_epistemic_profile` sweeps (1) along a reference trajectory, which
is the per-timestep view this project cares about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .features import FeatureEncoder
from .posteriors import WeightPosterior
from .sampling import FixedNoiseSampler
from .utils import make_generator

LOG_2PI_E = math.log(2 * math.pi * math.e)


def gaussian_entropy(
    variances: torch.Tensor, observation_noise: float = 1e-6
) -> torch.Tensor:
    """Differential entropy of a diagonal Gaussian with the given variances.

        H = 0.5 * sum_j log(2 pi e (v_j + sigma^2))

    ``observation_noise`` keeps the log finite when a feature dimension is
    constant across draws, and sets the floor below which differences in
    variance stop mattering.
    """
    return 0.5 * (LOG_2PI_E + torch.log(variances + observation_noise)).sum(dim=-1)


def _require_samples(num: int, minimum: int, what: str) -> None:
    if num < minimum:
        raise ValueError(f"{what} must be at least {minimum}, got {num}")


# --------------------------------------------------------------------------- #
# 1. Local (per-step) epistemic uncertainty
# --------------------------------------------------------------------------- #


@dataclass
class LocalUncertainty:
    """Posterior disagreement about ``x_0`` at one ``(x_t, t)``."""

    mean_xstart: torch.Tensor
    """(N, C, H, W) posterior-averaged ``x_0`` prediction."""
    variance: torch.Tensor
    """(N, C, H, W) per-element variance across weight draws."""
    uncertainty_map: torch.Tensor
    """(N, 1, H, W) channel-averaged variance — the heatmap."""
    scalar: torch.Tensor
    """(N,) mean variance over channels and space."""
    num_samples: int


@torch.no_grad()
def local_epistemic_uncertainty(
    sampler: FixedNoiseSampler,
    posterior: WeightPosterior,
    x_t: torch.Tensor,
    t: int,
    class_labels: Sequence[int] | torch.Tensor,
    *,
    num_samples: int = 16,
    seed: int = 0,
    already_expanded: bool = False,
) -> LocalUncertainty:
    """Variance of the ``x_0`` prediction across posterior weight draws.

    Uncertainty is measured in ``x_0`` space rather than ``epsilon`` space on
    purpose: the ``eps -> x_0`` conversion divides by ``sqrt(alpha_bar_t)``, so
    equal ``epsilon`` disagreement means wildly different things at ``t = 999``
    and ``t = 1``, and the resulting profile is dominated by the schedule rather
    than by the model.
    """
    _require_samples(num_samples, 2, "num_samples")
    num_samples = posterior.effective_num_samples(num_samples)
    model = sampler.bundle.model
    generator = make_generator(seed)

    predictions = []
    for m in range(num_samples):
        params = posterior.sample(index=m, generator=generator)
        with posterior.use_parameters(model, params):
            out = sampler.predict_xstart(
                x_t, t, class_labels, already_expanded=already_expanded
            )
        predictions.append(out["pred_xstart"])

    stacked = torch.stack(predictions)  # (M, N, C, H, W)
    mean = stacked.mean(dim=0)
    variance = stacked.var(dim=0, unbiased=True)

    return LocalUncertainty(
        mean_xstart=mean,
        variance=variance,
        uncertainty_map=variance.mean(dim=1, keepdim=True),
        scalar=variance.flatten(1).mean(dim=1),
        num_samples=num_samples,
    )


@dataclass
class UncertaintyProfile:
    """Local epistemic uncertainty as a function of denoising step."""

    step_indices: tuple[int, ...]
    """Reverse-loop indices (0 = last denoising step)."""
    timesteps: tuple[int, ...]
    """Corresponding timesteps on the original 1000-step schedule."""
    scalar: torch.Tensor
    """(S, N) mean ``x_0`` variance at each recorded step."""
    maps: torch.Tensor | None = None
    """(S, N, 1, H, W) heatmaps, when ``keep_maps=True``."""
    num_samples: int = 0


@torch.no_grad()
def local_epistemic_profile(
    sampler: FixedNoiseSampler,
    posterior: WeightPosterior,
    *,
    z: torch.Tensor,
    class_labels: Sequence[int] | torch.Tensor,
    step_noise: torch.Tensor | None = None,
    num_samples: int = 8,
    record_every: int = 10,
    seed: int = 0,
    keep_maps: bool = False,
    progress: bool = True,
) -> UncertaintyProfile:
    """Sweep :func:`local_epistemic_uncertainty` along one reference trajectory.

    The reference trajectory is generated under the posterior *mean*, and every
    recorded ``x_t`` is then re-evaluated under ``num_samples`` posterior draws.
    Because all draws see the same ``x_t``, this isolates the denoiser's local
    disagreement from the drift that accumulates when each draw follows its own
    trajectory (use :func:`trajectory_uncertainty` for the latter).
    """
    model = sampler.bundle.model
    diffusion = sampler.bundle.diffusion

    with posterior.use_parameters(model, posterior.mean()):
        reference = sampler.sample(
            z,
            class_labels,
            step_noise=step_noise,
            record_every=record_every,
            record_latents=True,
        )

    if reference.latent_trace is None:
        raise RuntimeError("Reference trajectory recorded no latents")

    steps = reference.trace_steps
    iterator = enumerate(steps)
    if progress:
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415

            iterator = tqdm(list(iterator), desc="Timestep profile")
        except ImportError:
            pass

    scalars, maps = [], []
    for position, step_index in iterator:
        x_t = reference.latent_trace[position].to(sampler.bundle.device)
        local = local_epistemic_uncertainty(
            sampler,
            posterior,
            x_t,
            step_index,
            class_labels,
            num_samples=num_samples,
            seed=seed,
            already_expanded=True,
        )
        scalars.append(local.scalar.cpu())
        if keep_maps:
            maps.append(local.uncertainty_map.cpu())

    timestep_map = list(getattr(diffusion, "timestep_map", range(sampler.num_steps)))
    return UncertaintyProfile(
        step_indices=tuple(steps),
        timesteps=tuple(int(timestep_map[i]) for i in steps),
        scalar=torch.stack(scalars),
        maps=torch.stack(maps) if maps else None,
        num_samples=posterior.effective_num_samples(num_samples),
    )


# --------------------------------------------------------------------------- #
# 2. Whole-sample (trajectory) uncertainty
# --------------------------------------------------------------------------- #


@dataclass
class PosteriorPredictive:
    """Generations from ``M`` weight draws sharing one diffusion noise draw."""

    latents: torch.Tensor
    """(M, N, C, h, w)"""
    images: torch.Tensor | None = None
    """(M, N, 3, H, W) in [-1, 1], when ``decode=True``."""
    xstart_trace: torch.Tensor | None = None
    """(M, S, N, C, h, w) per-step ``x_0`` predictions, when recorded."""
    trace_steps: tuple[int, ...] = ()


@torch.no_grad()
def posterior_predictive(
    sampler: FixedNoiseSampler,
    posterior: WeightPosterior,
    *,
    z: torch.Tensor,
    class_labels: Sequence[int] | torch.Tensor,
    step_noise: torch.Tensor | None = None,
    num_weight_samples: int = 16,
    seed: int = 0,
    decode: bool = True,
    record_every: int | None = None,
    progress: bool = True,
) -> PosteriorPredictive:
    """Generate one image per posterior draw with the diffusion noise held fixed.

    Each ``theta_m`` is drawn once and pinned for the whole reverse trajectory.
    Resampling weights at every step would simulate a model that changes
    mid-generation, which is not a draw from the posterior over models.
    """
    _require_samples(num_weight_samples, 2, "num_weight_samples")
    num_weight_samples = posterior.effective_num_samples(num_weight_samples)
    model = sampler.bundle.model
    generator = make_generator(seed)

    draws = range(num_weight_samples)
    if progress:
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415

            draws = tqdm(draws, desc="Posterior draws")
        except ImportError:
            pass

    latents, traces = [], []
    for m in draws:
        params = posterior.sample(index=m, generator=generator)
        with posterior.use_parameters(model, params):
            out = sampler.sample(
                z, class_labels, step_noise=step_noise, record_every=record_every
            )
        latents.append(out.latents.cpu())
        if out.pred_xstart_trace is not None:
            traces.append(out.pred_xstart_trace)
        trace_steps = out.trace_steps

    stacked = torch.stack(latents)
    images = None
    if decode:
        images = torch.stack(
            [
                sampler.bundle.decode(item.to(sampler.bundle.device)).cpu()
                for item in stacked
            ]
        )

    return PosteriorPredictive(
        latents=stacked,
        images=images,
        xstart_trace=torch.stack(traces) if traces else None,
        trace_steps=trace_steps if traces else (),
    )


@dataclass
class TrajectoryUncertainty:
    """Posterior disagreement between complete generations."""

    pixel_uncertainty: torch.Tensor
    """(N,) mean pixel variance across weight draws."""
    pixel_map: torch.Tensor
    """(N, 1, H, W) spatial pixel-variance heatmap."""
    semantic_uncertainty: torch.Tensor
    """(N,) mean feature variance — the score to prefer over pixel variance."""
    semantic_entropy: torch.Tensor
    """(N,) Gaussian entropy of the posterior-predictive feature distribution."""
    feature_dim: int
    num_weight_samples: int
    encoder_name: str = ""
    extras: dict = field(default_factory=dict)


@torch.no_grad()
def trajectory_uncertainty(
    predictive: PosteriorPredictive,
    encoder: FeatureEncoder | None = None,
    *,
    observation_noise: float = 1e-6,
) -> TrajectoryUncertainty:
    """Score a set of posterior-predictive generations.

    Reports pixel variance (easy to visualise, easy to fool) alongside feature
    variance and the Gaussian entropy of the predictive feature distribution.
    Prefer the semantic numbers when ranking samples.
    """
    if predictive.images is None:
        raise ValueError("posterior_predictive(..., decode=True) is required")

    images = predictive.images  # (M, N, 3, H, W)
    num_draws, batch = images.shape[0], images.shape[1]
    _require_samples(num_draws, 2, "number of weight draws")

    pixel_variance = images.float().var(dim=0, unbiased=True)  # (N, 3, H, W)
    pixel_map = pixel_variance.mean(dim=1, keepdim=True)
    pixel_scalar = pixel_variance.flatten(1).mean(dim=1)

    if encoder is None:
        zeros = torch.zeros(batch)
        return TrajectoryUncertainty(
            pixel_uncertainty=pixel_scalar,
            pixel_map=pixel_map,
            semantic_uncertainty=zeros,
            semantic_entropy=zeros,
            feature_dim=0,
            num_weight_samples=num_draws,
        )

    features = torch.stack(
        [encoder(images[m]) for m in range(num_draws)]
    )  # (M, N, d)
    feature_variance = features.var(dim=0, unbiased=True)  # (N, d)

    return TrajectoryUncertainty(
        pixel_uncertainty=pixel_scalar,
        pixel_map=pixel_map,
        semantic_uncertainty=feature_variance.mean(dim=-1),
        semantic_entropy=gaussian_entropy(feature_variance, observation_noise),
        feature_dim=features.shape[-1],
        num_weight_samples=num_draws,
        encoder_name=getattr(encoder, "name", type(encoder).__name__),
        extras={"feature_variance": feature_variance},
    )


# --------------------------------------------------------------------------- #
# 3. Epistemic / aleatoric decomposition
# --------------------------------------------------------------------------- #


@dataclass
class UncertaintyDecomposition:
    """Law-of-total-variance split of predictive uncertainty."""

    epistemic: torch.Tensor
    """(N,) Var_theta[E_z[Y]] — disagreement between plausible models."""
    aleatoric: torch.Tensor
    """(N,) E_theta[Var_z[Y]] — variability a single model still supports."""
    total: torch.Tensor
    """(N,) epistemic + aleatoric."""
    epistemic_raw: torch.Tensor
    """(N,) uncorrected estimate, before removing the aleatoric leak."""
    num_weight_samples: int
    num_noise_samples: int
    feature_dim: int
    encoder_name: str = ""
    bias_corrected: bool = True

    def ratio(self) -> torch.Tensor:
        """Epistemic share of total uncertainty, in [0, 1]."""
        return self.epistemic / self.total.clamp_min(1e-12)


def decompose_feature_tensor(
    features: torch.Tensor, *, bias_correct: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split an ``(M, K, N, d)`` feature tensor by the law of total variance.

    Separated from the sampling loop so a cached set of features can be
    re-decomposed without regenerating images — and so the estimator itself can
    be tested against data with known variance components.

    Returns ``(epistemic, aleatoric, epistemic_raw)``, each of shape ``(N,)`` and
    each a *trace* of a covariance (summed over feature dimensions).
    """
    if features.ndim != 4:
        raise ValueError(f"Expected (M, K, N, d), got {tuple(features.shape)}")
    num_weights, num_noise = features.shape[0], features.shape[1]
    _require_samples(num_weights, 2, "M (weight draws)")
    _require_samples(num_noise, 2, "K (noise draws)")

    features = features.double()
    mu_m = features.mean(dim=1)  # (M, N, d)
    mu = mu_m.mean(dim=0)  # (N, d)

    epistemic_raw = (mu_m - mu.unsqueeze(0)).pow(2).sum(dim=-1).sum(dim=0) / (
        num_weights - 1
    )
    aleatoric = (
        (features - mu_m.unsqueeze(1)).pow(2).sum(dim=-1).sum(dim=1) / (num_noise - 1)
    ).mean(dim=0)

    epistemic = epistemic_raw
    if bias_correct:
        # E[Var_m(mu_m)] = Var_theta + Var_z / K, so the second term is a leak.
        epistemic = (epistemic_raw - aleatoric / num_noise).clamp_min(0.0)

    return epistemic.float(), aleatoric.float(), epistemic_raw.float()


@torch.no_grad()
def decompose_uncertainty(
    sampler: FixedNoiseSampler,
    posterior: WeightPosterior,
    encoder: FeatureEncoder,
    *,
    class_labels: Sequence[int] | torch.Tensor,
    num_weight_samples: int = 8,
    num_noise_samples: int = 4,
    weight_seed: int = 0,
    noise_seed: int = 1000,
    bias_correct: bool = True,
    progress: bool = True,
) -> UncertaintyDecomposition:
    """Nested weight x noise sampling, split by the law of total variance.

        Var[Y] = Var_theta[E_z[Y|theta]] + E_theta[Var_z[Y|theta]]
                 \\_____ epistemic ______/   \\______ aleatoric ______/

    The same ``K`` noise seeds are reused for every weight draw (common random
    numbers). Without that, ordinary seed-to-seed variation leaks into the
    epistemic term and the split is meaningless.

    ``bias_correct`` removes the residual leak that remains even with shared
    seeds: the sample variance of ``mu_m`` estimates
    ``Var_theta + Var_z / K``, so ``Var_z / K`` is subtracted off. Without it,
    the epistemic term is inflated whenever ``K`` is small. When the corrected
    value clamps to zero, the run simply cannot resolve epistemic uncertainty at
    this ``K`` — raise ``K`` rather than reading the zero as a finding.
    """
    _require_samples(num_weight_samples, 2, "num_weight_samples")
    _require_samples(num_noise_samples, 2, "num_noise_samples")
    num_weight_samples = posterior.effective_num_samples(num_weight_samples)

    model = sampler.bundle.model
    labels = torch.as_tensor(class_labels, dtype=torch.long)
    if labels.ndim == 0:
        labels = labels[None]
    batch = labels.shape[0]

    # Common random numbers: one shared (z_T, step-noise) pair per k.
    shared_noise = [
        (
            sampler.initial_latent(batch, seed=noise_seed + k),
            sampler.make_step_noise(batch, seed=noise_seed + k)
            if sampler.method == "ddpm"
            else None,
        )
        for k in range(num_noise_samples)
    ]

    weight_generator = make_generator(weight_seed)
    total_runs = num_weight_samples * num_noise_samples
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415

            bar = tqdm(total=total_runs, desc="Nested M x K")
        except ImportError:
            pass

    features: list[list[torch.Tensor]] = []
    for m in range(num_weight_samples):
        params = posterior.sample(index=m, generator=weight_generator)
        per_draw = []
        with posterior.use_parameters(model, params):
            for z, step_noise in shared_noise:
                out = sampler.sample(z, labels, step_noise=step_noise)
                images = sampler.bundle.decode(out.latents)
                per_draw.append(encoder(images))
                if bar is not None:
                    bar.update(1)
        features.append(torch.stack(per_draw))
    if bar is not None:
        bar.close()

    y = torch.stack(features)  # (M, K, N, d)
    epistemic, aleatoric, epistemic_raw = decompose_feature_tensor(
        y, bias_correct=bias_correct
    )

    return UncertaintyDecomposition(
        epistemic=epistemic,
        aleatoric=aleatoric,
        total=epistemic + aleatoric,
        epistemic_raw=epistemic_raw,
        num_weight_samples=num_weight_samples,
        num_noise_samples=num_noise_samples,
        feature_dim=y.shape[-1],
        encoder_name=getattr(encoder, "name", type(encoder).__name__),
        bias_corrected=bias_correct,
    )
