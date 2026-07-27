"""Bayesian uncertainty estimation for Diffusion Transformers.

A standard diffusion model captures only *aleatoric* variability: sample it a
thousand times and you learn about the spread of ``p(x | c)`` under one fixed
parameter vector, not about how much the model itself could have been wrong.
Measuring *epistemic* uncertainty needs a distribution over the denoiser's
weights.

This package implements that pipeline on top of the vendored DiT checkout:

``posteriors`` / ``laplace`` / ``lora``
    Approximate posteriors ``q(theta)`` over a chosen subset of weights — a
    post-hoc diagonal Laplace fit, a variational Bayesian LoRA fit, and deep
    ensembles as the reference baseline.
``sampling``
    Reverse diffusion with the initial latent and every step's noise pinned, so
    that variation across draws is attributable to the weights.
``estimators``
    Per-step ``x_0`` variance, whole-sample posterior-predictive uncertainty in
    pixel and feature space, and the epistemic/aleatoric split.
``features`` / ``validation``
    Semantic encoders, and the checks that decide whether a score is useful.

Typical use::

    from src.uncertainty import load_dit, load_posterior, FixedNoiseSampler
    from src.uncertainty import posterior_predictive, trajectory_uncertainty

    bundle = load_dit(num_sampling_steps=250)
    posterior = load_posterior("results/posteriors/laplace_last_layer.pt", bundle.device)
    sampler = FixedNoiseSampler(bundle)

    z = sampler.initial_latent(1, seed=0)
    noise = sampler.make_step_noise(1, seed=0)
    predictive = posterior_predictive(
        sampler, posterior, z=z, class_labels=[207],
        step_noise=noise, num_weight_samples=16,
    )
    score = trajectory_uncertainty(predictive)
"""

from .data import cache_latents, cached_latent_loader, imagenet_latent_loader
from .dit import DiTBundle, bootstrap_dit, load_dit, select_parameters
from .estimators import (
    LocalUncertainty,
    PosteriorPredictive,
    TrajectoryUncertainty,
    UncertaintyDecomposition,
    UncertaintyProfile,
    decompose_feature_tensor,
    decompose_uncertainty,
    gaussian_entropy,
    local_epistemic_profile,
    local_epistemic_uncertainty,
    posterior_predictive,
    trajectory_uncertainty,
)
from .features import (
    CLIPEncoder,
    FeatureEncoder,
    PixelEncoder,
    TimmEncoder,
    VAEEncoder,
    build_encoder,
)
from .laplace import LaplaceConfig, fit_diagonal_laplace, optimize_prior_precision
from .lora import (
    BayesianLoRAPosterior,
    LoRAFitConfig,
    fit_variational_lora,
    inject_variational_lora,
    load_lora_posterior,
    lora_deep_ensemble,
)
from .posteriors import (
    DeepEnsemblePosterior,
    DiagonalGaussianPosterior,
    MAPPosterior,
    WeightPosterior,
    load_posterior,
)
from .sampling import FixedNoiseSampler, GuidanceConfig, SamplingOutput
from .validation import (
    UncertaintyReport,
    auroc,
    evaluate_uncertainty,
    pearson_corr,
    selective_curve,
    spearman_corr,
)

__all__ = [
    "BayesianLoRAPosterior",
    "CLIPEncoder",
    "DeepEnsemblePosterior",
    "DiTBundle",
    "DiagonalGaussianPosterior",
    "FeatureEncoder",
    "FixedNoiseSampler",
    "GuidanceConfig",
    "LaplaceConfig",
    "LoRAFitConfig",
    "LocalUncertainty",
    "MAPPosterior",
    "PixelEncoder",
    "PosteriorPredictive",
    "SamplingOutput",
    "TimmEncoder",
    "TrajectoryUncertainty",
    "UncertaintyDecomposition",
    "UncertaintyProfile",
    "UncertaintyReport",
    "VAEEncoder",
    "WeightPosterior",
    "auroc",
    "bootstrap_dit",
    "build_encoder",
    "cache_latents",
    "cached_latent_loader",
    "decompose_feature_tensor",
    "decompose_uncertainty",
    "evaluate_uncertainty",
    "fit_diagonal_laplace",
    "fit_variational_lora",
    "gaussian_entropy",
    "imagenet_latent_loader",
    "inject_variational_lora",
    "load_dit",
    "load_lora_posterior",
    "load_posterior",
    "local_epistemic_profile",
    "local_epistemic_uncertainty",
    "lora_deep_ensemble",
    "optimize_prior_precision",
    "pearson_corr",
    "posterior_predictive",
    "select_parameters",
    "selective_curve",
    "spearman_corr",
    "trajectory_uncertainty",
]
