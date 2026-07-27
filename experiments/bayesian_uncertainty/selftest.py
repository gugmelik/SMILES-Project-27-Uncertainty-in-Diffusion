#!/usr/bin/env python
"""End-to-end check of the Bayesian uncertainty pipeline on a toy DiT.

Builds a tiny randomly-initialised DiT (8x8 latents, depth 2) and a stub VAE so
the whole stack runs on CPU in well under a minute with no checkpoint download.
It verifies behaviour, not image quality — most importantly the invariants that
would silently produce meaningless numbers:

* a point-mass posterior must report *exactly zero* epistemic uncertainty;
* a spread-out posterior must report more than zero;
* with the noise pinned, repeated runs must be bit-identical;
* freshly injected LoRA adapters must not change the model's output.

Run::

    python experiments/bayesian_uncertainty/selftest.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.uncertainty import (  # noqa: E402
    DiTBundle,
    FixedNoiseSampler,
    GuidanceConfig,
    PixelEncoder,
    VAEEncoder,
    decompose_uncertainty,
    gaussian_entropy,
    decompose_feature_tensor,
    local_epistemic_profile,
    local_epistemic_uncertainty,
    posterior_predictive,
    trajectory_uncertainty,
)
from src.uncertainty.dit import bootstrap_dit, select_parameters  # noqa: E402
from src.uncertainty.laplace import LaplaceConfig, fit_diagonal_laplace  # noqa: E402
from src.uncertainty.lora import (  # noqa: E402
    BayesianLoRAPosterior,
    LoRAFitConfig,
    fit_variational_lora,
    inject_variational_lora,
    lora_deep_ensemble,
)
from src.uncertainty.posteriors import (  # noqa: E402
    DiagonalGaussianPosterior,
    MAPPosterior,
)
from src.uncertainty.validation import (  # noqa: E402
    auroc,
    evaluate_uncertainty,
    selective_curve,
    spearman_corr,
)

LATENT_SIZE = 8
IMAGE_SIZE = 64
NUM_CLASSES = 8
STEPS = 6


class StubVAE(nn.Module):
    """Minimal stand-in exposing the AutoencoderKL surface the code touches."""

    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(4, 3, kernel_size=8, stride=8)
        self.down = nn.Conv2d(3, 4, kernel_size=8, stride=8)

    def decode(self, latents):
        return SimpleNamespace(sample=torch.tanh(self.up(latents)))

    def encode(self, images):
        mean = self.down(images)
        return SimpleNamespace(
            latent_dist=SimpleNamespace(mean=mean, sample=lambda: mean)
        )


def build_bundle(seed: int = 0) -> DiTBundle:
    bootstrap_dit()
    from diffusion import create_diffusion  # noqa: PLC0415
    from models import DiT  # noqa: PLC0415

    torch.manual_seed(seed)
    model = DiT(
        input_size=LATENT_SIZE,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=NUM_CLASSES,
    ).eval()
    # A freshly initialised DiT zeroes its output layers, which would make every
    # variance identically zero and hide real bugs. Perturb them.
    with torch.no_grad():
        for parameter in model.final_layer.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))

    return DiTBundle(
        model=model,
        vae=StubVAE().eval(),
        diffusion=create_diffusion(str(STEPS)),
        device=torch.device("cpu"),
        latent_size=LATENT_SIZE,
        image_size=IMAGE_SIZE,
        num_sampling_steps=STEPS,
    )


def wide_posterior(model, spec: str = "last_layer", sigma: float = 0.02):
    names = select_parameters(model, spec)
    lookup = dict(model.named_parameters())
    mean = {name: lookup[name].detach().clone() for name in names}
    std = {name: torch.full_like(value, sigma) for name, value in mean.items()}
    return DiagonalGaussianPosterior(names, mean, std)


def random_data(num_batches: int = 8, batch_size: int = 2, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(num_batches):
        yield (
            torch.randn(batch_size, 4, LATENT_SIZE, LATENT_SIZE, generator=generator),
            torch.randint(0, NUM_CLASSES, (batch_size,), generator=generator),
        )


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_use_parameters(bundle):
    posterior = MAPPosterior(bundle.model, select_parameters(bundle.model, "last_layer"))
    name = posterior.parameter_names[0]
    original = dict(bundle.model.named_parameters())[name].detach().clone()

    replacement = {name: torch.zeros_like(original)}
    with posterior.use_parameters(bundle.model, replacement):
        inside = dict(bundle.model.named_parameters())[name]
        assert torch.equal(inside, torch.zeros_like(original)), "swap did not apply"

    after = dict(bundle.model.named_parameters())[name]
    assert torch.equal(after, original), "weights were not restored on exit"
    return f"{len(posterior.parameter_names)} tensors swap and restore cleanly"


def check_map_is_zero_uncertainty(bundle, sampler):
    posterior = MAPPosterior(bundle.model, select_parameters(bundle.model, "last_layer"))
    x_t = sampler.initial_latent(1, seed=3)

    local = local_epistemic_uncertainty(
        sampler, posterior, x_t, STEPS - 1, [1], num_samples=4
    )
    assert float(local.scalar.max()) == 0.0, f"expected 0, got {float(local.scalar.max())}"

    predictive = posterior_predictive(
        sampler,
        posterior,
        z=x_t,
        class_labels=[1],
        step_noise=sampler.make_step_noise(1, seed=3),
        num_weight_samples=3,
        progress=False,
    )
    score = trajectory_uncertainty(predictive, PixelEncoder(size=16))
    assert float(score.pixel_uncertainty[0]) == 0.0, "point mass produced pixel variance"
    assert float(score.semantic_uncertainty[0]) == 0.0, "point mass produced feature variance"
    return "point-mass posterior gives exactly zero epistemic uncertainty"


def check_spread_posterior_is_positive(bundle, sampler):
    posterior = wide_posterior(bundle.model)
    x_t = sampler.initial_latent(1, seed=3)

    local = local_epistemic_uncertainty(
        sampler, posterior, x_t, STEPS - 1, [1], num_samples=6
    )
    assert float(local.scalar[0]) > 0, "spread posterior gave zero local uncertainty"
    assert local.uncertainty_map.shape == (1, 1, LATENT_SIZE, LATENT_SIZE), (
        f"unexpected heatmap shape {tuple(local.uncertainty_map.shape)}"
    )
    assert torch.isfinite(local.variance).all(), "non-finite variance"
    return f"local uncertainty = {float(local.scalar[0]):.3e} with a 8x8 heatmap"


def check_fixed_noise_determinism(bundle, sampler):
    posterior = wide_posterior(bundle.model)
    z = sampler.initial_latent(1, seed=5)
    noise = sampler.make_step_noise(1, seed=5)

    runs = [
        posterior_predictive(
            sampler,
            posterior,
            z=z,
            class_labels=[2],
            step_noise=noise,
            num_weight_samples=3,
            seed=11,
            progress=False,
        )
        for _ in range(2)
    ]
    assert torch.equal(runs[0].latents, runs[1].latents), (
        "same seed produced different generations; the weight draws or the step "
        "noise are not reproducible"
    )

    different = posterior_predictive(
        sampler,
        posterior,
        z=z,
        class_labels=[2],
        step_noise=noise,
        num_weight_samples=3,
        seed=12,
        progress=False,
    )
    assert not torch.equal(runs[0].latents, different.latents), (
        "different weight seeds produced identical generations"
    )
    return "identical seeds replay exactly; different weight seeds diverge"


def check_lora_is_output_preserving(bundle):
    from models import DiT  # noqa: PLC0415

    torch.manual_seed(1)
    model = DiT(
        input_size=LATENT_SIZE,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=NUM_CLASSES,
    ).eval()

    x = torch.randn(2, 4, LATENT_SIZE, LATENT_SIZE)
    t = torch.tensor([3, 3])
    y = torch.tensor([1, 2])
    with torch.no_grad():
        before = model(x, t, y)

    adapters = inject_variational_lora(model, "late_blocks:1", rank=2)
    with torch.no_grad():
        after = model(x, t, y)
    assert torch.allclose(before, after, atol=1e-6), (
        "LoRA injection changed the model output; adapters must start at Delta W = 0"
    )

    posterior = BayesianLoRAPosterior(adapters)
    # Widen the posterior so a draw is visibly different from the mean.
    for adapter in adapters.values():
        with torch.no_grad():
            adapter.lora_B_rho.fill_(0.5)
            adapter.lora_A_rho.fill_(0.5)
    with posterior.use_parameters(model, posterior.sample(generator=torch.Generator().manual_seed(0))):
        with torch.no_grad():
            perturbed = model(x, t, y)
    assert not torch.allclose(before, perturbed, atol=1e-6), (
        "installing a LoRA draw did not change the output"
    )

    with torch.no_grad():
        restored = model(x, t, y)
    assert torch.allclose(before, restored, atol=1e-6), "LoRA buffers were not restored"
    return f"{len(adapters)} adapters inject transparently and perturb when sampled"


def check_laplace_fit(bundle):
    from diffusion import create_diffusion  # noqa: PLC0415

    result = fit_diagonal_laplace(
        bundle.model,
        create_diffusion(""),
        random_data(num_batches=6, batch_size=2),
        LaplaceConfig(
            parameter_spec="last_layer",
            num_samples=8,
            optimize_prior_steps=40,
        ),
        progress=False,
    )
    stds = torch.cat([value.flatten() for value in result.posterior.std.values()])
    assert torch.isfinite(stds).all(), "non-finite posterior std"
    assert float(stds.min()) > 0, "non-positive posterior std"

    fisher = torch.cat([value.flatten() for value in result.fisher.values()])
    assert float(fisher.max()) > 0, "Fisher is identically zero — no curvature captured"
    return (
        f"fitted {stds.numel():,} params, mean std={float(stds.mean()):.3e}, "
        f"max Fisher={float(fisher.max()):.3e}"
    )


def check_lora_fit(bundle):
    from diffusion import create_diffusion  # noqa: PLC0415
    from models import DiT  # noqa: PLC0415

    torch.manual_seed(2)
    model = DiT(
        input_size=LATENT_SIZE,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=NUM_CLASSES,
    ).eval()

    def repeating():
        while True:
            yield from random_data(num_batches=4, batch_size=2, seed=7)

    posterior = fit_variational_lora(
        model,
        create_diffusion(""),
        repeating(),
        LoRAFitConfig(target_spec="last_layer", rank=2, steps=6, batch_size=2, log_every=3),
        progress=False,
    )
    history = getattr(posterior, "history", [])
    assert history, "no training history recorded"
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history), "non-finite ELBO"

    ensemble = lora_deep_ensemble(posterior, num_members=3)
    assert ensemble.num_members == 3
    assert ensemble.effective_num_samples(10) == 3, "ensemble ignored its member limit"
    return f"ELBO ran ({len(history)} logs), ensemble of 3 members frozen"


def check_decomposition_math():
    """Recover known variance components from synthetic features.

    Build ``y[m,k] = mu_m + noise`` with ``Var_theta`` and ``Var_z`` chosen by
    hand, then check the estimator returns them. This is the only check that can
    tell a correct decomposition from a plausible-looking one.
    """
    torch.manual_seed(0)
    num_weights, num_noise = 600, 40
    true_epistemic, true_aleatoric = 1.0, 4.0

    mu_m = torch.randn(num_weights, 1, 1) * true_epistemic**0.5
    y = mu_m.unsqueeze(1) + torch.randn(
        num_weights, num_noise, 1, 1
    ) * true_aleatoric**0.5

    epistemic, aleatoric, raw = decompose_feature_tensor(y)
    epistemic, aleatoric, raw = float(epistemic[0]), float(aleatoric[0]), float(raw[0])

    assert abs(aleatoric - true_aleatoric) / true_aleatoric < 0.1, (
        f"aleatoric {aleatoric:.3f} != {true_aleatoric}"
    )
    assert abs(epistemic - true_epistemic) / true_epistemic < 0.25, (
        f"epistemic {epistemic:.3f} != {true_epistemic}"
    )
    # Without the correction the epistemic term is inflated by Var_z / K.
    assert raw > epistemic, "bias correction did not reduce the epistemic term"
    assert abs(raw - (true_epistemic + true_aleatoric / num_noise)) < 0.3, (
        f"uncorrected estimate {raw:.3f} does not match the predicted leak"
    )

    uncorrected, _, _ = decompose_feature_tensor(y, bias_correct=False)
    assert abs(float(uncorrected[0]) - raw) < 1e-6
    return (
        f"recovered epistemic={epistemic:.3f} (true 1.0), "
        f"aleatoric={aleatoric:.3f} (true 4.0), uncorrected={raw:.3f}"
    )


def check_decomposition(bundle, sampler):
    encoder = PixelEncoder(size=16)

    point_mass = MAPPosterior(bundle.model, select_parameters(bundle.model, "last_layer"))
    flat = decompose_uncertainty(
        sampler,
        point_mass,
        encoder,
        class_labels=[1],
        num_weight_samples=2,
        num_noise_samples=3,
        progress=False,
    )
    assert float(flat.epistemic[0]) == 0.0, "point mass produced epistemic uncertainty"
    assert float(flat.epistemic_raw[0]) == 0.0, "point mass produced raw epistemic term"
    assert float(flat.aleatoric[0]) > 0, "no aleatoric variability across noise seeds"

    spread = decompose_uncertainty(
        sampler,
        wide_posterior(bundle.model, sigma=0.05),
        encoder,
        class_labels=[1],
        num_weight_samples=3,
        num_noise_samples=3,
        progress=False,
    )
    assert float(spread.epistemic_raw[0]) > 0, "spread posterior gave no weight disagreement"
    assert float(spread.epistemic_raw[0]) >= float(spread.epistemic[0]), (
        "bias correction increased the epistemic term"
    )
    ratio = float(spread.ratio()[0])
    assert 0.0 <= ratio <= 1.0, f"epistemic ratio out of range: {ratio}"
    return (
        f"point mass -> epistemic=0, aleatoric={float(flat.aleatoric[0]):.3e}; "
        f"spread -> raw epistemic={float(spread.epistemic_raw[0]):.3e}"
    )


def check_profile(bundle, sampler):
    posterior = wide_posterior(bundle.model)
    profile = local_epistemic_profile(
        sampler,
        posterior,
        z=sampler.initial_latent(1, seed=9),
        class_labels=[3],
        step_noise=sampler.make_step_noise(1, seed=9),
        num_samples=4,
        record_every=2,
        keep_maps=True,
        progress=False,
    )
    expected = len(range(0, STEPS, 2))
    assert profile.scalar.shape == (expected, 1), (
        f"expected {(expected, 1)}, got {tuple(profile.scalar.shape)}"
    )
    assert profile.maps is not None and profile.maps.shape[0] == expected
    assert len(profile.timesteps) == expected
    assert all(t < 1000 for t in profile.timesteps), "timestep mapping out of range"
    return f"profile over {expected} steps, timesteps {profile.timesteps}"


def check_vae_encoder(bundle):
    encoder = VAEEncoder(bundle.vae)
    images = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE).clamp(-1, 1)
    features = encoder(images)
    assert features.shape[0] == 3 and features.ndim == 2
    assert features.device.type == "cpu"
    return f"VAE encoder returns {tuple(features.shape)} features"


def check_entropy_monotone():
    low = gaussian_entropy(torch.full((1, 16), 1e-3))
    high = gaussian_entropy(torch.full((1, 16), 1e-1))
    assert float(high) > float(low), "entropy did not increase with variance"
    return f"entropy rises {float(low):.2f} -> {float(high):.2f} with variance"


def check_validation_metrics():
    assert abs(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) - 1.0) < 1e-9
    assert abs(auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) - 0.0) < 1e-9
    assert abs(auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) - 0.5) < 1e-9, "ties mishandled"
    assert abs(spearman_corr([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman_corr([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9

    # A perfectly informative score: the least uncertain samples are the best.
    curve = selective_curve([0.1, 0.2, 0.3, 0.4], [1.0, 0.8, 0.6, 0.4])
    assert curve.value[0] > curve.value[-1], "selective curve should decay"
    assert abs(curve.baseline - 0.7) < 1e-9

    report = evaluate_uncertainty(
        [0.1, 0.5, 0.9], name="test", error=[0.1, 0.5, 0.9], corrupted_labels=[0, 0, 1]
    )
    assert report.spearman is not None and report.spearman > 0.99
    assert report.aurc is not None
    return f"metrics agree with hand-computed values ({report.summary()})"


CHECKS = [
    ("parameter swapping", lambda b, s: check_use_parameters(b)),
    ("point-mass posterior", check_map_is_zero_uncertainty),
    ("spread posterior", check_spread_posterior_is_positive),
    ("fixed-noise determinism", check_fixed_noise_determinism),
    ("LoRA injection", lambda b, s: check_lora_is_output_preserving(b)),
    ("Laplace fit", lambda b, s: check_laplace_fit(b)),
    ("variational LoRA fit", lambda b, s: check_lora_fit(b)),
    ("decomposition math", lambda b, s: check_decomposition_math()),
    ("epistemic/aleatoric split", check_decomposition),
    ("timestep profile", check_profile),
    ("VAE encoder", lambda b, s: check_vae_encoder(b)),
    ("entropy", lambda b, s: check_entropy_monotone()),
    ("validation metrics", lambda b, s: check_validation_metrics()),
]


def main() -> int:
    torch.set_grad_enabled(False)  # mirrors the notebooks' ambient setting
    bundle = build_bundle()
    sampler = FixedNoiseSampler(bundle, GuidanceConfig(cfg_scale=2.0))

    failures = 0
    for name, check in CHECKS:
        try:
            detail = check(bundle, sampler)
            print(f"  PASS  {name}: {detail}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")
            traceback.print_exc()

    total = len(CHECKS)
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
