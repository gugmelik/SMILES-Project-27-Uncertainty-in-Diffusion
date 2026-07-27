# Bayesian uncertainty in DiT

Implementation notes for `src/uncertainty/`. This adds an **epistemic** uncertainty
signal to the project, complementing the existing training-free proxies
(`U_ens`, `U_attn`).

## Why the existing proxies are not epistemic

A diffusion model has two distinct sources of uncertainty:

| Source | Meaning | Captured by |
|--------|---------|-------------|
| **Aleatoric** | Variability inherent in `p(x \| c)`: the initial latent and the noise injected at each reverse step | Sampling the fixed model many times — this is what `U_ens` measures |
| **Epistemic** | Uncertainty about the denoiser's *parameters*, because training data are finite | Nothing in the repo before this change |

`U_ens` varies the seed while holding `theta` fixed at the pretrained point
estimate, so it is a (perfectly reasonable) aleatoric measure. No amount of
seed-resampling from one checkpoint reveals what the model does not know.
Getting at epistemic uncertainty requires a distribution `q(theta)` over weights
and a way to run the sampler under draws from it.

## The pipeline

```
                 fit on ImageNet latents
  DiT-XL/2  ──────────────────────────────▶  q(theta)      posteriors.py
  (frozen)     Laplace  /  variational LoRA                laplace.py, lora.py
                                                │
                        draw theta_m ───────────┤
                                                ▼
   z_T, step noise, class  ──▶  reverse process  ──▶  x_0^(m)   sampling.py
              (pinned)              (250 steps)
                                                │
                                                ▼
                          variance across m, in x_0 / pixel /
                          feature space, and the epistemic /
                          aleatoric split                      estimators.py
                                                │
                                                ▼
                          does the score predict failure?      validation.py
```

## 1. A posterior over weights

Making all 675M DiT-XL/2 parameters Bayesian is not tractable, so only a
selected subset is treated as random and the rest stays frozen at the
pretrained value. `dit.select_parameters` resolves the subset:

| Spec | Parameters | Notes |
|------|-----------|-------|
| `last_layer` | `final_layer.linear` (~37k) | BayesDiff-style; fits in minutes |
| `last_layer+adaln` | plus the final adaLN projection (~2.7M) | |
| `late_blocks:N` | `attn.proj` and `mlp.fc2` of the last N blocks | ~6.6M per block |
| `regex:<pattern>` | anything matching | escape hatch |

Two fitting methods, both leaving the backbone frozen:

**Diagonal Laplace** (`laplace.py`) — post-hoc, training-free. Accumulates the
diagonal empirical Fisher of the diffusion loss and sets
`q = N(theta_MAP, (lambda I + F)^-1)`. Gradients are taken one example at a
time, because the empirical Fisher is a sum of *per-sample* squared gradients
and a batch gradient is not a substitute. The prior precision `lambda` is tuned
by maximising the diagonal-Laplace evidence, which needs no extra forward passes
once `F` is accumulated.

**Variational LoRA** (`lora.py`) — Bayes-by-Backprop over low-rank adapters on
late blocks. Adapters start at `Delta W = 0` (the self-test asserts injection
does not change the model's output), so the posterior mean stays near the
pretrained model while the spread carries the signal. More expressive than
last-layer Laplace, which can only represent uncertainty that survives the final
projection.

**Deep ensembles** (`DeepEnsemblePosterior`) are the reference baseline. A
Laplace posterior that reports far less disagreement than an ensemble on the
same inputs is too tight to trust; `lora_deep_ensemble` freezes a set of draws
into a finite ensemble for that comparison.

## 2. Pinning the diffusion randomness

`FixedNoiseSampler` precomputes the entire `(T, N, C, H, W)` noise schedule from
a seed so the identical sequence can be replayed under each `theta_m`. Without
this, seed-to-seed variation swamps the weight disagreement and the "epistemic"
number is really just `U_ens` again.

Two details that are easy to get wrong:

- **`theta_m` is drawn once per trajectory**, not per step. Resampling weights
  at every step simulates a model that changes mid-generation, which is not a
  draw from the posterior over models.
- **Common random numbers** in the nested decomposition: the same `K` noise
  seeds are reused for every weight draw.

## 3. The three measurements

### Local (per-step) — `local_epistemic_uncertainty`

Variance of the `x_0` prediction across weight draws at a single `(x_t, t)`.
One denoiser call per draw, so it is cheap.

Measured in `x_0` space rather than `epsilon` space deliberately: the
`eps -> x_0` conversion divides by `sqrt(alpha_bar_t)`, so equal `epsilon`
disagreement means very different things at `t = 999` and `t = 1`, and an
`epsilon`-space profile mostly reflects the noise schedule.

`local_epistemic_profile` sweeps this along a reference trajectory generated
under the posterior mean, which is the per-timestep view this project is about:
because every draw sees the *same* `x_t`, it isolates local disagreement from
the drift that accumulates when each draw follows its own trajectory.

### Whole-sample — `posterior_predictive` + `trajectory_uncertainty`

One complete generation per weight draw with the diffusion noise pinned, scored
three ways: pixel variance (and its heatmap), feature variance, and the Gaussian
entropy of the predictive feature distribution.

Prefer the feature-space numbers. Pixel variance is easily fooled — a one-pixel
translation or a background change produces large pixel variance with unchanged
semantics. `features.py` provides `VAEEncoder` (no extra downloads),
`TimmEncoder` (DINOv2 through the existing `timm` dependency), `CLIPEncoder`,
and `PixelEncoder` as the baseline to beat.

### Decomposition — `decompose_uncertainty`

Nested `M` weight draws x `K` noise draws, split by the law of total variance:

```
Var[Y] = Var_theta[E_z[Y|theta]]  +  E_theta[Var_z[Y|theta]]
         \_____ epistemic ______/     \______ aleatoric ______/
```

One subtlety worth knowing: even with shared seeds, the sample variance of the
per-model means estimates `Var_theta + Var_z / K`, so the raw epistemic term is
inflated whenever `K` is small. `bias_correct=True` (the default) subtracts the
leak. When the corrected value clamps to zero, the run cannot *resolve*
epistemic uncertainty at that `K` — raise `K` rather than reporting the zero as
a finding. Both the corrected and raw values are returned.

`decompose_feature_tensor` exposes the estimator on a cached `(M, K, N, d)`
tensor, so features can be re-decomposed without regenerating images.

Cost is `M * K` full trajectories, so this is the expensive measurement.

## 4. Validating the score

A plausible-looking heatmap is not evidence. `validation.py` implements the
checks that distinguish a useful score from a pretty one: rank correlation with
per-sample error or quality, AUROC for separating corrupted generations, and
selective-generation curves (drop the most uncertain samples; does the average
quality of what remains actually improve?).

The honest description of what these numbers are is **posterior model
disagreement**, not a calibrated probability that an image is correct.
`DiagonalGaussianPosterior.scale` exists precisely because a Laplace
approximation to a heavily over-parameterised denoiser is rarely calibrated out
of the box.

## Usage

Self-test (tiny toy DiT, CPU, ~6 seconds, no checkpoint needed):

```bash
python experiments/bayesian_uncertainty/selftest.py
```

Fit a posterior:

```bash
# Post-hoc Laplace on the last layer — start here
python experiments/bayesian_uncertainty/fit_posterior.py laplace \
    --params last_layer --num-samples 512 \
    --latent-cache results/latent_cache_2048.pt \
    --out results/posteriors/laplace_last_layer.pt

# Variational LoRA on the last 4 blocks
python experiments/bayesian_uncertainty/fit_posterior.py lora \
    --targets late_blocks:4 --rank 4 --steps 2000 \
    --latent-cache results/latent_cache_2048.pt \
    --out results/posteriors/lora_late4.pt
```

Measure:

```bash
python experiments/bayesian_uncertainty/measure_uncertainty.py \
    --posterior results/posteriors/laplace_last_layer.pt \
    --classes 207 88 980 \
    --predictive --profile --decompose \
    --num-weight-samples 16 --num-noise-samples 4 \
    --encoder vae --save-images \
    --out results/bayesian_uncertainty
```

`--posterior-kind map` runs the whole thing against a point-mass posterior; every
epistemic number must come out exactly zero, which is a good end-to-end sanity
check on a new configuration.

From Python:

```python
from src.uncertainty import (
    FixedNoiseSampler, load_dit, load_posterior,
    posterior_predictive, trajectory_uncertainty,
)

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
```

## Suggested experiments

1. **Timestep profile.** Does epistemic uncertainty peak early (mode selection)
   or late (detail synthesis)? Compare against the existing `U_ens` profile —
   if they peak at the same `t`, the Bayesian machinery is adding nothing.
2. **Epistemic ratio vs class KID.** The repo already has per-class KID in
   `results/ensemble_K50_fid/`. Do high-KID classes have a higher epistemic
   share?
3. **OOD conditions.** Epistemic uncertainty should rise for rare or ambiguous
   classes; aleatoric should not necessarily.
4. **Laplace vs LoRA vs ensemble.** Does last-layer Laplace under-report
   relative to a LoRA ensemble, as the Fisher-information literature suggests?

## References

- Kristiadi et al., *BayesDiff* — last-layer Laplace for pixel-wise diffusion uncertainty
- Jazbec et al., *Generative Uncertainty in Diffusion Models* — fixed-noise, varying-weights posterior predictive; feature-space scoring
- Blundell et al., *Weight Uncertainty in Neural Networks* — Bayes by Backprop
- Lakshminarayanan et al., *Deep Ensembles* — the reference baseline
