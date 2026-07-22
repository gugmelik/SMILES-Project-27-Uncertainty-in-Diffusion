# Experiment 1 — Moving backward along the trajectory (renoise-k)

Uncertainty of DiT-XL/2 via sensitivity of the final image to perturbations of an
intermediate latent, following the project protocol. Sized for a server GPU
(A100/H100-class); see "Small GPU" below for a laptop-scale check.

## Protocol

1. **Reference trajectory.** One DDPM trajectory (250 respaced steps, CFG=4.0)
   from a fixed starting point (`--traj-seed`); the noised latent `x_t` is saved
   at every step.
2. **Renoise-k perturbation.** From a saved `x_t` we jump `k ∈ {1, 2, 5}`
   sampler steps backward along the chain with the forward kernel

   `x_{t+k} = sqrt(ᾱ_{t+k}/ᾱ_t) · x_t + sqrt(1 − ᾱ_{t+k}/ᾱ_t) · ε`,

   then denoise back to the end. The ensemble has `M = 16` members
   (`--ensemble`), each with its **own perturbation seed** for `ε`.
3. **Separation of the two noise sources.** The perturbation seed and the
   denoising seed come from **separate `torch.Generator`s**. In the main
   (`perturb`) mode the denoising noise is *identical for all ensemble members*
   (drawn once per step and broadcast), so the spread of the ensemble is caused
   by the perturbation alone.
4. **Baseline (`denoise` mode, k=0).** The same `x_t` is continued `M` times
   with *different* denoising noise — the "continue from the same point with
   different denoising" check the proposal asks a plot for. Dashed lines in the
   figures.
5. **Noise families.** `--noise-types gaussian uniform laplace` renoises with a
   different unit-variance family while denoising stays fixed Gaussian.

## Spread metrics (per ensemble)

| column | meaning |
|---|---|
| `clip_pair_cosdist`, `dino_pair_cosdist` | mean pairwise cosine distance in CLIP ViT-B/32 / DINOv2-S embeddings |
| `clip_var`, `dino_var` | mean squared distance to the embedding centroid |
| `lpips_pair` | mean pairwise LPIPS (AlexNet) |
| `latent_mse_pair` | mean pairwise MSE between final VAE latents |
| `pixel_mse_pair` | mean pairwise MSE in pixel space |
| `*_to_ref` | mean distance of members to the unperturbed reference image |

Saved-latent positions: `--t-fracs 0.1 0.3 0.5 0.7 0.9` = fraction of denoising
completed (0.1 = early / high noise, 0.9 = late / low noise).

## Run on the server

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r experiments/amir/requirements.txt

# main run — DiT checkpoint (~2.7 GB), VAE, CLIP, DINOv2, LPIPS weights are
# downloaded automatically on first use
python experiments/amir/exp1_renoise.py --classes 207 88

# noise-type ablation
python experiments/amir/exp1_renoise.py --classes 207 \
    --noise-types gaussian uniform laplace \
    --outdir experiments/amir/outputs/exp1_noisetypes

# figures
python experiments/amir/plot_exp1.py
```

Detached run: `nohup python experiments/amir/exp1_renoise.py > exp1.log 2>&1 &`
(progress lines are flushed, and `exp1_results.csv` is rewritten after every
trajectory position, so partial results survive an interruption).

Rough cost per class at defaults: 250-step trajectory + 4 ensembles × 5
positions × ~half-length denoising at batch 32 ≈ 2.7k model steps — on the
order of 15–25 min on an A100.

### Small GPU

```bash
python experiments/amir/exp1_renoise.py --classes 207 --num-steps 50 --ensemble 8
```

## Outputs (`experiments/amir/outputs/exp1/`)

- `exp1_results.csv` — one row per (class, t, mode, k, noise type)
- `grids/*.png` — reference image (leftmost) + the M ensemble members, for eyeballing
- `trajectories/` — reference images and saved latents per class
- `figures/` — spread vs k, spread vs t (with the k=0 baseline), noise-family comparison

---

# Scaling the backward run across difficulty tiers

`tiers.py` holds the 45 ImageNet classes split into **hardest / medium / easiest**
by per-class KID (from `images_classes/kid_difficulty_tiers.png`), plus approximate
KID values for the difficulty axis. To run Experiment 1 over all of them and
aggregate per tier:

```bash
# lighter config for the sweep: M=8, three checkpoints. ~4x cheaper per class.
python experiments/amir/exp1_renoise.py \
    --classes 63 38 34 47 51 33 979 61 21 44 36 35 67 29 65 \
              10 9 91 71 49 16 57 360 974 41 83 48 26 94 69 \
              15 92 19 14 37 85 25 95 13 82 88 24 22 72 90 \
    --ensemble 8 --t-fracs 0.3 0.5 0.7 \
    --outdir experiments/amir/outputs/exp1_tiers

# per-class + per-tier figures, and the uncertainty-vs-KID scatter
python experiments/amir/plot_tiers.py --results experiments/amir/outputs/exp1_tiers/exp1_results.csv
```

`plot_tiers.py` maps each `class_id` back to its tier and produces:

- `tier_spread_vs_t_k*.png` — spread vs trajectory position, one line per tier (± sem across classes)
- `tier_spread_vs_k.png` — spread vs jump size, per tier
- `uncertainty_vs_kid.png` — per-class spread at the most sensitive checkpoint against KID, with a Spearman ρ; this is the direct test of "does backward-perturbation uncertainty track class difficulty?"

The full 45-class run at these settings is on the order of a few hours on an A100;
the CSV is rewritten after every class, so it is safe to stop early and plot what
finished.

---

# Geometric perturbation of the latent (`exp_geom.py`)

Instead of noise, apply a controlled **geometric** transform to a saved latent —
rotation, translation, scaling, horizontal flip, shear — then denoise and see how
the image responds. Two modes (both run by default):

- **sweep** — deterministic magnitude sweep; measures each final's *deviation from
  the untransformed reference* (CLIP/DINOv2/LPIPS/latent-MSE) and an
  *equivariance residual* (`LPIPS(final(T(x_t)), T(final(x_t)))` — is a latent-space
  move the same as an image-space move?).
- **ensemble** — random transform parameters per member, shared denoising noise;
  measures pairwise spread, exactly as in Experiment 1.

```bash
# 3 classes per tier, both modes, all five transforms
python experiments/amir/exp_geom.py --classes-per-tier 3
# or the full 45 classes
python experiments/amir/exp_geom.py --all-classes
# figures
python experiments/amir/plot_geom.py
```

Outputs in `experiments/amir/outputs/geom/`:

- `geom_results.csv` — one row per (class, tier, checkpoint, transform, magnitude/level, mode)
- `grids/` — `sweep_*` strips (reference + each magnitude) and `ens_*` strips (reference + random ensemble)
- `figures/` — `sweep_<transform>.png`, `ensemble_<transform>.png`, `transform_summary.png`

`exp_geom.py` imports its model/sampling/metric code from `exp1_renoise.py`, so
**keep both files (and `tiers.py`) in the same folder**.

## Implementation notes

- The stock DiT sampler draws noise from the **global** RNG, which cannot
  separate perturbation from denoising stochasticity; sampling is therefore
  re-implemented in `denoise_from()` on top of `diffusion.p_mean_variance` with
  explicit generators. With `shared_noise=True` the per-step noise is shape
  `[1, ...]` broadcast over the batch.
- Step indices are in the **respaced** schedule, so `k` counts *sampler* steps
  (at 250 respaced of 1000 training steps, `k=1` ≈ 4 original DDPM steps). The
  `ᾱ` ratios use the respaced `alphas_cumprod`, consistent with the trajectory.
- With CFG the batch is duplicated (`cat([x, x])`); `forward_with_cfg` ignores
  the second half, which only rides along — final samples are the first half.
- The DiT checkpoint resolves as: `--ckpt` → copy vendored in `DiT/` →
  auto-download to `./pretrained_models/`.
