# Experimentation setup

## Goal
Measure how sensitive DiT-XL/2 generation is to a **controlled, directional** corruption of mid-trajectory latents: shift along an empirically estimated PCA axis of the latent space, then denoise to completion with a fixed reverse-step noise seed. Compare finals to an unperturbed control via CLIP / DINOv2 / LPIPS / latent MSE.

- **Notebook:** `notebooks/exp3_pca_perturbation.ipynb`
- **Results:** `results/exp3_pca_perturbation/`

## Model & sampling

| Setting | Value |
|--------|--------|
| Model | DiT-XL/2, ImageNet-256 |
| VAE | `stabilityai/sd-vae-ft-ema` |
| CFG scale | 4.0 |
| Reverse steps \(T\) | 250 |
| Latent shape | \(4 \times 32 \times 32\) |
| Reference traj seed | `traj_seed = 0` (fixed per class) |
| Shared denoise seed | `shared_denoise_seed = 12345` |
| Checkpoints | every 50 reverse steps → \(s \in \{50, 100, 150, 200\}\) |

## Class selection (KID difficulty tiers)
Same split as `results/ensemble_K50_fid/kid_difficulty_tiers.png` (per-class KID vs ImageNet val):

| Tier | Rule | Count |
|------|------|-------|
| Hardest | highest KID | 15 |
| Medium | middle of ranking | 15 |
| Easiest | lowest KID | 15 |

**Total:** 45 classes.

**Spotlights** (median-KID class per tier, for image grids): hardest **61** (boa_constrictor), medium **360** (otter), easiest **95** (jacamar).

## Protocol (per class)

1. **Reference trajectory** — one reverse path from fixed `traj_seed`; cache `traj[s]` at checkpoints. Step noises of this traj are **not** saved (only latents).
2. **PCA basis (per checkpoint)** — generate `K_pca=32` extra full trajectories (same class, independent seeds). Stack latents at checkpoint \(s\) → \(X \in \mathbb{R}^{32 \times D}\). Center + SVD → unit directions \(v_c\) and scales \(\sigma_c = \sqrt{\lambda_c}\). Fit **separately at each \(s\)**. Reference traj is **not** in the PCA fit.
3. **Control** — from each \(s\), denoise with shared reverse-step noise, no PCA shift:
   \(x_T^{\mathrm{ctrl}} = \mathrm{Denoise}(x_s,\ \varepsilon^{\mathrm{shared}}_{s:T})\).
4. **PCA perturbation** — for PC \(c \in \{0,1,2\}\) and \(\alpha \in \{-3,-1,+1,+3\}\):
   \(x_{\mathrm{pert}} = x_s + \alpha \cdot \sigma_c \cdot v_c\).
   Same timestep \(s\) (no forward/backward noise jump). Denoise with the **same** \(\varepsilon^{\mathrm{shared}}\) as control.
5. **Metrics (perturbed vs control)** — `clip_cos_dist`, `dino_cos_dist`, `lpips`, `latent_mse`, `pixel_mse`.

## Design choices
- Shared denoise seed across all \((\mathrm{PC},\alpha)\) at a given \(s\) → differences attributed to the directional shift.
- Reference traj noises are **not** reused when resuming; resume uses `shared_denoise_seed`. Control at different \(s\) ≠ continue the original traj, so control images can differ across checkpoints.
- PCA is ensemble-based (across seeds), not spatial/within-image PCA.
- Caches under `class_XXXX/`: `traj_*.pt`, `pca_ensemble_K32.pt`, `metrics_pca_perturbation.json` — finished classes skip on re-run.

## How to read grids
`grid_s{S}_pc{K}.png` columns: **(1)** control \(\alpha=0\) from checkpoint \(S\) | **(2–5)** \(\alpha = -3,-1,+1,+3\) along that PC.

Spotlight grids: **\(s=100\)** and **\(s=150\)**.

## One-line summary
At mid-denoising, push the reference latent along the top seed-variation PCA axes by \(\pm 1,\pm 3\) std, finish with fixed reverse noise, and measure how much the final image moves — across easy/medium/hard ImageNet classes.
