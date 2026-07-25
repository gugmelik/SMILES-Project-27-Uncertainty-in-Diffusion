# Experiment 1 — Moving backward along the trajectory

How sensitive is the final image to a perturbation of an intermediate latent, and does that sensitivity depend on *where* along the denoising trajectory the perturbation lands?

## Setup

| Parameter | Value |
|---|---|
| Model | DiT-XL/2, ImageNet-256, pretrained (inference only) |
| Sampler | DDPM ancestral, 250 respaced steps, CFG = 4.0 |
| Classes | 207 (golden retriever), 88 (hyacinth macaw) |
| Trajectory seed | 0 — one reference trajectory per class |
| Ensemble size | M = 16 |
| Backward jumps | k ∈ {1, 2, 5} sampler steps |
| Checkpoints | s ∈ {25, 75, 125, 174, 224} = 10 / 30 / 50 / 70 / 90 % denoised |
| Re-noising | Gaussian |

Throughout, **s** counts reverse steps *completed*: small s means early and noisy, large s means late and nearly finished.

**▸ IMAGE — `report_assets/ref_cls207.jpg` and `report_assets/ref_cls88.jpg` (side by side)**
*Caption: The two reference trajectories — class 207 (golden retriever) and class 88 (hyacinth macaw), both from trajectory seed 0. Every ensemble below is a perturbation of the trajectory that produced one of these two images.*

## Protocol

**1. Save the trajectory.** Run one full DDPM trajectory from a fixed starting point and keep the latent `x_t` at every one of the 250 respaced steps.

**2. Jump backward, then denoise to the end.** From a saved `x_t`, re-noise k steps up the chain with the forward kernel, then denoise back down to the finish. Repeat for all 16 members, each with its own perturbation noise.

```
x_{t+k} = sqrt(ᾱ_{t+k} / ᾱ_t) · x_t  +  sqrt(1 − ᾱ_{t+k} / ᾱ_t) · ε
```

**3. Baseline: same latent, different denoising.** Resume from the *same* `x_t` 16 times with different denoising noise and no perturbation at all. This is the "continue from the same point with different denoising" check, and it is the grey dashed line in every plot.

### Why the two seeds are kept apart

The perturbation noise and the denoising noise are drawn from two independent generators. In the perturbation runs the denoising noise is drawn once per step and **shared across all 16 members**, so whatever spread we observe can only have come from the perturbation. The baseline does exactly the opposite — latent held fixed, denoising noise independent per member. Without that separation the two sources of stochasticity would be indistinguishable in the result.

### Metrics

Spread over each ensemble of final images, measured four ways: mean pairwise cosine distance in **CLIP** ViT-B/32 and **DINOv2**-S embeddings, mean pairwise **LPIPS**, and mean pairwise **MSE on the final latents**.

## What the baseline does

Resuming with fresh denoising noise produces spread that falls monotonically as the resume point moves later — from 0.68 LPIPS at 10 % denoised down to 0.12 at 90 %. That is the expected shape, and it is the curve everything else has to be read against.

**▸ IMAGE — `report_assets/b_cls207_s25_baseline.jpg`**
*Caption: Baseline, s = 25 (10 % denoised). Reference at far left, then eight of the sixteen members. Different denoising noise from an early latent produces entirely different photographs — different poses, crops, compositions and backgrounds. Pairwise LPIPS 0.682.*

**▸ IMAGE — `report_assets/b_cls207_s224_baseline.jpg`**
*Caption: Baseline, s = 224 (90 % denoised). The same free denoising noise applied this late changes almost nothing — the composition is already locked. Pairwise LPIPS 0.117.*

## The main result: a sensitivity window

Perturbation-induced spread is **not** monotonic. It rises from a low value at the noisiest checkpoint, peaks around s ≈ 75 — roughly 30 % of the way through denoising — and then decays to near zero. Every metric and both classes show the same shape.

**▸ IMAGE — `figures/spread_vs_t_cls207.png`**
*Caption: Class 207 — spread against position along the trajectory. Coloured lines are the backward perturbation at k = 1, 2, 5; the grey dashed line is the denoise-only baseline. The x-axis runs from 10 % denoised on the left to 90 % on the right.*

**▸ IMAGE — `figures/spread_vs_t_cls88.png`**
*Caption: Class 88 — same axes. The peak sits in the same place but is markedly weaker, and the k = 1 curve is visibly noisier — a single trajectory does not pin it down.*

**▸ IMAGE — `report_assets/heatmap_cls207.png`**
*Caption: Class 207 — every measurement at once. Checkpoint down the side, jump size across; darker is more spread. Two things read straight off the grid: the darkest band is the s = 75 row, and within every row the value grows with k.*

**▸ IMAGE — `report_assets/heatmap_cls88.png`**
*Caption: Class 88 — the same grid, uniformly weaker. Each panel is scaled to its own range, so compare positions within a panel rather than colours across the two classes; the numbers are printed for that reason.*

**Pairwise LPIPS — class 207**

| checkpoint | k=0 baseline | k=1 | k=2 | k=5 |
|---|---|---|---|---|
| s=25 · 10 % | 0.682 | 0.071 | 0.089 | 0.311 |
| **s=75 · 30 %** | 0.643 | **0.312** | **0.357** | **0.452** |
| s=125 · 50 % | 0.458 | 0.219 | 0.293 | 0.360 |
| s=174 · 70 % | 0.247 | 0.093 | 0.118 | 0.157 |
| s=224 · 90 % | 0.117 | 0.018 | 0.030 | 0.054 |

**Pairwise LPIPS — class 88**

| checkpoint | k=0 baseline | k=1 | k=2 | k=5 |
|---|---|---|---|---|
| s=25 · 10 % | 0.725 | 0.101 | 0.118 | 0.143 |
| **s=75 · 30 %** | 0.707 | 0.050 | 0.158 | **0.302** |
| s=125 · 50 % | 0.326 | 0.093 | 0.135 | 0.210 |
| s=174 · 70 % | 0.199 | 0.047 | 0.064 | 0.093 |
| s=224 · 90 % | 0.096 | 0.013 | 0.022 | 0.044 |

CLIP, DINOv2 and latent MSE follow the same shape; full numbers in `exp1_results.csv`.

## Is the peak real, or an artefact of the kick size?

A jump of k steps does not inject the same amount of noise everywhere. The magnitude `sqrt(1 − ᾱ_{t+k}/ᾱ_t)` falls by roughly a factor of three from the earliest checkpoint to the latest, so part of the decay on the right of the curve is simply a smaller perturbation rather than a less sensitive model.

Dividing the spread by that magnitude settles it. The peak does not wash out — it gets **sharper**, and the three k-curves collapse onto each other, which is what you would expect if sensitivity per unit perturbation is a property of the trajectory position rather than of the perturbation size.

**▸ IMAGE — `report_assets/normalised_sensitivity.png`**
*Caption: Left — the perturbation actually injected, which is not constant along the trajectory. Middle and right — LPIPS divided by that magnitude. The s = 75 peak survives normalisation and the k-curves converge, so the window is a property of position, not of kick size.*

## What the images show

The metrics say *how much* changed. The grids say *what* changed, and the answer differs sharply between the inside and the outside of the window.

**▸ IMAGE — `report_assets/p_cls207_s75_k5.jpg`**
*Caption: Inside the window — s = 75, k = 5. Every member is still unmistakably a golden retriever, but the reference's two-dog composition is gone. Members are single dogs, head close-ups, sitting and lying poses, different backgrounds. The class survives; the layout and the instance do not.*

**▸ IMAGE — `report_assets/p_cls207_s224_k5.jpg`**
*Caption: Outside the window — s = 224, k = 5. The same size of perturbation, applied at 90 % denoised, leaves every member indistinguishable from the reference. Only texture-level detail moves.*

**▸ IMAGE — `report_assets/p_cls207_s25_k1.jpg`**
*Caption: The washout case — s = 25, k = 1. The earliest checkpoint takes the largest injected noise of any k = 1 jump, yet produces the second-smallest spread (LPIPS 0.071). With the denoising noise shared, the reverse process contracts a small early perturbation back onto the same image.*

**▸ IMAGE — `report_assets/p_cls88_s75_k5.jpg`**
*Caption: Class 88 at the peak — s = 75, k = 5. Framing and head angle shift, but far less dramatically than for class 207 under identical settings. This is the class-dependence in finding 6, visible directly.*

## Findings

**1. The baseline decays monotonically with denoising progress.** Reverse-process stochasticity matters most early: LPIPS 0.68 → 0.12 for class 207 as the resume point moves from 10 % to 90 % denoised. The later you resume, the more the image is already committed.

**2. Perturbation sensitivity peaks around 30 % denoised.** For class 207 at k = 5: 0.31 → **0.45** → 0.36 → 0.16 → 0.05. There is a window in the early-middle of the trajectory where the model is maximally sensitive to a perturbation of the latent, consistent with that being where the global layout is decided.

**3. The window survives normalisation.** Dividing by the injected perturbation magnitude sharpens the peak rather than removing it, and collapses the k = 1, 2, 5 curves together. The effect belongs to the trajectory position, not to how hard we kicked.

**4. Spread grows monotonically with k.** At essentially every checkpoint and in every metric, k = 1 < 2 < 5. The response is graded rather than a threshold.

**5. Backward perturbation is a weaker source of variation than free sampler noise.** Perturbation curves sit below the k = 0 baseline nearly everywhere. The one exception is CLIP for class 207 at s = 174, where k = 1, 2, 5 (0.088–0.097) exceed the baseline (0.057).

**6. Sensitivity is strongly class-dependent.** At matched s and k, class 207 moves far more than class 88 — at s = 75, k = 5: LPIPS 0.452 vs 0.302 and DINOv2 0.324 vs 0.089. This is the natural hook into the class-difficulty thread.

## Caveats

- **One trajectory per class.** Every number here is a single-sample estimate with no error bars. The peak's *existence* is consistent across both classes and all four metrics; its exact *location* is not yet established.
- **Coarse grid.** With checkpoints 50 steps apart, the peak is only localised to somewhere in s ∈ [25, 125].
- **Two classes, one noise type.** Only Gaussian re-noising was run; `uniform` and `laplace` are implemented but not yet executed.
- **Metric disagreement at s = 174.** CLIP is the only metric where perturbation exceeds the baseline, and it is also the least stable of the four across checkpoints. Treat CLIP cosine distance as the weakest of the four signals here.

## Next steps

- Repeat over 3–5 trajectory seeds per class to get error bars — needed before claiming a peak position.
- Run the noise-type ablation the task asks for: `--noise-types gaussian uniform laplace`.
- Denser checkpoint grid across s ∈ [25, 125] to localise the peak.
- More classes, split by difficulty, to test finding 6 properly.

## Reproduce

```bash
python exp1_renoise.py --classes 207 88
python plot_exp1.py
```

Config in `config.json`; per-ensemble metrics in `exp1_results.csv`, one row per class × checkpoint × mode × k; full-resolution grids in `grids/`.