# Pre-defence deck — what to add

Current deck has 6 slides. Below: the 7 that are missing, plus fixes to the 2
existing result slides. Slide text is in English (to match the deck); the
rationale for each is in the commit history / the review notes.

Proposed final order — **new** slides marked:

| # | Slide | Status |
|---|-------|--------|
| 1 | Title | keep |
| 2 | Background & Motivation | keep |
| 3 | **Related Work & Novelty** | NEW |
| 4 | Project Goal & Objectives | keep |
| 5 | **Method — the three proxies** | NEW |
| 6 | **Experimental setup** | NEW |
| 7 | Seed-ensemble uncertainty | keep + fix axis |
| 8 | Attention entropy | keep + add ln(256) line |
| 9 | **Routing variance — the missing proxy** | NEW |
| 10 | **Task 1.4 — 50-class panel** | NEW (`fig1`) |
| 11 | **Timestep profiles by difficulty / lemon vs crane** | NEW (`fig2`, `fig4`) |
| 12 | **Spatial per-patch heatmaps** | NEW (`fig5`) |
| 13 | **Key finding** | NEW |
| 14 | Conclusion & Roadmap | rewrite |
| 15 | **Contributions** | NEW |

---

## Slide 3 — Related Work & Novelty

> Our novelty is stated in the proposal and is currently nowhere in the deck.
> This is the slide that answers "so what's new?".

- **TREAD** (Krause et al., 2025) — token routing for efficient DiT training.
  Presented **purely as an efficiency method**; the stochastic-depth training it
  induces is never analysed. **We repurpose its routing as an inference-time
  epistemic mechanism — we are the first to use it as a UQ substrate.**
- **Generative Uncertainty in Diffusion** (Jazbec et al., 2025) — Bayesian UQ via
  last-layer Laplace, for **U-Net / flow matching**. No DiT, no per-timestep
  profile, aleatoric explicitly left as future work.
- **HyperDM** (Chan et al., NeurIPS 2024) — epistemic + aleatoric via a
  hyper-network ensemble. Rigorous but **computationally heavy**; no
  timestep-resolved analysis.
- **Timestep-Aware Block Masking** (2025) — different DiT blocks activate at
  different timesteps → supports our early-vs-late hypothesis.
- **Attention entropy** — established UQ proxy in NLP / ViT, **never applied to
  the diffusion timestep axis**.

**Gap we fill:** routing variance as a legitimate epistemic proxy + attention
entropy, analysed **per timestep**, decomposed into epistemic/aleatoric, and
linked to generation quality. Training-free.

---

## Slide 5 — Method: the three proxies

> The deck currently shows plots of quantities it never defines.

All three are **inference-time, training-free**. `D_θ` is the x̂₀-prediction.

1. **Ensemble variance** — vary the seed:
   `U_ens(t, c) = Var_k [ D_θ(x_t^(k), t, c) ]`, with `z^(1..K) ~ N(0, I)` run
   through the *full* reverse path. → **aleatoric-ish**: data noise.

2. **Routing variance** — vary the sub-network, seed **fixed**:
   `U_route(t, c) = Var_m [ D_θ^{r^(m)}(x_t, t, c) | z ]`, `r^(1..M)` random TREAD
   routes. → **epistemic**: model/sub-network ambiguity.

3. **Attention entropy** — single pass:
   `U_attn(t, ℓ) = − Σ_ij A_ij^(ℓ) log A_ij^(ℓ)`.
   We report the **mean per-query** entropy → bounded by `ln N = ln 256 = 5.55`,
   which gives the curve an interpretable ceiling (= uniform attention).

**Decomposition (law of total variance)** — the point of having *two independent*
noise sources:
`U_epi(t,c) = Var_r[ D | z ]`, `U_ale(t,c) = E_r[ Var_z[ D ] ]`.
Seeds alone cannot separate model ambiguity from data stochasticity; routes + seeds can.

---

## Slide 6 — Experimental setup

> Always asked. Currently stated nowhere.

| | |
|---|---|
| Model | **DiT-XL/2**, 675M params, 28 blocks, patch 2 → 256 tokens |
| Checkpoint | official `DiT-XL-2-256x256.pt` (ImageNet-256), **inference only, no retraining** |
| VAE | `stabilityai/sd-vae-ft-ema`, latent 32×32×4 |
| Sampler | DDPM ancestral, **250 steps** |
| Guidance | `cfg_scale = 1.0` (pure conditional — guidance inflates Var_k by a class-independent factor) |
| Seeds | **K = 16** per class |
| Classes | **50**, stratified easy / medium / hard |
| Bins | 20 timestep bins over t ∈ [0, 1000) |
| Hardware | single RTX 4070 (12 GB), ≈1 min/class |

**No ImageNet images are loaded.** Every proxy is measured on trajectories the
model generates from `z ~ N(0, I)` conditioned on a class label. "50 ImageNet
classes" = 50 label integers; the ImageNet-ness lives in the checkpoint's weights.

---

## Slide 9 — Routing variance: the proxy we could not compute

> **This slide is mandatory.** Right now the deck presents 2 of 3 proxies and
> never says the third is missing. First question from the room.

- `U_route` is the **only epistemic** proxy, and the proposal calls it *"the
  linchpin"* — the epistemic/aleatoric split in Tasks 3 and 5 is unjustified
  without it.
- It is **only legitimate on a TREAD-trained model**: routing variance means
  something because the network was *trained* as an ensemble of sub-networks.
  Injecting routes (or MC-Dropout) into vanilla DiT, which never saw them in
  training, gives uninterpretable variance — the proposal says this explicitly.
- **Blocker: no public TREAD DiT-XL/2 checkpoint.** We cannot run it, and we
  cannot substitute vanilla DiT.
- **What is ready:** a model-agnostic reference implementation
  (`routing_variance`, `route_sanity`, `epi_ale_decomposition`) validated on a
  small TREAD-trained DiT we trained ourselves. Swapping in a real TREAD
  checkpoint is a one-function change.
- **Ask:** does the curator have access to a TREAD checkpoint, or should we train
  a TREAD DiT-B/2 ourselves? This decision gates Tasks 2, 3 and 5.

---

## Slide 10 — Task 1.4: all proxies over 50 ImageNet classes

Figure: `figures/fig1_class_bin_heatmap.png`

- 50 classes × 20 timestep bins × 2 proxies (`U_route` column present but NaN).
- Stratification criterion — **expected difficulty of p(x|c)**:
  - *easy* — one canonical object, plain background, essentially unimodal (lemon, orange, golf ball…)
  - *medium* — one object with real pose / lighting / background variation (golden retriever, zebra, macaw…)
  - *hard* — polysemous labels, cluttered multi-object scenes (crane ×2, restaurant, coral reef…)
- Caveat to state out loud: the proposal asks for stratification **by expected FID
  contribution**. Per-class FID needs real ImageNet reference images; we stratify
  semantically instead and flag proper FID-based stratification as future work.
- Deliverable: `results/task1_panel50.csv`.

---

## Slide 11 — Timestep profiles by difficulty; lemon vs crane

Figures: `figures/fig2_difficulty_curves.png`, `figures/fig4_lemon_vs_crane.png`

- `U_ens` and `U_attn` per difficulty tier, mean ± 1 std over classes.
- The proposal's own key comparison: **unimodal "lemon" (951) vs the two senses of
  "crane" — bird (134) and machine (517)**. Read off the numbers from the figure
  before the defence and state whether the ambiguous class really does show a
  larger early-step spread. *(If it does not — say so. A negative result that is
  measured beats a hypothesis that is asserted.)*

---

## Slide 12 — Spatial: per-patch uncertainty

Figure: `figures/fig5_patch_heatmaps.png`

- `U_ens(t, p)` on the 16×16 DiT token grid, at early / mid / late t.
- Early: uncertainty is diffuse. Late: it concentrates on object boundaries and
  texture — the seeds have agreed *what* to draw and now disagree on *detail*.
- This is the Task-4 preview; comparison against SAM/DINO masks is next.

---

## Slide 13 — Key finding  ← THE SLIDE OF THE DECK

Measured on 50 classes × 20 bins, DiT-XL/2, K=16, 250 steps, cfg=1.0.
Two independent findings, both **negative**, both clean.

### (a) `U_ens` RISES along the denoising path — the proposal predicts the opposite

- Hypothesis in the proposal: early (high σ) = *mode selection* = **high** seed
  variance; late (low σ) = within-mode refinement = **low** variance.
- Measured: starts at **≈0.03–0.05**, dips further around **t ≈ 900**, then grows
  monotonically to **≈0.50–0.65** at t → 0. Reproduced independently by two team
  members on different code — a property of the metric, not a bug.
- **Why:** at high σ the model resolves nothing and predicts x̂₀ ≈ the conditional
  mean ("the average lemon") — all seeds agree. By the end each seed has committed
  to *its own* lemon — so they disagree maximally.

### (b) `U_ens` ANTI-correlates with class difficulty — again the opposite

| | Spearman vs difficulty tier | p |
|---|---|---|
| `U_ens` early (t≥800) | **−0.62** | <0.001 |
| `U_ens` late (t≤200) | **−0.72** | <0.001 |
| `U_attn` late | +0.05 | 0.72 (null) |

Top-6 classes by `U_ens` are **all "easy"**: tennis ball 0.720, lemon 0.706,
Granny Smith 0.658, golf ball 0.658, orange 0.655, bubble 0.636.
Bottom: valley 0.453, pizza 0.458, crane-bird 0.462, bookshop 0.468.

**The proposal's own key comparison fails in the predicted direction:**

| class | `U_ens` early | `U_ens` late |
|---|---|---|
| lemon (unimodal) | **0.0470** | 0.7065 |
| crane — bird (ambiguous) | 0.0232 | 0.4618 |
| crane — machine (ambiguous) | 0.0304 | 0.4946 |

The proposal predicts the *ambiguous* class shows the larger early-step spread.
Measured: **lemon has ~2× the early variance of either crane.**

### Why — and what it means

`Var_k[x̂₀]` in VAE latent space is dominated by **how much the class's dominant
low-frequency layout swings**, not by semantic ambiguity. A lemon is one big
saturated blob: seeds disagree about its position, size and count, and that moves
a lot of latent mass. A restaurant / valley / coral reef is generic clutter in
every sample — mid-tone texture everywhere — so seeds agree more, per patch.

Normalising each class curve by its own late value shrinks the effect
(−0.62 → **−0.48**) but does **not** remove it, so it is not purely a contrast
scale factor: the *shape* really is class-dependent.

**Consequences (this is what to say out loud):**
1. Raw latent `Var[x̂₀]` is **not a valid cross-class uncertainty proxy**. It is
   fine along the timestep axis *within* one class; comparing *between* classes it
   measures image statistics.
2. This makes `U_route` **more** necessary, not less — the epistemic proxy is the
   only one that could isolate model ambiguity from image statistics.
3. Task 5 (uncertainty → quality) will be confounded by this unless the proxy is
   normalised or moved into a semantic space (CLIP/DINO embeddings of decoded x̂₀).

**Caveat, state it honestly:** our difficulty tiers are *semantic hand-labels*, and
they are confounded with "clutter". To claim anti-correlation with difficulty
proper, we need **per-class FID** — which does require real ImageNet reference
images. That is the one place in this project where the dataset is genuinely needed.

### (c) Attention entropy carries no class information

`U_attn` is a clean function of t and of block depth — but Spearman vs difficulty
is **+0.05, p = 0.72**. Null. It decays as predicted, and block 0 sits pinned at
the `ln 256 = 5.55` ceiling (uniform attention = the early block selects nothing),
so depth is where selection happens. But it does **not** separate classes.

---

## Slide 14 — Conclusion & Roadmap (rewrite)

> Current roadmap invents its own agenda (uncertainty-adaptive sampling). Good
> idea, but the curator wrote Tasks 2–5 — the roadmap should land on them, then
> add the idea as an extension.

**Done (Task 1):** 2 of 3 proxies implemented on pretrained DiT-XL/2, training-free,
logged over 50 classes × 20 timestep bins. Key finding above.

**Blocked:** `U_route` — needs a TREAD checkpoint (see slide 9).

**Next, per the proposal:**
- **Task 2** — validate `U_route` as epistemic: OOD sensitivity, capacity trend
  across DiT-B/L/XL, non-redundancy vs `U_ens`, route-sanity ablation.
- **Task 3** — epistemic/aleatoric decomposition; find the mode-transition point t*.
- **Task 4** — spatial heatmaps vs SAM/DINO segmentation masks.
- **Task 5** — link uncertainty to per-sample quality (CLIP score, IS contribution,
  intra-class LPIPS); Spearman correlation, early vs late.

**Our extension:** uncertainty-adaptive sampling — allocate solver steps by
`U_attn`, skip low-entropy blocks. Validate against TREAD's uniform-50% baseline on FID.

---

## Slide 15 — Contributions

Gurgen Melikian — …
Pavel Podlesnyi — …
Amir Nigmatullin — …
Hajira Amjad — …
Curator: Nina Konovalova

---

## Fixes to the two existing result slides

**Slide 7 (seed-ensemble).** The x-axis says `Timestep 0 … 250`, but that is the
**sampling-step index**, and 0 = pure noise. It runs *backwards* relative to the
diffusion timestep t. As drawn it reads as if uncertainty grows with t, which is
the opposite of the truth. Relabel: `sampling step (0 = noise → 249 = image)`, or
better, plot against t descending so both slides share one axis.

**Slide 8 (attention entropy).** Add a horizontal line at `ln 256 = 5.55` (the
maximum). Then block 0 visibly sits *on* the ceiling — attention is uniform, the
early block is not selecting anything — which is the whole point of the slide and
is currently unsaid.
