# Dense-k rerun + photo panels — how to run (A100)

Two follow-ups the reviewer asked for after the tier results:
1. **See it on photos** — the actual ensemble images, not just the LPIPS/CLIP numbers.
2. **Denser k** — fill in the spread-vs-k curve (tier run only had k ∈ {1,2,5}).

Both come out of one GPU run (`exp1_densek.py`) plus two CPU plotting steps.
Scope: **15 classes** (5 per difficulty tier), dense k = **{1,2,3,4,5,7,10,15,20,30}**,
checkpoints t ∈ {0.3, 0.5, 0.7}, ensemble M=8. Expect ~3 h on the 4070 baseline;
faster on the A100.

## Prerequisites (all gitignored — must be present on the server)
- `DiT/` source (auto-downloaded by the script if missing) and the checkpoint
  `DiT/DiT-XL-2-256x256.pt`:
  ```bash
  wget https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt -P DiT/
  ```
- Python deps: `torch torchvision diffusers transformers timm lpips pandas matplotlib pillow numpy`
- Cached trajectories are **optional**: if `outputs/exp1_tiers/trajectories/latents_class*.pt`
  are present they are reused; if not, `exp1_densek.py` regenerates them from the
  same seed (bit-identical). So a fresh clone with only the checkpoint works.

## 1. Dense-k run (GPU, ~3 h)
```bash
cd experiments/amir
python exp1_densek.py --outdir outputs/exp1_densek
```
Writes `outputs/exp1_densek/exp1_densek.csv` (450 rows) and per-(class,t,k)
grids under `outputs/exp1_densek/grids/`.

Quick smoke test first (a couple of minutes):
```bash
python exp1_densek.py --classes 63 974 13 --ks 1 5 30 --t-fracs 0.3 \
    --outdir outputs/exp1_densek_smoke
```

## 2. Dense-k curves (CPU, seconds)
```bash
python plot_densek.py            # reads outputs/exp1_densek/exp1_densek.csv
```
Writes into `exp1_tiers/images/`:
- `densek_spread_vs_k.png` — spread vs k (log x), by tier, averaged over t.
- `densek_spread_vs_k_per_t.png` — same, one panel per checkpoint.

## 3. Photo montages (CPU, seconds)
```bash
python build_photo_panels.py     # reads the dense-run grids (and tier grids if present)
```
Writes into `exp1_tiers/images/`:
- `photo_k_divergence.png` — one class per tier, rows k=1/2/5, ref + 8 members.
- `photo_tier_compare.png` — the three tiers at t=0.3, k=5, side by side.
- `photo_densek_{hardest,medium,easiest}.png` — full dense-k strip per tier.

Representative classes default to hardest=63, medium=974, easiest=13
(`--reps H M E` to change). All k the panels need (1,2,5 included) are produced
by the dense run, so these work even without the original tier grids.

## Notes
- k=1,2,5 in the dense CSV reproduce the tier-run numbers exactly (same seeds),
  so the new curve is continuous with the old three points — no separate merge.
- `latent MSE` and `pairwise LPIPS` are the cleanest panels; CLIP/DINO are noisier.
