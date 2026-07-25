"""Figures for the pre-defence deck, from results/task1_panel50.csv.

  fig1_class_bin_heatmap.png  Task 1.4 deliverable: U_ens and U_attn over
                              50 classes x 20 timestep bins, rows grouped easy/medium/hard.
  fig2_difficulty_curves.png  mean +/- 1 std timestep profile per difficulty tier.
  fig3_attn_per_block.png     attention entropy for an early / mid / late block,
                              averaged over the 50 classes, against the ln(256) ceiling.
  fig4_lemon_vs_crane.png     the proposal's key comparison: unimodal "lemon" vs the
                              two senses of "crane".
  fig5_patch_heatmaps.png     per-patch U_ens at early / mid / late t (spatial, Task 4 preview).

The x-axis everywhere is the diffusion timestep t (1000 = pure noise, 0 = clean
image) drawn right-to-left, so the reading direction matches the denoising path.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
FIGS = os.path.join(REPO, "figures")
os.makedirs(FIGS, exist_ok=True)

LN256 = np.log(256)
TIERS = ["easy", "medium", "hard"]
TIER_COLOR = {"easy": "#4C72B0", "medium": "#DD8452", "hard": "#C44E52"}

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

df = pd.read_csv(os.path.join(RES, "task1_panel50.csv"))
df["tier_rank"] = df["difficulty"].map({t: i for i, t in enumerate(TIERS)})
centres = np.sort(df["t_centre"].unique())


def _denoise_axis(ax):
    """t runs 1000 -> 0 left-to-right, i.e. noise -> image."""
    ax.set_xlim(1000, 0)
    ax.set_xlabel("diffusion timestep $t$   (1000 = noise $\\rightarrow$ 0 = image)")


# --------------------------------------------------------------- fig 1 --------
order = df.sort_values(["tier_rank", "class_id"])["class_id"].drop_duplicates().tolist()
names = {c: df[df.class_id == c].class_name.iloc[0] for c in order}
tiers = {c: df[df.class_id == c].difficulty.iloc[0] for c in order}

fig, axes = plt.subplots(1, 2, figsize=(13, 9))
for ax, proxy in zip(axes, ["U_ens", "U_attn"]):
    M = np.stack([df[df.class_id == c].sort_values("t_centre", ascending=False)[proxy].values
                  for c in order])
    Mn = (M - M.min(1, keepdims=True)) / (np.ptp(M, axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(Mn, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(0, len(centres), 3))
    ax.set_xticklabels(sorted(centres, reverse=True)[::3])
    ax.set_xlabel("diffusion timestep $t$  (noise $\\rightarrow$ image)")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{names[c]}" for c in order], fontsize=6)
    for lbl, c in zip(ax.get_yticklabels(), order):
        lbl.set_color(TIER_COLOR[tiers[c]])
    ax.set_title(f"$U_\\mathrm{{{proxy.split('_')[1]}}}$  (row-normalised)")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # tier separators
    bounds = np.cumsum([sum(1 for c in order if tiers[c] == t) for t in TIERS])[:-1]
    for b in bounds:
        ax.axhline(b - 0.5, color="w", lw=1.4)
fig.suptitle("Task 1.4 — three proxies logged over 50 ImageNet classes x 20 timestep bins\n"
             "(blue = easy, orange = medium, red = hard;  $U_\\mathrm{route}$ absent: "
             "needs a TREAD checkpoint)", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig1_class_bin_heatmap.png"), bbox_inches="tight")
plt.close()

# --------------------------------------------------------------- fig 2 --------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, proxy, lab in zip(axes, ["U_ens", "U_attn"],
                          [r"$U_\mathrm{ens}(t)$", r"$U_\mathrm{attn}(t)$"]):
    for tier in TIERS:
        sub = df[df.difficulty == tier]
        g = sub.groupby("t_centre")[proxy]
        m, s = g.mean(), g.std()
        ax.plot(m.index, m.values, color=TIER_COLOR[tier], lw=2, label=tier)
        ax.fill_between(m.index, m - s, m + s, color=TIER_COLOR[tier], alpha=0.15)
    _denoise_axis(ax)
    ax.set_ylabel(lab)
    ax.legend(title="difficulty", fontsize=8)
axes[1].axhline(LN256, color="grey", ls="--", lw=1)
axes[1].text(980, LN256 - 0.06, f"max $\\ln 256$ = {LN256:.2f}", fontsize=7, color="grey")
axes[0].set_title("Seed-ensemble variance rises as the path is denoised")
axes[1].set_title("Attention entropy falls as the path is denoised")
fig.suptitle("Timestep profiles by class difficulty (mean $\\pm$ 1 std over classes)",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig2_difficulty_curves.png"), bbox_inches="tight")
plt.close()

# --------------------------------------------------------------- fig 3 --------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, col, title in zip(
    axes,
    ["U_attn_block0", "U_attn_block14", "U_attn_block27"],
    ["block 0 (early)", "block 14 (mid)", "block 27 (late)"],
):
    g = df.groupby("t_centre")[col]
    m, s = g.mean(), g.std()
    ax.plot(m.index, m.values, color="#8172B3", lw=2)
    ax.fill_between(m.index, m - s, m + s, color="#8172B3", alpha=0.2)
    ax.axhline(LN256, color="grey", ls="--", lw=1)
    _denoise_axis(ax)
    ax.set_title(title)
axes[0].set_ylabel(r"$U_\mathrm{attn}(t,\ell)$  [nats]")
axes[0].text(980, LN256 - 0.08, f"max $\\ln 256$ = {LN256:.2f}", fontsize=7, color="grey")
fig.suptitle("Attention entropy per DiT block, averaged over the 50 classes\n"
             "early block sits at the uniform-attention ceiling; depth is where selection happens",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig3_attn_per_block.png"), bbox_inches="tight")
plt.close()

# --------------------------------------------------------------- fig 4 --------
key = [(951, "lemon — unimodal", "#4C72B0"),
       (134, "crane (bird)", "#C44E52"),
       (517, "crane (machine)", "#DD8452")]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for cid, lab, col in key:
    sub = df[df.class_id == cid].sort_values("t_centre")
    if sub.empty:
        continue
    axes[0].plot(sub.t_centre, sub.U_ens, color=col, lw=2, marker="o", ms=3, label=lab)
    axes[1].plot(sub.t_centre, sub.U_attn, color=col, lw=2, marker="o", ms=3, label=lab)
for ax, lab in zip(axes, [r"$U_\mathrm{ens}(t)$", r"$U_\mathrm{attn}(t)$"]):
    _denoise_axis(ax)
    ax.set_ylabel(lab)
    ax.legend(fontsize=8)
axes[1].axhline(LN256, color="grey", ls="--", lw=1)
fig.suptitle("Key comparison from the proposal: unimodal 'lemon' vs the two senses of 'crane'",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig4_lemon_vs_crane.png"), bbox_inches="tight")
plt.close()

# --------------------------------------------------------------- fig 5 --------
npz = np.load(os.path.join(RES, "task1_panel50_patchmaps.npz"))
show = [c for c in (951, 207, 134, 762) if str(c) in npz]
fig, axes = plt.subplots(len(show), 3, figsize=(7.5, 2.4 * len(show)))
axes = np.atleast_2d(axes)
for r, cid in enumerate(show):
    maps, tvals = npz[str(cid)], npz[f"{cid}_t"]
    nm = df[df.class_id == cid].class_name.iloc[0]
    for c in range(3):
        ax = axes[r, c]
        im = ax.imshow(maps[c], cmap="magma")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if r == 0:
            ax.set_title(["early ($t\\approx$ %d)", "mid ($t\\approx$ %d)",
                          "late ($t\\approx$ %d)"][c] % tvals[c], fontsize=9)
        if c == 0:
            ax.set_ylabel(f"{nm}\n({cid})", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Per-patch $U_\\mathrm{ens}(t,p)$ on the 16x16 DiT token grid\n"
             "where the seeds disagree, spatially", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig5_patch_heatmaps.png"), bbox_inches="tight")
plt.close()

# --------------------------------------------------------------- summary ------
early = df[df.t_centre >= 800].groupby("difficulty")[["U_ens", "U_attn"]].mean()
late = df[df.t_centre <= 200].groupby("difficulty")[["U_ens", "U_attn"]].mean()
summary = pd.concat({"early (t>=800, noise)": early, "late (t<=200, image)": late}, axis=1).round(4)
summary.to_csv(os.path.join(RES, "task1_early_late_by_difficulty.csv"))
print(summary.to_string())
print(f"\nfigures -> {FIGS}")
