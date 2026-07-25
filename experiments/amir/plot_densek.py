import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tiers as T

METRICS = [("lpips_pair", "pairwise LPIPS"),
           ("clip_pair_cosdist", "CLIP cos. dist."),
           ("dino_pair_cosdist", "DINOv2 cos. dist."),
           ("latent_mse_pair", "latent MSE")]


def _tier_curve(ax, sub, col):
    for tier in T.TIER_ORDER:
        g = sub[sub["tier"] == tier].groupby("k")[col].agg(["mean", "sem"])
        if g.empty:
            continue
        ax.plot(g.index, g["mean"], marker="o", ms=4, color=T.TIER_COLORS[tier], label=tier)
        ax.fill_between(g.index, g["mean"] - g["sem"].fillna(0),
                        g["mean"] + g["sem"].fillna(0),
                        color=T.TIER_COLORS[tier], alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("k (steps back, log)")
    ax.grid(alpha=0.3, which="both")


def plot_avg_over_t(df, outdir):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.3 * len(METRICS), 3.5))
    for ax, (col, label) in zip(axes, METRICS):
        _tier_curve(ax, df, col)
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel("ensemble spread (tier mean ± sem, over t)")
    axes[-1].legend(fontsize=9, title="difficulty")
    ks = sorted(int(k) for k in df["k"].unique())
    fig.suptitle(f"Spread vs jump size — dense k = {ks}", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "densek_spread_vs_k.png", dpi=150)
    plt.close(fig)


def plot_per_t(df, outdir, col="lpips_pair", label="pairwise LPIPS"):
    tfracs = sorted(df["t_frac"].unique())
    fig, axes = plt.subplots(1, len(tfracs), figsize=(4.3 * len(tfracs), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tfracs):
        _tier_curve(ax, df[df["t_frac"] == t], col)
        ax.set_title(f"t = {t}", fontsize=10)
    axes[0].set_ylabel(f"{label} (tier mean ± sem)")
    axes[-1].legend(fontsize=9, title="difficulty")
    fig.suptitle(f"Spread vs k by checkpoint — {label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "densek_spread_vs_k_per_t.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/exp1_densek/exp1_densek.csv")
    ap.add_argument("--outdir", default="../../exp1_tiers/images")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[df["mode"] == "perturb"].copy()
    df["tier"] = df["class_id"].map(T.tier_of)
    df = df[df["tier"].notna()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_avg_over_t(df, outdir)
    plot_per_t(df, outdir)
    print(f"dense-k figures -> {outdir}  "
          f"({df['class_id'].nunique()} classes, "
          f"k={sorted(int(k) for k in df['k'].unique())})")


if __name__ == "__main__":
    main()
