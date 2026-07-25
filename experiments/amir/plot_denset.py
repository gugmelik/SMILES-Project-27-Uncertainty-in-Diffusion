import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import tiers as T

METRICS = [("lpips_pair", "pairwise LPIPS"),
           ("clip_pair_cosdist", "CLIP cos. dist."),
           ("dino_pair_cosdist", "DINOv2 cos. dist."),
           ("latent_mse_pair", "latent MSE")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/exp1_denset/exp1_densek.csv")
    ap.add_argument("--outdir", default="../../exp1_tiers/images")
    ap.add_argument("--ks", type=int, nargs="+", default=None,
                    help="which k to plot (default: every k in the CSV, one figure each)")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[df["mode"] == "perturb"].copy()
    df["tier"] = df["class_id"].map(T.tier_of)
    df = df[df["tier"].notna()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ks = args.ks if args.ks is not None else sorted(int(k) for k in df["k"].unique())

    for k in ks:
        sub = df[df["k"] == k]
        if sub.empty:
            print(f"[skip] k={k}: no rows")
            continue
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.3 * len(METRICS), 3.4))
        for ax, (col, label) in zip(axes, METRICS):
            for tier in T.TIER_ORDER:
                g = sub[sub["tier"] == tier].groupby("t_frac")[col].agg(["mean", "sem"])
                if g.empty:
                    continue
                ax.plot(g.index, g["mean"], marker="o", color=T.TIER_COLORS[tier], label=tier)
                ax.fill_between(g.index, g["mean"] - g["sem"].fillna(0),
                                g["mean"] + g["sem"].fillna(0),
                                color=T.TIER_COLORS[tier], alpha=0.15)
            ax.set_xlabel("fraction of denoising completed")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("ensemble spread (tier mean ± sem)")
        axes[-1].legend(fontsize=9, title="difficulty")
        n_t = sub["t_frac"].nunique()
        fig.suptitle(f"Backward perturbation k={k} — spread by difficulty tier "
                     f"({n_t} checkpoints)", fontsize=12)
        fig.tight_layout()
        out = outdir / f"tier_spread_vs_t_k{k}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}  (k={k}, t={sorted(sub['t_frac'].unique())})")


if __name__ == "__main__":
    main()
