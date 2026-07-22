
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import tiers as T

DEV_METRICS = [("lpips_dev", "LPIPS to ref"),
               ("clip_dev", "CLIP dist. to ref"),
               ("dino_dev", "DINOv2 dist. to ref"),
               ("latmse_dev", "latent MSE to ref")]
SPREAD_METRICS = [("lpips_pair", "pairwise LPIPS"),
                  ("clip_pair_cosdist", "CLIP cos. dist."),
                  ("dino_pair_cosdist", "DINOv2 cos. dist."),
                  ("latent_mse_pair", "latent MSE")]


def _tier_lines(ax, sub, xcol, ycol):
    for tier in T.TIER_ORDER:
        g = sub[sub["tier"] == tier].groupby(xcol)[ycol].agg(["mean", "sem"])
        if g.empty:
            continue
        ax.plot(g.index, g["mean"], marker="o", color=T.TIER_COLORS[tier], label=tier)
        ax.fill_between(g.index, g["mean"] - g["sem"].fillna(0),
                        g["mean"] + g["sem"].fillna(0),
                        color=T.TIER_COLORS[tier], alpha=0.15)


def plot_sweep(df, outdir):
    sw = df[df["mode"] == "sweep"]
    for kind in sw["kind"].unique():
        sub = sw[sw["kind"] == kind]
        # deviation panels + one equivariance panel
        cols = DEV_METRICS + [("equivariance_lpips", "equivariance residual")]
        fig, axes = plt.subplots(1, len(cols), figsize=(3.9 * len(cols), 3.4))
        for ax, (col, label) in zip(axes, cols):
            _tier_lines(ax, sub, "param", col)
            ax.set_xlabel(f"{kind} parameter")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("tier mean ± sem")
        axes[-1].legend(fontsize=9, title="difficulty")
        fig.suptitle(f"Geometric sweep — {kind}", fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / f"sweep_{kind}.png", dpi=150)
        plt.close(fig)


def plot_ensemble(df, outdir):
    en = df[df["mode"] == "ensemble"]
    if en.empty:
        return
    for kind in en["kind"].unique():
        sub = en[en["kind"] == kind]
        fig, axes = plt.subplots(1, len(SPREAD_METRICS), figsize=(4.3 * len(SPREAD_METRICS), 3.4))
        for ax, (col, label) in zip(axes, SPREAD_METRICS):
            _tier_lines(ax, sub, "level", col)
            ax.set_xlabel(f"{kind} random level (±)")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("ensemble spread (tier mean ± sem)")
        axes[-1].legend(fontsize=9, title="difficulty")
        fig.suptitle(f"Geometric ensemble spread — {kind}", fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / f"ensemble_{kind}.png", dpi=150)
        plt.close(fig)


def plot_transform_summary(df, outdir):
    sw = df[df["mode"] == "sweep"].copy()
    peak = (sw.sort_values("magnitude").groupby(["kind", "tier", "class_id"])
              .tail(1))
    agg = peak.groupby(["kind", "tier"])["lpips_dev"].mean().unstack("tier")
    agg = agg.reindex(columns=[t for t in T.TIER_ORDER if t in agg.columns])
    ax = agg.plot(kind="bar", figsize=(9, 4),
                  color=[T.TIER_COLORS[t] for t in agg.columns])
    ax.set_ylabel("LPIPS to ref at max magnitude")
    ax.set_xlabel("transform")
    ax.set_title("Which geometric move changes the image most?")
    ax.legend(title="difficulty")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "transform_summary.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str,
                    default=str(Path(__file__).resolve().parent / "outputs" / "geom"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    df = pd.read_csv(outdir / "geom_results.csv")
    if "tier" not in df or df["tier"].isna().all():
        df["tier"] = df["class_id"].map(T.tier_of)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    plot_sweep(df, figdir)
    plot_ensemble(df, figdir)
    plot_transform_summary(df, figdir)
    print(f"geometric figures -> {figdir}")


if __name__ == "__main__":
    main()