
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


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = pd.Series(x[ok]).rank().values
    ry = pd.Series(y[ok]).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_spread_vs_t_by_tier(df, outdir):
    per = df[df["mode"] == "perturb"].copy()
    for k in sorted(per["k"].unique()):
        sub = per[per["k"] == k]
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.3 * len(METRICS), 3.4))
        for ax, (col, label) in zip(axes, METRICS):
            for tier in T.TIER_ORDER:
                g = sub[sub["tier"] == tier].groupby("t_frac")[col].agg(["mean", "sem"])
                if g.empty:
                    continue
                ax.plot(g.index, g["mean"], marker="o", color=T.TIER_COLORS[tier],
                        label=tier)
                ax.fill_between(g.index, g["mean"] - g["sem"].fillna(0),
                                g["mean"] + g["sem"].fillna(0),
                                color=T.TIER_COLORS[tier], alpha=0.15)
            ax.set_xlabel("fraction of denoising completed")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("ensemble spread (tier mean ± sem)")
        axes[-1].legend(fontsize=9, title="difficulty")
        fig.suptitle(f"Backward perturbation k={k} — spread by difficulty tier", fontsize=12)
        fig.tight_layout()
        fig.savefig(outdir / f"tier_spread_vs_t_k{k}.png", dpi=150)
        plt.close(fig)


def plot_spread_vs_k_by_tier(df, outdir):
    per = df[df["mode"] == "perturb"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.3 * len(METRICS), 3.4))
    for ax, (col, label) in zip(axes, METRICS):
        for tier in T.TIER_ORDER:
            g = per[per["tier"] == tier].groupby("k")[col].mean()
            if g.empty:
                continue
            ax.plot(g.index, g.values, marker="o", color=T.TIER_COLORS[tier], label=tier)
        ax.set_xlabel("k (steps back)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("spread (mean over t)")
    axes[-1].legend(fontsize=9, title="difficulty")
    fig.suptitle("Spread vs jump size, by difficulty tier", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "tier_spread_vs_k.png", dpi=150)
    plt.close(fig)


def plot_uncertainty_vs_kid(df, outdir):
    per = df[df["mode"] == "perturb"].copy()
    kmax = per["k"].max()
    pk = per[per["k"] == kmax]
    peak_t = pk.groupby("t_frac")["lpips_pair"].mean().idxmax()
    sel = pk[pk["t_frac"] == peak_t]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.3 * len(METRICS), 3.6))
    for ax, (col, label) in zip(axes, METRICS):
        xs, ys = [], []
        for _, r in sel.iterrows():
            kid = T.kid_of(r["class_id"])
            if kid is None or not np.isfinite(r[col]):
                continue
            ax.scatter(kid, r[col], s=34, color=T.TIER_COLORS.get(r["tier"], "#888"),
                       edgecolor="white", linewidth=0.5, zorder=3)
            xs.append(kid)
            ys.append(r[col])
        rho = _spearman(xs, ys)
        if len(xs) >= 2:
            b, a = np.polyfit(xs, ys, 1)
            xr = np.array([min(xs), max(xs)])
            ax.plot(xr, a + b * xr, color="#444", lw=1, ls="--", zorder=2)
        ax.set_xlabel("per-class KID (difficulty)")
        ax.set_title(f"{label}\nSpearman ρ = {rho:.2f}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"spread at t={peak_t:.1f}, k={kmax}")
    handles = [plt.Line2D([], [], marker="o", ls="", color=T.TIER_COLORS[t], label=t)
               for t in T.TIER_ORDER]
    axes[-1].legend(handles=handles, fontsize=9, title="tier")
    fig.suptitle("Does backward-perturbation uncertainty track class difficulty?", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "uncertainty_vs_kid.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True,
                    help="exp1_results.csv from the multi-class run")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    res = Path(args.results)
    df = pd.read_csv(res)
    df["tier"] = df["class_id"].map(T.tier_of)
    n_known = df["tier"].notna().sum()
    if n_known == 0:
        raise SystemExit("no classes in this CSV are in tiers.py — nothing to aggregate")
    df = df[df["tier"].notna()]

    outdir = Path(args.outdir) if args.outdir else res.parent / "figures_tiers"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_spread_vs_t_by_tier(df, outdir)
    plot_spread_vs_k_by_tier(df, outdir)
    plot_uncertainty_vs_kid(df, outdir)
    print(f"tier figures -> {outdir} "
          f"({df['class_id'].nunique()} classes across "
          f"{df['tier'].nunique()} tiers)")


if __name__ == "__main__":
    main()