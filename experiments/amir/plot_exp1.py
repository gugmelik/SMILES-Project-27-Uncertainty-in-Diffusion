
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRICS = [
    ("lpips_pair", "pairwise LPIPS"),
    ("clip_pair_cosdist", "pairwise CLIP cosine distance"),
    ("dino_pair_cosdist", "pairwise DINOv2 cosine distance"),
    ("latent_mse_pair", "pairwise latent MSE"),
]


def plot_spread_vs_k(df, outdir):
    per = df[df["mode"] == "perturb"]
    for (cls, noise), sub in per.groupby(["class_id", "noise_type"]):
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.4))
        for ax, (col, label) in zip(axes, METRICS):
            for frac, g in sub.groupby("t_frac"):
                g = g.sort_values("k")
                ax.plot(g["k"], g[col], marker="o", label=f"t={frac:.1f}")
                base = df[(df["mode"] == "denoise") & (df["class_id"] == cls)
                          & (df["t_frac"] == frac)]
                if len(base):
                    ax.axhline(base[col].iloc[0], ls="--", lw=0.8,
                               color=ax.lines[-1].get_color(), alpha=0.6)
            ax.set_xlabel("k (steps re-noised backward)")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("ensemble spread")
        axes[-1].legend(fontsize=8, title="denoising progress\n(dashed = k=0 baseline)")
        fig.suptitle(f"class {cls} — spread vs perturbation size ({noise} renoise)")
        fig.tight_layout()
        fig.savefig(outdir / f"spread_vs_k_cls{cls}_{noise}.png", dpi=150)
        plt.close(fig)


def plot_spread_vs_t(df, outdir):
    for cls, sub in df[df["noise_type"] == "gaussian"].groupby("class_id"):
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.4))
        for ax, (col, label) in zip(axes, METRICS):
            base = sub[sub["mode"] == "denoise"].sort_values("t_frac")
            ax.plot(base["t_frac"], base[col], marker="s", ls="--", color="gray",
                    label="k=0 (denoise seeds only)")
            for k, g in sub[sub["mode"] == "perturb"].groupby("k"):
                g = g.sort_values("t_frac")
                ax.plot(g["t_frac"], g[col], marker="o", label=f"k={k}")
            ax.set_xlabel("fraction of denoising completed at $x_t$")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("ensemble spread")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"class {cls} — spread vs position along trajectory")
        fig.tight_layout()
        fig.savefig(outdir / f"spread_vs_t_cls{cls}.png", dpi=150)
        plt.close(fig)


def plot_noise_types(df, outdir):
    per = df[df["mode"] == "perturb"]
    if per["noise_type"].nunique() < 2:
        return
    for cls, sub in per.groupby("class_id"):
        fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.4))
        for ax, (col, label) in zip(axes, METRICS):
            for noise, g in sub.groupby("noise_type"):
                g = g.groupby("k")[col].mean().reset_index().sort_values("k")
                ax.plot(g["k"], g[col], marker="o", label=noise)
            ax.set_xlabel("k")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("spread (mean over t)")
        axes[-1].legend(fontsize=8, title="renoise type")
        fig.suptitle(f"class {cls} — renoising noise families")
        fig.tight_layout()
        fig.savefig(outdir / f"noise_types_cls{cls}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str,
                    default=str(Path(__file__).resolve().parent / "outputs" / "exp1"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    df = pd.read_csv(outdir / "exp1_results.csv")
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    plot_spread_vs_k(df, figdir)
    plot_spread_vs_t(df, figdir)
    plot_noise_types(df, figdir)
    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
