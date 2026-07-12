"""Small plotting helpers shared across the notebooks (consistent style)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# a colour-blind-safe, consistent palette for the three proxies
PROXY_COLORS = {"U_ens": "#4C72B0", "U_route": "#DD8452",
                "U_attn": "#55A868", "U_epi": "#C44E52", "U_ale": "#8172B3"}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "image.cmap": "magma",
    })


def show_grid(imgs, titles=None, ncols=8, size=1.4, cmap=None, suptitle=None):
    """imgs: (B,H,W,3) in [0,1] or (B,H,W) scalar maps."""
    n = len(imgs)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i], cmap=cmap)
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def normalize_path_axis(t):
    """Map recorded timesteps (high->low) to a 0..1 'denoising progress' axis."""
    t = np.asarray(t, dtype=float)
    return (t.max() - t) / (t.max() - t.min() + 1e-9)
