"""Synthetic, class-labelled image dataset for the Task-1 demo.

Why synthetic instead of ImageNet-256?
--------------------------------------
The real project targets DiT-XL/2 + TREAD on ImageNet-256, which needs a 2.7 GB
checkpoint and a multi-GPU eval setup.  To make Task 1 *runnable and inspectable*
end-to-end, we generate a small labelled dataset in pixel space (32x32 RGB) whose
class structure is designed to exercise exactly the phenomena the proxies target:

  * EASY classes      -> one shape, fixed colour, centred  (low uncertainty).
  * MEDIUM / HARD     -> position/size/colour jitter, busy background.
  * AMBIGUOUS classes -> the label maps to TWO possible shapes (a bimodal
    conditional p(x|c)).  This is the toy analogue of the proposal's "crane"
    (bird vs machine): the model must *commit to a mode* early -> high early-step
    epistemic uncertainty.

Every generator function here is deterministic given a seed, so the dataset,
plots and metrics are reproducible.  The same notebooks run unchanged against the
real DiT-XL/2 by replacing this module's `sample_batch` with real latents.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch

from .config import IMG_SIZE, IMG_CH


# ---------------------------------------------------------------------------
# Shape primitives: each returns a boolean mask (H, W) of the drawn shape.
# ---------------------------------------------------------------------------
def _grid(size: int):
    ax = np.linspace(-1.0, 1.0, size)
    return np.meshgrid(ax, ax)  # xx, yy


def _circle(size, cx, cy, r):
    xx, yy = _grid(size)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _ring(size, cx, cy, r):
    xx, yy = _grid(size)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (d <= r) & (d >= r * 0.55)


def _square(size, cx, cy, r):
    xx, yy = _grid(size)
    return (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)


def _diamond(size, cx, cy, r):
    xx, yy = _grid(size)
    return (np.abs(xx - cx) + np.abs(yy - cy)) <= r


def _triangle(size, cx, cy, r):
    xx, yy = _grid(size)
    # upward triangle centred at (cx, cy)
    a = (yy - cy) <= r
    b = (yy - cy) >= -r + 1.6 * np.abs(xx - cx)
    return a & b


def _cross(size, cx, cy, r):
    xx, yy = _grid(size)
    h = (np.abs(yy - cy) <= r * 0.33) & (np.abs(xx - cx) <= r)
    v = (np.abs(xx - cx) <= r * 0.33) & (np.abs(yy - cy) <= r)
    return h | v


SHAPES: dict[str, Callable] = {
    "circle": _circle, "ring": _ring, "square": _square,
    "diamond": _diamond, "triangle": _triangle, "cross": _cross,
}


# ---------------------------------------------------------------------------
# Class registry.  Difficulty / ambiguity are *designed in*, then verified
# empirically in notebook 00 (we don't just assert them).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassSpec:
    name: str
    shapes: tuple[str, ...]     # >1 shape => ambiguous (bimodal conditional)
    color: tuple[float, float, float]
    difficulty: str             # 'easy' | 'medium' | 'hard'
    jitter: float               # size/colour jitter magnitude
    bg_noise: float             # background clutter level
    pos_var: float = 0.10       # spatial position spread (0=centred, ->0.6 corners)
    mode_colors: tuple = ()     # optional per-mode colours (ambiguous classes)
    mode_pos: tuple = ()        # optional per-mode centre offsets (ambiguous)

    @property
    def ambiguous(self) -> bool:
        return len(self.shapes) > 1


# 16 classes: a compact stand-in for the "50 ImageNet classes" of Task 1.4,
# spanning easy / medium / hard and including 3 deliberately ambiguous classes.
# `pos_var` sets how much the object can move around the canvas -- position is a
# *global-structure* choice the model must commit to early, so higher pos_var
# raises early-step ensemble variance even for unimodal classes.  Ambiguous
# classes additionally have two well-separated modes (shape+colour+region).
CLASS_SPECS: list[ClassSpec] = [
    # ---- easy: unambiguous, centred, clean --------------------------------
    ClassSpec("red_circle",     ("circle",),   (0.90, 0.15, 0.15), "easy",   0.05, 0.02, 0.06),
    ClassSpec("green_square",   ("square",),   (0.15, 0.80, 0.20), "easy",   0.05, 0.02, 0.06),
    ClassSpec("blue_triangle",  ("triangle",), (0.20, 0.35, 0.90), "easy",   0.05, 0.02, 0.06),
    ClassSpec("yellow_diamond", ("diamond",),  (0.95, 0.85, 0.10), "easy",   0.05, 0.02, 0.06),
    # ---- medium: colour jitter + moderate position spread -----------------
    ClassSpec("cyan_ring",      ("ring",),     (0.10, 0.80, 0.85), "medium", 0.15, 0.06, 0.28),
    ClassSpec("magenta_cross",  ("cross",),    (0.85, 0.15, 0.75), "medium", 0.15, 0.06, 0.28),
    ClassSpec("orange_circle",  ("circle",),   (0.95, 0.55, 0.10), "medium", 0.16, 0.08, 0.30),
    ClassSpec("teal_square",    ("square",),   (0.10, 0.55, 0.55), "medium", 0.16, 0.08, 0.30),
    # ---- hard: small shape, heavy clutter, wide position spread -----------
    ClassSpec("purple_diamond", ("diamond",),  (0.55, 0.20, 0.75), "hard",   0.22, 0.16, 0.42),
    ClassSpec("lime_triangle",  ("triangle",), (0.65, 0.90, 0.20), "hard",   0.22, 0.16, 0.42),
    ClassSpec("brown_ring",     ("ring",),     (0.55, 0.35, 0.15), "hard",   0.24, 0.18, 0.44),
    ClassSpec("gray_cross",     ("cross",),    (0.55, 0.55, 0.55), "hard",   0.24, 0.18, 0.44),
    ClassSpec("pink_circle",    ("circle",),   (0.95, 0.55, 0.70), "hard",   0.22, 0.16, 0.42),
    # ---- ambiguous: two well-separated modes (shape+colour+region) --------
    ClassSpec("crane", ("triangle", "square"), (0.30, 0.30, 0.35), "hard", 0.12, 0.08, 0.10,
              mode_colors=((0.90, 0.75, 0.15), (0.30, 0.45, 0.95)),
              mode_pos=((-0.45, -0.35), (0.45, 0.35))),
    ClassSpec("mouse", ("circle", "cross"), (0.60, 0.60, 0.62), "medium", 0.12, 0.08, 0.10,
              mode_colors=((0.85, 0.30, 0.30), (0.25, 0.75, 0.55)),
              mode_pos=((-0.40, 0.35), (0.40, -0.35))),
    ClassSpec("bank", ("ring", "diamond"), (0.25, 0.55, 0.45), "medium", 0.12, 0.08, 0.10,
              mode_colors=((0.20, 0.55, 0.95), (0.95, 0.55, 0.20)),
              mode_pos=((0.0, -0.45), (0.0, 0.45))),
]

NUM_CLASSES = len(CLASS_SPECS)
CLASS_NAMES = [c.name for c in CLASS_SPECS]
EASY_CLASSES = [i for i, c in enumerate(CLASS_SPECS) if c.difficulty == "easy"]
HARD_CLASSES = [i for i, c in enumerate(CLASS_SPECS) if c.difficulty == "hard"]
AMBIGUOUS_CLASSES = [i for i, c in enumerate(CLASS_SPECS) if c.ambiguous]


# ---------------------------------------------------------------------------
def _render_one(spec: ClassSpec, rng: np.random.Generator, force_mode: int | None = None):
    """Render a single (3, H, W) image in [-1, 1] for the given class spec."""
    size = IMG_SIZE
    # background: soft coloured clutter
    bg = np.stack([np.full((size, size), 0.10 + 0.05 * rng.random()) for _ in range(3)])
    if spec.bg_noise > 0:
        bg = bg + spec.bg_noise * rng.standard_normal((3, size, size))

    # choose which shape (mode) this sample takes
    mode = force_mode if force_mode is not None else rng.integers(len(spec.shapes))
    shape_fn = SHAPES[spec.shapes[mode]]

    j = spec.jitter
    # object centre: per-mode anchor (ambiguous) + isotropic position spread
    ax, ay = (spec.mode_pos[mode] if spec.mode_pos else (0.0, 0.0))
    cx = ax + spec.pos_var * rng.uniform(-1, 1)
    cy = ay + spec.pos_var * rng.uniform(-1, 1)
    r = 0.42 - 0.12 * (spec.difficulty == "hard") + j * rng.uniform(-0.3, 0.3)
    r = float(np.clip(r, 0.22, 0.60))
    mask = shape_fn(size, cx, cy, r)

    base_color = np.array(spec.mode_colors[mode] if spec.mode_colors else spec.color)
    color = np.clip(base_color + j * 0.4 * rng.standard_normal(3), 0.0, 1.0)

    img = bg.copy()
    for ch in range(3):
        img[ch][mask] = color[ch]
    img = np.clip(img, 0.0, 1.0)
    return (img * 2.0 - 1.0).astype(np.float32), mode  # -> [-1, 1]


def sample_batch(labels: np.ndarray | list[int], seed: int | None = None):
    """Return (images[-1,1] float32 tensor (B,3,H,W), modes np.ndarray)."""
    rng = np.random.default_rng(seed)
    imgs, modes = [], []
    for c in labels:
        img, mode = _render_one(CLASS_SPECS[int(c)], rng)
        imgs.append(img)
        modes.append(mode)
    x = torch.from_numpy(np.stack(imgs))
    return x, np.asarray(modes)


def build_dataset(samples_per_class: int, seed: int = 0):
    """Full training set: (images tensor (N,3,H,W) in [-1,1], labels tensor (N,))."""
    rng_labels = np.random.default_rng(seed)
    labels = np.repeat(np.arange(NUM_CLASSES), samples_per_class)
    rng_labels.shuffle(labels)
    x, _ = sample_batch(labels, seed=seed)
    return x, torch.from_numpy(labels).long()


def to_display(x: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1]  ->  (B,H,W,3) in [0,1] for matplotlib.imshow."""
    x = x.detach().float().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    return x.permute(0, 2, 3, 1).numpy()


def class_table():
    """Pandas DataFrame describing the class registry (used in notebook 00)."""
    import pandas as pd
    return pd.DataFrame([{
        "id": i, "name": c.name, "difficulty": c.difficulty,
        "ambiguous": c.ambiguous, "n_modes": len(c.shapes),
        "shapes": "/".join(c.shapes), "pos_var": c.pos_var,
        "jitter": c.jitter, "bg_noise": c.bg_noise,
    } for i, c in enumerate(CLASS_SPECS)])
