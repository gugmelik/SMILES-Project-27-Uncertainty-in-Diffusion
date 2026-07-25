"""Global configuration: paths, device, image/model/diffusion hyper-parameters.

Kept deliberately small so the whole Task-1 pipeline runs end-to-end in a couple
of minutes on a laptop GPU (and still finishes on CPU).  The variable names match
the notation in the proposal (K seeds, M routes, timestep bins, selection rate).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
import torch

# ---------------------------------------------------------------- paths -------
PKG_DIR = Path(__file__).resolve().parent
NB_DIR = PKG_DIR.parent                      # .../notebooks
ARTIFACTS = NB_DIR / "artifacts"             # trained checkpoint + cached data
RESULTS = NB_DIR / "results"                 # logged metric tables (Task 1.4)
for _d in (ARTIFACTS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

CKPT_PATH = ARTIFACTS / "tiny_tread_dit.pt"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()


# ------------------------------------------------------------- image data -----
IMG_SIZE = 32          # we operate directly in pixel space (no VAE needed)
IMG_CH = 3


# --------------------------------------------------------------- model --------
@dataclass
class ModelCfg:
    img_size: int = IMG_SIZE
    in_ch: int = IMG_CH
    patch: int = 4                 # -> (32/4)^2 = 64 tokens
    hidden: int = 192
    depth: int = 8
    heads: int = 6
    num_classes: int = 16          # overwritten from the dataset at build time
    # TREAD routing (see models.TinyDiT / proposal Task 1.2)
    route_i: int = 2               # first routed block  (i ~ 2)
    route_j: int = 4               # last routed block   (j ~ B-4 with B=8)
    selection_rate: float = 0.5    # fraction of tokens routed around blocks i..j

    @property
    def n_patches(self) -> int:
        return (self.img_size // self.patch) ** 2


# ----------------------------------------------------------- diffusion --------
@dataclass
class DiffCfg:
    timesteps: int = 1000          # training schedule length
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    n_bins: int = 20               # timestep bins used for all uncertainty logs
    ddim_steps: int = 40           # sampling steps along the denoising path


# ---------------------------------------------------- uncertainty proxies -----
@dataclass
class ProxyCfg:
    K_seeds: int = 16              # ensemble variance  (proposal: K = 16)
    M_routes: int = 16             # routing variance   (proposal: M = 16)


# ---------------------------------------------------------- training ----------
@dataclass
class TrainCfg:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    ema_decay: float = 0.999
    samples_per_class: int = 384
    seed: int = 0


@dataclass
class Cfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    diff: DiffCfg = field(default_factory=DiffCfg)
    proxy: ProxyCfg = field(default_factory=ProxyCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


CFG = Cfg()


def seed_everything(seed: int = 0) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
