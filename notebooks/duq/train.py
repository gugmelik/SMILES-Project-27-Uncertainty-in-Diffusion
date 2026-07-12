"""Train the tiny TREAD-DiT on the synthetic dataset.

Crucially, training uses **random TREAD routing per step** (a fresh route + token
mask each batch).  This is what turns the network into an ensemble of
sub-networks, so that at inference `U_route` is a legitimate *epistemic* signal
rather than an arbitrary perturbation (proposal: "TREAD as a stochastic-depth
training procedure").  The checkpoint is cached to disk; re-runs load it.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F

from .config import CFG, CKPT_PATH, seed_everything
from .models import TinyDiT, Route, EMA
from .diffusion import Diffusion
from . import data as ddata


def train(cfg=CFG, device=None, force: bool = False, log_every: int = 25,
          tread_prob: float = 0.75, verbose: bool = True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.model.num_classes = ddata.NUM_CLASSES

    if CKPT_PATH.exists() and not force:
        model = TinyDiT(cfg.model).to(device)
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["ema"])
        model.eval()
        if verbose:
            print(f"Loaded cached checkpoint: {CKPT_PATH}  "
                  f"(loss={ckpt.get('final_loss', float('nan')):.4f})")
        return model, ckpt.get("history", [])

    seed_everything(cfg.train.seed)
    X, Y = ddata.build_dataset(cfg.train.samples_per_class, seed=cfg.train.seed)
    X, Y = X.to(device), Y.to(device)
    N = X.shape[0]
    diff = Diffusion(cfg.diff, device)

    model = TinyDiT(cfg.model).to(device)
    ema = EMA(model, cfg.train.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=0.0)
    if verbose:
        print(f"Model params: {model.num_params()/1e6:.2f}M | "
              f"dataset: {N} imgs, {ddata.NUM_CLASSES} classes | device: {device}")

    history = []
    steps_per_epoch = N // cfg.train.batch_size
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        perm = torch.randperm(N, device=device)
        run = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * cfg.train.batch_size:(s + 1) * cfg.train.batch_size]
            x0, y = X[idx], Y[idx]
            b = x0.shape[0]
            t = torch.randint(0, cfg.diff.timesteps, (b,), device=device)
            noise = torch.randn_like(x0)
            x_t = diff.q_sample(x0, t, noise)

            # random TREAD route this step (stochastic depth), or full path
            if torch.rand(1).item() < tread_prob:
                route = Route.sample(cfg.model, b, cfg.model.n_patches, device)
            else:
                route = None

            eps = model(x_t, t, y, route=route)
            loss = F.mse_loss(eps, noise)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)
            run += loss.item()
            if verbose and (s % log_every == 0):
                print(f"  epoch {epoch:2d} step {s:3d}/{steps_per_epoch}  "
                      f"loss {loss.item():.4f}")
        avg = run / max(1, steps_per_epoch)
        history.append(avg)
        if verbose:
            print(f"epoch {epoch:2d} done | avg loss {avg:.4f} | "
                  f"{time.time()-t0:.1f}s")

    ema_model = TinyDiT(cfg.model).to(device)
    ema.copy_to(ema_model); ema_model.eval()
    torch.save({"ema": ema_model.state_dict(), "raw": model.state_dict(),
                "history": history, "final_loss": history[-1],
                "cfg_model": cfg.model.__dict__}, CKPT_PATH)
    if verbose:
        print(f"Saved checkpoint -> {CKPT_PATH}")
    return ema_model, history
