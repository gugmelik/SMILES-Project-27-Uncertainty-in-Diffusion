"""The three Task-1 uncertainty proxies (+ epistemic/aleatoric decomposition).

All functions treat the model as a black box and return both a **scalar curve**
over the denoising path (for timestep profiles, Task 3) and **spatial maps**
(for heatmaps, Task 4).  Notation matches the proposal:

  U_ens(t, c)   = Var_k [ D(x_t^(k), t, c) ]          across K noise seeds
  U_route(t, c) = Var_m [ D_r^(m)(x_t, t, c) | z ]    across M TREAD routes
  U_attn(t, l)  = mean-query Shannon entropy of A^(l) across heads
  U_epi / U_ale : law of total variance over (routes, seeds)

`D` is the x0-prediction (`diffusion.predict_x0`), the standard target for
diffusion uncertainty.  Variance is reduced over channels & pixels for the scalar
curve; spatial maps keep the (H, W) layout and are optionally pooled to the DiT
patch grid.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from .models import Route


# --------------------------------------------------------------- helpers ------
def _reduce_scalar(var_map: torch.Tensor) -> float:
    """(C,H,W) or (H,W) variance map -> mean scalar."""
    return float(var_map.mean().item())


def to_patch_grid(spatial: torch.Tensor, grid: int) -> torch.Tensor:
    """(H,W) pixel map -> (grid,grid) patch map by average pooling."""
    x = spatial[None, None].float()
    return F.adaptive_avg_pool2d(x, grid)[0, 0]


# ----------------------------------------------- 1) ensemble variance U_ens ---
@torch.no_grad()
def ensemble_variance(model, diff, cls: int, K: int, device,
                      base_seed: int = 0, eta: float = 0.0):
    """Run K seeds through the full path; variance of x0_hat at each step.

    eta=0 (default) uses deterministic DDIM, giving the clean "high early ->
    decay late" timestep profile the proposal describes.  eta=1 switches to
    stochastic (ancestral) sampling, under which different seeds can commit to
    different modes (see the mode-selection demo in notebook 01); note that the
    injected per-step noise then adds a large class-independent variance floor.

    Returns dict: timesteps, scalar curve U_ens(t), and per-step spatial maps
    (channel-averaged pixel variance).
    """
    y = torch.full((K,), cls, device=device, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(base_seed + 1234 * cls)
    noise = torch.randn(K, model.cfg.in_ch, model.cfg.img_size, model.cfg.img_size,
                        device=device, generator=g)
    gpath = torch.Generator(device=device).manual_seed(base_seed + 4321 * cls)
    _, traj = diff.ddim_sample(model, y, noise, route=None, record=True,
                               eta=eta, generator=gpath)

    ts, curve, maps = [], [], []
    for step in traj:
        x0 = step["x0_hat"]                       # (K, C, H, W)
        var = x0.var(dim=0, unbiased=False)       # (C, H, W)
        ts.append(step["t"])
        curve.append(_reduce_scalar(var))
        maps.append(var.mean(0).cpu())            # (H, W)
    return {"t": np.array(ts), "curve": np.array(curve), "maps": maps,
            "proxy": "U_ens", "cls": cls}


# ------------------------------------------------- 2) routing variance U_route -
@torch.no_grad()
def routing_variance(model, diff, cls: int, M: int, device,
                     n_ref_seeds: int = 4, base_seed: int = 0):
    """At fixed seed(s), variance of x0_hat across M random TREAD routes.

    We average over `n_ref_seeds` reference trajectories for stability; each
    supplies the x_t at every step, at which M routed predictions are drawn.
    """
    cfg = model.cfg
    y1 = torch.full((1,), cls, device=device, dtype=torch.long)
    ref_curves, ref_maps = [], None

    for s in range(n_ref_seeds):
        g = torch.Generator(device=device).manual_seed(base_seed + 777 * cls + s)
        noise = torch.randn(1, cfg.in_ch, cfg.img_size, cfg.img_size,
                            device=device, generator=g)
        # reference full-path trajectory -> x_t at each step (fixed seed z)
        _, traj = diff.ddim_sample(model, y1, noise, route=None, record=True)

        curve, maps = [], []
        for step in traj:
            x_t = step["x_t"].expand(M, -1, -1, -1)         # (M, C, H, W)
            tb = torch.full((M,), step["t"], device=device, dtype=torch.long)
            yM = torch.full((M,), cls, device=device, dtype=torch.long)
            rg = torch.Generator(device=device).manual_seed(
                base_seed + 999 * cls + 13 * s + step["t"])
            preds = []
            for m in range(M):
                route = Route.sample(cfg, batch=1, n_patches=cfg.n_patches,
                                     device=device, generator=rg)
                route.routed_mask = route.routed_mask.expand(M, -1)
                eps = model(x_t[m:m+1], tb[:1], yM[:1], route=Route(
                    route.i, route.j, route.routed_mask[m:m+1]))
                preds.append(diff.predict_x0(x_t[m:m+1], tb[:1], eps))
            preds = torch.cat(preds, 0)                     # (M, C, H, W)
            var = preds.var(dim=0, unbiased=False)          # (C, H, W)
            curve.append(_reduce_scalar(var))
            maps.append(var.mean(0).cpu())
        ref_curves.append(curve)
        if ref_maps is None:
            ref_maps = maps
            ts = [step["t"] for step in traj]
        else:
            ref_maps = [a + b for a, b in zip(ref_maps, maps)]

    ref_maps = [m / n_ref_seeds for m in ref_maps]
    return {"t": np.array(ts), "curve": np.array(ref_curves).mean(0),
            "maps": ref_maps, "proxy": "U_route", "cls": cls}


@torch.no_grad()
def route_sanity(model, diff, cls: int, device, n_seeds: int = 8, base_seed: int = 0):
    """Task 1.2 sanity: does inference-time routing degrade x0 catastrophically?

    Returns mean MSE between full-path and routed final samples (smaller = routing
    is a meaningful perturbation, not destruction).
    """
    cfg = model.cfg
    y = torch.full((n_seeds,), cls, device=device, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(base_seed + cls)
    noise = torch.randn(n_seeds, cfg.in_ch, cfg.img_size, cfg.img_size,
                        device=device, generator=g)
    x_full, _ = diff.ddim_sample(model, y, noise, route=None, record=False)
    route = Route.sample(cfg, batch=n_seeds, n_patches=cfg.n_patches, device=device)
    x_route, _ = diff.ddim_sample(model, y, noise, route=route, record=False)
    mse = F.mse_loss(x_route, x_full).item()
    signal = x_full.var().item()
    return {"mse": mse, "signal_var": signal, "rel": mse / (signal + 1e-8),
            "x_full": x_full.cpu(), "x_route": x_route.cpu()}


# ------------------------------------------------- 3) attention entropy U_attn -
@torch.no_grad()
def attention_entropy(model, diff, cls: int, device, n_seeds: int = 8,
                      base_seed: int = 0):
    """Per-layer Shannon entropy of attention maps along the denoising path.

    Returns dict with timesteps, per-layer curves (n_bins-like steps x depth),
    the layer-averaged scalar curve, and one representative attention map per
    (early/mid/late) step for visualisation.
    """
    cfg = model.cfg
    y = torch.full((n_seeds,), cls, device=device, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(base_seed + 55 * cls)
    noise = torch.randn(n_seeds, cfg.in_ch, cfg.img_size, cfg.img_size,
                        device=device, generator=g)
    # reference trajectory (full path) to obtain x_t at each step
    _, traj = diff.ddim_sample(model, y, noise, route=None, record=True)

    ts, per_layer, sample_maps = [], [], {}
    for step in traj:
        tb = torch.full((n_seeds,), step["t"], device=device, dtype=torch.long)
        _, attn = model(step["x_t"], tb, y, return_attn=True)  # (L,B,H,N,N)
        # per-query entropy: -sum_j A_ij log A_ij, averaged over queries/heads/batch
        ent = -(attn * (attn.clamp_min(1e-12)).log()).sum(-1)  # (L,B,H,N)
        ent = ent.mean(dim=(1, 2, 3))                          # (L,)
        per_layer.append(ent.cpu().numpy())
        ts.append(step["t"])
        sample_maps[step["t"]] = attn[:, 0].mean(0).cpu()      # (H, N, N) head maps of sample 0
    per_layer = np.array(per_layer)                            # (steps, L)
    return {"t": np.array(ts), "per_layer": per_layer,
            "curve": per_layer.mean(1), "sample_maps": sample_maps,
            "proxy": "U_attn", "cls": cls}


# --------------------------------------- epistemic / aleatoric decomposition --
@torch.no_grad()
def epi_ale_decomposition(model, diff, cls: int, device,
                          n_routes: int = 8, n_seeds: int = 8, base_seed: int = 0):
    """Law of total variance over (routes, seeds).

    Builds P[r, s] = D_{r}(x_t^{(s)}, t, c) at each step, then
      U_epi = mean_s Var_r(P[:, s])   (route variance = model/sub-network noise)
      U_ale = mean_r Var_s(P[r, :])   (seed  variance = data noise)
    """
    cfg = model.cfg
    # reference trajectories for n_seeds seeds
    trajs = []
    for s in range(n_seeds):
        g = torch.Generator(device=device).manual_seed(base_seed + 4242 * cls + s)
        noise = torch.randn(1, cfg.in_ch, cfg.img_size, cfg.img_size,
                            device=device, generator=g)
        y1 = torch.full((1,), cls, device=device, dtype=torch.long)
        _, traj = diff.ddim_sample(model, y1, noise, route=None, record=True)
        trajs.append(traj)

    # pre-sample R routes (shared across seeds/steps for a clean grid)
    rg = torch.Generator(device=device).manual_seed(base_seed + 31 * cls)
    routes = [Route.sample(cfg, 1, cfg.n_patches, device, generator=rg)
              for _ in range(n_routes)]

    n_steps = len(trajs[0])
    U_epi, U_ale, ts = [], [], []
    for k in range(n_steps):
        t = trajs[0][k]["t"]
        # P[r, s] -> x0 map (C,H,W)
        P = torch.empty(n_routes, n_seeds, cfg.in_ch, cfg.img_size, cfg.img_size,
                        device=device)
        for s in range(n_seeds):
            x_t = trajs[s][k]["x_t"]
            tb = torch.full((1,), t, device=device, dtype=torch.long)
            y1 = torch.full((1,), cls, device=device, dtype=torch.long)
            for r, route in enumerate(routes):
                eps = model(x_t, tb, y1, route=route)
                P[r, s] = diff.predict_x0(x_t, tb, eps)[0]
        u_epi = P.var(dim=0, unbiased=False).mean(dim=0).mean().item()  # mean_s Var_r
        u_ale = P.var(dim=1, unbiased=False).mean(dim=0).mean().item()  # mean_r Var_s
        U_epi.append(u_epi); U_ale.append(u_ale); ts.append(t)
    return {"t": np.array(ts), "U_epi": np.array(U_epi), "U_ale": np.array(U_ale),
            "cls": cls}


def estimate_t_star(t: np.ndarray, curve: np.ndarray) -> int:
    """Mode-transition point: timestep of steepest drop in the curve (Task 3).

    `t` runs high->low along the denoising path; returns the t at the largest
    negative step-to-step change (steepest collapse of uncertainty).
    """
    d = np.diff(curve)
    return int(t[:-1][np.argmin(d)])
