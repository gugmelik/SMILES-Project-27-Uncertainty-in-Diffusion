"""Standard DDPM schedule + DDIM sampling, plus x0 recovery and timestep bins.

Only the pieces Task 1 needs:
  * `q_sample`       : forward diffusion  x_t = sqrt(abar) x0 + sqrt(1-abar) eps
  * `predict_x0`     : recover x0_hat from a model's eps prediction (the quantity
                       whose variance every proxy measures)
  * `ddim_sample`    : deterministic denoising path, recording x0_hat at each step
  * `timestep_bins`  : map the continuous schedule to `n_bins` bins so all proxies
                       are logged on a common, comparable timestep axis.
"""
from __future__ import annotations
import numpy as np
import torch

from .config import DiffCfg


class Diffusion:
    def __init__(self, cfg: DiffCfg, device):
        self.cfg = cfg
        self.device = device
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps,
                               device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.abar = torch.cumprod(self.alphas, dim=0)          # \bar alpha_t

    # -------------------------------------------------------------- forward ---
    def q_sample(self, x0, t, noise):
        a = self.abar[t].sqrt().view(-1, 1, 1, 1)
        am = (1 - self.abar[t]).sqrt().view(-1, 1, 1, 1)
        return a * x0 + am * noise

    def predict_x0(self, x_t, t, eps, clamp: bool = True):
        """x0_hat = (x_t - sqrt(1-abar) eps) / sqrt(abar).

        At high noise 1/sqrt(abar) is huge, so raw x0_hat is numerically wild;
        we clamp to the data range [-1, 1] ("static thresholding", as in standard
        DDIM samplers).  This is the quantity whose variance the proxies measure.
        """
        a = self.abar[t].sqrt().view(-1, 1, 1, 1)
        am = (1 - self.abar[t]).sqrt().view(-1, 1, 1, 1)
        x0 = (x_t - am * eps) / a
        return x0.clamp(-1, 1) if clamp else x0

    # -------------------------------------------------------------- sampling --
    def ddim_timesteps(self):
        steps = self.cfg.ddim_steps
        ts = np.linspace(self.cfg.timesteps - 1, 0, steps).round().astype(int)
        return torch.from_numpy(ts).long().to(self.device)

    @torch.no_grad()
    def ddim_sample(self, model, y, noise, route=None, record=True, eta=0.0,
                    generator=None):
        """(Generalised) DDIM path from `noise`.

        eta=0 -> deterministic DDIM; eta=1 -> stochastic (ancestral) sampling,
        which injects fresh noise each step and lets different seeds explore
        different modes (needed so ensemble variance reflects mode diversity).

        Returns (x0_final, traj); traj records {t, x_t, x0_hat} at each step.
        `route` is passed to the model (None => full path).
        """
        x = noise
        ts = self.ddim_timesteps()
        traj = []
        for k in range(len(ts)):
            t = ts[k]
            tb = torch.full((x.shape[0],), int(t), device=self.device, dtype=torch.long)
            eps = model(x, tb, y, route=route)
            x0 = self.predict_x0(x, tb, eps)
            if record:
                traj.append({"t": int(t), "x_t": x.clone(), "x0_hat": x0.clone()})
            t_next = ts[k + 1] if k + 1 < len(ts) else torch.tensor(0, device=self.device)
            ab_t = self.abar[t]
            ab_n = (self.abar[t_next] if t_next > 0
                    else torch.tensor(1.0, device=self.device))
            # stochastic DDIM: sigma controls how much fresh noise is injected
            sigma = eta * ((1 - ab_n) / (1 - ab_t)).clamp(0, 1).sqrt() \
                        * (1 - ab_t / ab_n.clamp_min(1e-8)).clamp_min(0).sqrt()
            sigma = sigma if t_next > 0 else torch.zeros_like(ab_t)
            c = (1 - ab_n - sigma ** 2).clamp_min(0).sqrt()
            x = ab_n.sqrt() * x0 + c * eps
            if float(sigma) > 0:
                x = x + sigma * torch.randn(x.shape, device=self.device,
                                            generator=generator)
        return x, traj

    # ------------------------------------------------------------- bins -------
    def timestep_bins(self):
        """Edges and centres for `n_bins` uniform bins over [0, timesteps)."""
        edges = np.linspace(0, self.cfg.timesteps, self.cfg.n_bins + 1).astype(int)
        centres = ((edges[:-1] + edges[1:]) / 2).astype(int)
        return edges, centres

    def bin_of(self, t: int) -> int:
        edges, _ = self.timestep_bins()
        return int(np.clip(np.digitize(t, edges) - 1, 0, self.cfg.n_bins - 1))
