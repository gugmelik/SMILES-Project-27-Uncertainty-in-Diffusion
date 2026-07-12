"""
duq -- Diffusion Uncertainty Quantification support package.

Small, self-contained toolkit that mirrors the SMILES Project 27 setup
("Per-Timestep Uncertainty in DiT") on a scale that runs on a single GPU/CPU:

  * a faithful *small* DiT (AdaLN-Zero conditioning, multi-head self-attention
    with capturable attention maps) trained with TREAD-style token routing,
  * a standard DDPM/DDIM diffusion wrapper,
  * the three Task-1 uncertainty proxies: U_ens, U_route, U_attn,
  * a synthetic, class-labelled image dataset with deliberately *easy* and
    *ambiguous* classes so the epistemic-vs-aleatoric story is visible.

The public research target is DiT-XL/2 + TREAD DiT-XL/2 on ImageNet-256.
Everything here is written so the same functions apply to the real models:
swap `duq.models.TinyDiT` for the official DiT and point `diffusion` at the
pretrained schedule -- the proxy code (`duq.proxies`) is model-agnostic.
"""

from . import config, data, models, diffusion, proxies, viz  # noqa: F401
