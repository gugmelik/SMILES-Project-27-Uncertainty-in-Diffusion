# `duq/` — model-agnostic reference implementation of the three proxies

This folder holds **only the library**, not the experiments that produced the
committed results. Those live at the repo root (`task1_panel50.ipynb`).

| module | contents |
|--------|----------|
| `config.py` | hyper-parameters (`CFG`): K seeds, M routes, bins, TREAD span / selection rate |
| `data.py` | synthetic class registry + renderer (used only by the local scaffold) |
| `models.py` | `TinyDiT` (AdaLN-Zero, capturable attention), `Route` (TREAD), `EMA` |
| `diffusion.py` | DDPM/DDIM schedule, `predict_x0`, sampling, timestep bins |
| `proxies.py` | `ensemble_variance`, `routing_variance`, `attention_entropy`, `route_sanity`, `epi_ale_decomposition` |
| `train.py` | trains the local scaffold model |
| `viz.py` | plotting helpers |

## Why this exists

`U_route` (routing variance) is the project's only **epistemic** proxy, and it is
only legitimate on a **TREAD-trained** model — a network trained as a distribution
over sub-networks. There is **no publicly released TREAD DiT-XL/2 checkpoint**, so
it could not be run on the real target.

To have the implementation ready for the moment a checkpoint appears, `proxies.py`
is written **model-agnostically**: every function treats the model as a black box
`(x_t, t, y, route) -> eps`. It was validated against a small TREAD-style DiT
trained locally on a synthetic dataset. Swapping in a real TREAD checkpoint is a
one-function change.

> **The local scaffold is not committed.** It trains a model from scratch, which the
> proposal explicitly forbids for the actual experiments (*"No retraining or weight
> modification is required; we use publicly available checkpoints"*). It exists only
> to exercise `routing_variance` end-to-end. All reported results come from the
> pretrained DiT-XL/2, inference-only.

## Reported results come from

- [`../task1_panel50.ipynb`](../task1_panel50.ipynb) — 50 ImageNet classes × 20 timestep bins on pretrained DiT-XL/2
- [`../scripts/task1_panel50.py`](../scripts/task1_panel50.py) — the same, as a headless script
