# SMILES Project 27 — Uncertainty in Diffusion and Flow Matching Models

Research project for the [SMILES summer school](https://github.com/gugmelik/SMILES-Project-27-Uncertainty-in-Diffusion). We study **per-timestep uncertainty in Diffusion Transformers (DiT)** — how different uncertainty signals at early vs late denoising steps relate to generation quality.

**Project proposal:** see [`Uncertainty_in_Diffusion_transformers__smiles_.pdf`](Uncertainty_in_Diffusion_transformers__smiles_.pdf)

---

## Project summary

Diffusion Transformer models denoise in steps. Early steps (high noise) shape global structure and semantic mode; late steps (low noise) refine local details. This project asks whether those regimes show different kinds of uncertainty, and whether uncertainty at specific timesteps predicts final image quality.

We use inference-time, training-free uncertainty proxies:

| Proxy | Description | Models |
|-------|-------------|--------|
| **Ensemble variance** (`U_ens`) | Variance across multiple noise seeds | Vanilla DiT, TREAD DiT |
| **Routing variance** (`U_route`) | Variance across random TREAD sub-network routes | TREAD DiT only |
| **Attention entropy** (`U_attn`) | Shannon entropy of self-attention maps | Vanilla DiT, TREAD DiT |

Target checkpoints: pretrained **DiT-XL/2** and **TREAD DiT-XL/2** on ImageNet-256.

---

## How we collaborate

This repository is shared by multiple participants. **Do not push directly to `main`.** Each person works on their own branch so experiments, code, and partial results do not overwrite each other.

### Golden rules

1. **`main` is protected** — keep it stable; merge only reviewed, working changes.
2. **One branch per person (or per task)** — never commit someone else's work to your branch without asking.
3. **Push early, push often** — back up your work to the remote branch regularly.
4. **Pull before you push** — sync with `main` (or your base branch) to reduce merge conflicts.
5. **Open a Pull Request** when your work is ready for review — do not force-push to `main`.

### Branch naming

Use a clear, unique name:

```
<your-name>/<short-topic>
```

Examples:

- `alice/ensemble-variance`
- `bob/routing-variance`
- `carol/task3-timestep-profiles`
- `dave/heatmaps`

### Workflow (step by step)

#### 1. Clone and set up (first time only)

```bash
git clone https://github.com/gugmelik/SMILES-Project-27-Uncertainty-in-Diffusion.git
cd SMILES-Project-27-Uncertainty-in-Diffusion
```

#### 2. Start from the latest `main`

```bash
git checkout main
git pull origin main
```

#### 3. Create your branch

```bash
git checkout -b your-name/your-topic
```

#### 4. Do your work, commit locally

```bash
git add <files>
git commit -m "Short description of what you changed"
```

Write commit messages that explain *why*, not just *what*.

#### 5. Push your branch to GitHub

```bash
git push -u origin your-name/your-topic
```

The `-u` flag links your local branch to the remote — after the first push you can use `git push` alone.

#### 6. Open a Pull Request

On GitHub: **Compare & pull request** → target branch: `main` → add a short summary of what you did and how to test it → request review from a teammate.

#### 7. Stay up to date while working

If `main` has new commits, rebase or merge them into your branch:

```bash
git checkout your-name/your-topic
git fetch origin
git merge origin/main
# resolve any conflicts, then:
git push
```

### What to avoid

| Don't | Why |
|-------|-----|
| Push to `main` directly | Breaks others' baselines and causes conflicts |
| Commit large binaries (datasets, checkpoints) | Bloats the repo; use links or `.gitignore` |
| Edit files on someone else's branch | Their work may be overwritten |
| Force-push (`git push --force`) without agreement | Can destroy teammates' commits |

### Suggested folder layout (as the project grows)

Organize by task or owner to reduce collisions:

```
├── src/                  # shared library code (merge via PR)
├── experiments/
│   ├── alice/            # Alice's notebooks, scripts, configs
│   ├── bob/
│   └── ...
├── results/              # small result files only; large outputs → external storage
├── notebooks/
└── docs/
```

If you add a new top-level folder, mention it in your PR so others know where to put related work.

---

## Tasks (from project proposal)

| Task | Goal |
|------|------|
| **1** | Implement the three uncertainty proxies (`U_ens`, `U_route`, `U_attn`) on DiT-XL/2 and TREAD DiT-XL/2 |
| **2** | Validate routing variance as an epistemic signal (OOD, capacity, non-redundancy, route sanity) |
| **3** | Timestep profiles and epistemic/aleatoric decomposition; estimate mode-transition point `t*` |
| **4** | Spatial uncertainty heatmaps (seed vs routing) vs semantic regions |
| **5** | Link uncertainty to per-sample quality (CLIP, IS, LPIPS correlations) |

Full specifications are in the PDF proposal.

---

## Environment setup

> Dependencies will be added as the codebase grows. For now, plan on:

- Python 3.10+
- PyTorch
- Pretrained [DiT](https://github.com/facebookresearch/DiT) and [TREAD](https://github.com/) checkpoints
- ImageNet-256 evaluation setup

When `requirements.txt` or `environment.yml` is added to `main`, install from that file and note any extra steps in your PR.

---

## Communication

- **Code & results:** this repository (your branch → PR → `main`)
- **Questions & progress:** coordinate with the team (chat / meetings — add your channel link here if you use one)
- **Blockers:** open a GitHub Issue so others can see and help

---

## References

- Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT)
- Krause et al., *TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training*
- Jazbec et al., *Generative Uncertainty in Diffusion Models*
- Chan et al., *HyperDM* (NeurIPS 2024)

---

## License

TBD — agree as a team before publishing code publicly.
