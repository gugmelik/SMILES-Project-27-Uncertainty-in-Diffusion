"""Variational Bayesian LoRA adapters for a frozen denoiser.

Making every weight of DiT-XL/2 Bayesian is hopeless, and a last-layer Laplace
can only express uncertainty that survives the final projection. The middle
ground is to freeze the pretrained backbone, attach low-rank adapters to a few
late blocks, and learn a mean-field Gaussian posterior over *those* parameters
with Bayes-by-Backprop:

    L(phi) = E_{theta ~ q_phi}[ L_diffusion(theta) ] + beta * KL(q_phi || p)

Because the adapters start at ``Delta W = 0``, the posterior mean stays close to
the pretrained model while the posterior spread carries the epistemic signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import freeze, unconditional_class
from .posteriors import ParameterDict, WeightPosterior
from .utils import randn_like_from


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class VariationalLoRALinear(nn.Module):
    """``nn.Linear`` plus a low-rank adapter with a mean-field Gaussian posterior.

    Two forward modes:

    ``sample_weights = True``
        Reparameterised draw from ``q(A, B)``; differentiable, used for fitting.
    ``sample_weights = False``
        Uses the ``lora_A`` / ``lora_B`` buffers, which
        :class:`BayesianLoRAPosterior` overwrites with a specific posterior draw.
        This is what keeps ``theta_m`` *fixed for a whole trajectory*, which is
        required for the posterior-predictive construction to mean anything.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 4,
        alpha: float | None = None,
        prior_std: float = 1e-2,
        init_posterior_std: float = 1e-4,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = freeze(base)
        self.rank = rank
        self.scaling = (alpha if alpha is not None else rank) / rank
        self.prior_std = prior_std
        self.sample_weights = False

        in_features, out_features = base.in_features, base.out_features
        device, dtype = base.weight.device, base.weight.dtype
        rho_init = _inverse_softplus(init_posterior_std)

        self.lora_A_mu = nn.Parameter(
            torch.randn(rank, in_features, device=device, dtype=dtype)
            / math.sqrt(in_features)
        )
        self.lora_B_mu = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        self.lora_A_rho = nn.Parameter(
            torch.full((rank, in_features), rho_init, device=device, dtype=dtype)
        )
        self.lora_B_rho = nn.Parameter(
            torch.full((out_features, rank), rho_init, device=device, dtype=dtype)
        )

        self.register_buffer("lora_A", self.lora_A_mu.detach().clone())
        self.register_buffer("lora_B", self.lora_B_mu.detach().clone())

    @property
    def lora_A_sigma(self) -> torch.Tensor:
        return F.softplus(self.lora_A_rho)

    @property
    def lora_B_sigma(self) -> torch.Tensor:
        return F.softplus(self.lora_B_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.sample_weights:
            a = self.lora_A_mu + self.lora_A_sigma * torch.randn_like(self.lora_A_mu)
            b = self.lora_B_mu + self.lora_B_sigma * torch.randn_like(self.lora_B_mu)
        else:
            a, b = self.lora_A, self.lora_B
        return out + self.scaling * F.linear(F.linear(x, a), b)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(A,B) || N(0, prior_std^2 I)), summed over adapter parameters."""
        total = torch.zeros((), device=self.lora_A_mu.device)
        for mu, sigma in (
            (self.lora_A_mu, self.lora_A_sigma),
            (self.lora_B_mu, self.lora_B_sigma),
        ):
            total = total + (
                math.log(self.prior_std)
                - torch.log(sigma)
                + (sigma.pow(2) + mu.pow(2)) / (2 * self.prior_std**2)
                - 0.5
            ).sum()
        return total

    def variational_parameters(self) -> list[nn.Parameter]:
        return [self.lora_A_mu, self.lora_A_rho, self.lora_B_mu, self.lora_B_rho]


def select_lora_targets(model: nn.Module, spec: str) -> tuple[str, ...]:
    """Module paths (not parameter names) to wrap with adapters.

    ``"last_layer"``          -> ``final_layer.linear``
    ``"late_blocks:N"``       -> ``attn.proj`` and ``mlp.fc2`` of the last N blocks,
                                 plus ``final_layer.linear``
    ``"regex:<pattern>"``     -> every ``nn.Linear`` whose path matches
    """
    linears = [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]

    if spec.startswith("regex:"):
        import re  # noqa: PLC0415

        pattern = re.compile(spec[len("regex:") :])
        selected = [name for name in linears if pattern.search(name)]
    elif spec == "last_layer":
        selected = [name for name in linears if name == "final_layer.linear"]
    elif spec.startswith("late_blocks:"):
        num_blocks = int(spec.split(":", 1)[1])
        depth = len(model.blocks)
        if not 0 < num_blocks <= depth:
            raise ValueError(f"late_blocks:{num_blocks} out of range for depth {depth}")
        late = {depth - 1 - i for i in range(num_blocks)}
        wanted = {f"blocks.{i}.{s}" for i in late for s in ("attn.proj", "mlp.fc2")}
        wanted.add("final_layer.linear")
        selected = [name for name in linears if name in wanted]
    else:
        raise ValueError(f"Unknown LoRA target spec {spec!r}")

    if not selected:
        raise ValueError(f"Spec {spec!r} matched no nn.Linear modules")
    return tuple(selected)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child, replacement)


def inject_lora_at_paths(
    model: nn.Module,
    paths: Sequence[str],
    *,
    rank: int = 4,
    alpha: float | None = None,
    prior_std: float = 1e-2,
    init_posterior_std: float = 1e-4,
) -> dict[str, VariationalLoRALinear]:
    """Freeze ``model`` and wrap the linear layers at ``paths`` in place."""
    freeze(model)
    adapters: dict[str, VariationalLoRALinear] = {}
    for path in paths:
        base = model.get_submodule(path)
        if isinstance(base, VariationalLoRALinear):
            adapters[path] = base
            continue
        adapter = VariationalLoRALinear(
            base,
            rank=rank,
            alpha=alpha,
            prior_std=prior_std,
            init_posterior_std=init_posterior_std,
        )
        _replace_submodule(model, path, adapter)
        adapters[path] = adapter
    return adapters


def inject_variational_lora(
    model: nn.Module,
    target_spec: str = "late_blocks:4",
    *,
    rank: int = 4,
    alpha: float | None = None,
    prior_std: float = 1e-2,
    init_posterior_std: float = 1e-4,
) -> dict[str, VariationalLoRALinear]:
    """Freeze ``model`` and wrap the layers selected by ``target_spec``."""
    return inject_lora_at_paths(
        model,
        select_lora_targets(model, target_spec),
        rank=rank,
        alpha=alpha,
        prior_std=prior_std,
        init_posterior_std=init_posterior_std,
    )


def enable_weight_sampling(adapters: Mapping[str, VariationalLoRALinear], flag: bool):
    for adapter in adapters.values():
        adapter.sample_weights = flag


class BayesianLoRAPosterior(WeightPosterior):
    """Posterior over the adapter weights injected by :func:`inject_variational_lora`.

    ``sample()`` returns concrete ``lora_A`` / ``lora_B`` tensors, so a draw can
    be pinned for an entire reverse trajectory via
    :meth:`WeightPosterior.use_parameters`.
    """

    def __init__(self, adapters: Mapping[str, VariationalLoRALinear]):
        if not adapters:
            raise ValueError("No adapters supplied")
        self._adapters = dict(adapters)
        names: list[str] = []
        for path in self._adapters:
            names.extend([f"{path}.lora_A", f"{path}.lora_B"])
        super().__init__(names)

    def mean(self) -> ParameterDict:
        params: ParameterDict = {}
        for path, adapter in self._adapters.items():
            params[f"{path}.lora_A"] = adapter.lora_A_mu.detach().clone()
            params[f"{path}.lora_B"] = adapter.lora_B_mu.detach().clone()
        return params

    def sample(self, index=None, generator=None) -> ParameterDict:
        params: ParameterDict = {}
        for path, adapter in self._adapters.items():
            for key, mu, sigma in (
                ("lora_A", adapter.lora_A_mu, adapter.lora_A_sigma),
                ("lora_B", adapter.lora_B_mu, adapter.lora_B_sigma),
            ):
                mu = mu.detach()
                noise = randn_like_from(mu, generator)
                params[f"{path}.{key}"] = mu + sigma.detach() * noise
        return params

    def kl_divergence(self) -> torch.Tensor:
        return sum(
            adapter.kl_divergence() for adapter in self._adapters.values()
        )  # type: ignore[return-value]

    def variational_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for adapter in self._adapters.values():
            params.extend(adapter.variational_parameters())
        return params

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["adapters"] = {
            path: {
                "lora_A_mu": adapter.lora_A_mu.detach().cpu(),
                "lora_A_rho": adapter.lora_A_rho.detach().cpu(),
                "lora_B_mu": adapter.lora_B_mu.detach().cpu(),
                "lora_B_rho": adapter.lora_B_rho.detach().cpu(),
                "rank": adapter.rank,
                "scaling": adapter.scaling,
                "prior_std": adapter.prior_std,
            }
            for path, adapter in self._adapters.items()
        }
        return state

    def load_into(self, adapters: Mapping[str, VariationalLoRALinear], state: Mapping):
        for path, adapter in adapters.items():
            saved = state["adapters"][path]
            for key in ("lora_A_mu", "lora_A_rho", "lora_B_mu", "lora_B_rho"):
                getattr(adapter, key).data.copy_(saved[key].to(adapter.lora_A_mu.device))
            adapter.lora_A.copy_(adapter.lora_A_mu.detach())
            adapter.lora_B.copy_(adapter.lora_B_mu.detach())


def load_lora_posterior(
    model: nn.Module, path: str | Path, *, device: str | torch.device = "cpu"
) -> BayesianLoRAPosterior:
    """Re-inject adapters into ``model`` and restore a saved LoRA posterior.

    A LoRA posterior cannot be loaded standalone the way a Laplace posterior
    can: its parameters live *inside* the model, so the adapters have to be
    rebuilt at the same module paths first.
    """
    state = torch.load(Path(path), map_location=device, weights_only=False)
    if state.get("kind") != "BayesianLoRAPosterior":
        raise ValueError(f"{path} does not hold a BayesianLoRAPosterior")

    saved_adapters = state["adapters"]
    paths = list(saved_adapters)
    first = saved_adapters[paths[0]]
    adapters = inject_lora_at_paths(
        model,
        paths,
        rank=first["rank"],
        alpha=first["scaling"] * first["rank"],
        prior_std=first["prior_std"],
    )
    posterior = BayesianLoRAPosterior(adapters)
    posterior.load_into(adapters, state)
    return posterior


@dataclass
class LoRAFitConfig:
    target_spec: str = "late_blocks:4"
    rank: int = 4
    alpha: float | None = None
    prior_std: float = 1e-2
    init_posterior_std: float = 1e-4
    steps: int = 2000
    batch_size: int = 8
    learning_rate: float = 1e-3
    kl_weight: float = 1.0
    """Tempering factor beta on the KL term; <1 gives a wider, less collapsed q."""
    dataset_size: int | None = None
    """Used to scale the KL per minibatch; defaults to steps * batch_size."""
    label_dropout_prob: float = 0.1
    log_every: int = 100
    seed: int = 0


def fit_variational_lora(
    model: nn.Module,
    diffusion,
    data: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: LoRAFitConfig | None = None,
    *,
    adapters: Mapping[str, VariationalLoRALinear] | None = None,
    progress: bool = True,
) -> BayesianLoRAPosterior:
    """Fit the mean-field posterior over the adapters with Bayes-by-Backprop.

    ``data`` must be an iterable of ``(latents, labels)`` that repeats forever
    (see ``data.imagenet_latent_loader(..., repeat=True)``); it is consumed for
    ``config.steps`` minibatches.
    """
    config = config or LoRAFitConfig()
    device = next(model.parameters()).device
    torch.manual_seed(config.seed)

    if adapters is None:
        adapters = inject_variational_lora(
            model,
            config.target_spec,
            rank=config.rank,
            alpha=config.alpha,
            prior_std=config.prior_std,
            init_posterior_std=config.init_posterior_std,
        )
    posterior = BayesianLoRAPosterior(adapters)

    optimizer = torch.optim.Adam(
        posterior.variational_parameters(), lr=config.learning_rate
    )
    dataset_size = config.dataset_size or (config.steps * config.batch_size)
    uncond = unconditional_class(model)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    enable_weight_sampling(adapters, True)
    model.eval()
    iterator = iter(data)
    history: list[dict[str, float]] = []

    step_range = range(config.steps)
    if progress:
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415

            step_range = tqdm(step_range, desc="Bayesian LoRA")
        except ImportError:
            pass

    with torch.enable_grad():
        for step in step_range:
            try:
                latents, labels = next(iterator)
            except StopIteration as error:
                raise ValueError(
                    f"Data iterator exhausted at step {step}; pass a repeating loader."
                ) from error

            latents = latents.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)

            if config.label_dropout_prob > 0:
                drop = (
                    torch.rand(labels.shape, generator=generator)
                    < config.label_dropout_prob
                ).to(device)
                labels = torch.where(drop, torch.full_like(labels, uncond), labels)

            t = torch.randint(
                0, diffusion.num_timesteps, (latents.shape[0],), generator=generator
            ).to(device)

            terms = diffusion.training_losses(
                model, latents, t, model_kwargs={"y": labels}
            )
            num_dims = latents[0].numel()
            nll = 0.5 * num_dims * terms["mse"].mean()
            kl = posterior.kl_divergence()
            loss = nll + config.kl_weight * kl / dataset_size

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if config.log_every and step % config.log_every == 0:
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "nll": float(nll.detach()),
                        "kl": float(kl.detach()),
                    }
                )

    enable_weight_sampling(adapters, False)
    # Buffers hold the posterior mean until a draw is installed, so the model is
    # immediately usable as the deterministic "posterior-mean" generator.
    for adapter in adapters.values():
        adapter.lora_A.copy_(adapter.lora_A_mu.detach())
        adapter.lora_B.copy_(adapter.lora_B_mu.detach())

    posterior.history = history  # type: ignore[attr-defined]
    return posterior


def lora_deep_ensemble(
    posterior: BayesianLoRAPosterior,
    num_members: int,
    *,
    seed: int = 0,
):
    """Freeze ``num_members`` posterior draws into a finite ensemble.

    Handy when you want the *same* set of weight draws reused across many
    prompts or many noise seeds, e.g. for the nested decomposition where the
    weight draws must be shared across the inner noise loop.
    """
    from .posteriors import DeepEnsemblePosterior  # noqa: PLC0415

    generator = torch.Generator(device="cpu").manual_seed(seed)
    members = [posterior.sample(generator=generator) for _ in range(num_members)]
    return DeepEnsemblePosterior(members)
