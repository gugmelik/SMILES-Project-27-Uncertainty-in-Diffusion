"""Approximate posteriors q(theta) over a subset of the denoiser's weights.

A standard diffusion model is trained to a single point estimate ``theta_MAP``,
so drawing many images from it only exposes *aleatoric* variability (the initial
latent and the reverse-process noise). To expose *epistemic* uncertainty we need
a distribution over weights and a way to run the sampler under a draw from it.

Every posterior here exposes the same three-call contract::

    params = posterior.sample(index=m)
    with posterior.use_parameters(model, params):
        ...  # model now behaves like the m-th posterior draw

The subset of weights each posterior covers is fixed at construction time (see
``dit.select_parameters``); all remaining weights stay frozen at their MAP value.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from .utils import randint, randn_like_from

ParameterDict = dict[str, torch.Tensor]


def _tensor_lookup(model: nn.Module) -> dict[str, torch.Tensor]:
    """Name -> tensor for both parameters and buffers of ``model``."""
    lookup = dict(model.named_parameters())
    lookup.update(model.named_buffers())
    return lookup


class WeightPosterior(ABC):
    """Base class for an approximate posterior over a named subset of weights."""

    def __init__(self, parameter_names: Sequence[str]):
        self.parameter_names: tuple[str, ...] = tuple(parameter_names)

    @property
    def num_members(self) -> int | None:
        """Number of distinct draws, or ``None`` for a continuous posterior."""
        return None

    @property
    def num_parameters(self) -> int:
        return sum(tensor.numel() for tensor in self.mean().values())

    @abstractmethod
    def mean(self) -> ParameterDict:
        """Posterior mean weights (the 'default' model)."""

    @abstractmethod
    def sample(
        self,
        index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ParameterDict:
        """Draw ``theta_m ~ q(theta)``.

        ``index`` identifies the draw for posteriors with a finite number of
        members (deep ensembles); continuous posteriors ignore it.
        """

    @contextlib.contextmanager
    def use_parameters(self, model: nn.Module, params: Mapping[str, torch.Tensor]):
        """Temporarily install ``params`` into ``model``, restoring them on exit."""
        lookup = _tensor_lookup(model)
        missing = set(params) - set(lookup)
        if missing:
            raise KeyError(f"Model has no tensors named: {sorted(missing)}")

        saved: ParameterDict = {}
        try:
            for name, value in params.items():
                target = lookup[name]
                if value.shape != target.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: posterior {tuple(value.shape)} "
                        f"vs model {tuple(target.shape)}"
                    )
                saved[name] = target.data
                target.data = value.to(device=target.device, dtype=target.dtype)
            yield model
        finally:
            for name, value in saved.items():
                lookup[name].data = value

    @contextlib.contextmanager
    def sampled(
        self,
        model: nn.Module,
        index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[nn.Module]:
        """Shorthand for ``use_parameters(model, sample(index, generator))``."""
        with self.use_parameters(model, self.sample(index=index, generator=generator)):
            yield model

    def effective_num_samples(self, requested: int) -> int:
        """Clamp a requested draw count to the number of distinct draws available."""
        if self.num_members is None:
            return requested
        return min(requested, self.num_members)

    def state_dict(self) -> dict:
        return {"kind": type(self).__name__, "parameter_names": self.parameter_names}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)


class MAPPosterior(WeightPosterior):
    """Degenerate posterior: a point mass at the pretrained weights.

    Useful as a control — every estimator run against it must report exactly zero
    epistemic uncertainty, which makes it a good sanity check for the pipeline.
    """

    def __init__(self, model: nn.Module, parameter_names: Sequence[str]):
        super().__init__(parameter_names)
        lookup = _tensor_lookup(model)
        self._mean = {
            name: lookup[name].detach().clone() for name in self.parameter_names
        }

    def mean(self) -> ParameterDict:
        return {name: value.clone() for name, value in self._mean.items()}

    def sample(self, index=None, generator=None) -> ParameterDict:
        return self.mean()


class DiagonalGaussianPosterior(WeightPosterior):
    """Mean-field Gaussian ``q(theta) = N(mean, diag(std**2))``.

    This is what a diagonal Laplace approximation produces (see ``laplace.py``),
    but it is deliberately agnostic about where the standard deviations came
    from, so a diagonal SWAG or variational fit can reuse it.

    ``scale`` is a global temperature on the posterior standard deviation. A
    Laplace approximation to a heavily over-parameterised denoiser is rarely
    calibrated out of the box, so leaving one knob to tune against a held-out
    signal (see ``validation.py``) is more honest than pretending it is exact.
    """

    def __init__(
        self,
        parameter_names: Sequence[str],
        mean: Mapping[str, torch.Tensor],
        std: Mapping[str, torch.Tensor],
        *,
        scale: float = 1.0,
    ):
        super().__init__(parameter_names)
        self._mean = {name: mean[name].detach().clone() for name in self.parameter_names}
        self._std = {name: std[name].detach().clone() for name in self.parameter_names}
        self.scale = float(scale)

    def mean(self) -> ParameterDict:
        return {name: value.clone() for name, value in self._mean.items()}

    @property
    def std(self) -> ParameterDict:
        return {name: value.clone() for name, value in self._std.items()}

    def sample(self, index=None, generator=None) -> ParameterDict:
        params: ParameterDict = {}
        for name in self.parameter_names:
            mean = self._mean[name]
            noise = randn_like_from(mean, generator)
            params[name] = mean + self.scale * self._std[name] * noise
        return params

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(mean=self._mean, std=self._std, scale=self.scale)
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "DiagonalGaussianPosterior":
        return cls(
            state["parameter_names"],
            state["mean"],
            state["std"],
            scale=state.get("scale", 1.0),
        )

    def to(self, device: str | torch.device) -> "DiagonalGaussianPosterior":
        self._mean = {k: v.to(device) for k, v in self._mean.items()}
        self._std = {k: v.to(device) for k, v in self._std.items()}
        return self


class DeepEnsemblePosterior(WeightPosterior):
    """Finite posterior supported on a set of independently fitted weight sets.

    Deep ensembles are the reference baseline the Bayesian approximations should
    be checked against: if the Laplace or variational posterior reports far less
    disagreement than an ensemble on the same inputs, the approximation is too
    tight to trust.
    """

    def __init__(self, members: Sequence[Mapping[str, torch.Tensor]]):
        if not members:
            raise ValueError("DeepEnsemblePosterior needs at least one member")
        names = tuple(members[0])
        for i, member in enumerate(members[1:], start=1):
            if tuple(member) != names:
                raise ValueError(f"Member {i} has different parameter names")
        super().__init__(names)
        self._members = [
            {name: tensor.detach().clone() for name, tensor in member.items()}
            for member in members
        ]

    @property
    def num_members(self) -> int:
        return len(self._members)

    def mean(self) -> ParameterDict:
        return {
            name: torch.stack([m[name] for m in self._members]).mean(dim=0)
            for name in self.parameter_names
        }

    def sample(self, index=None, generator=None) -> ParameterDict:
        if index is None:
            index = randint(len(self._members), generator=generator)
        member = self._members[index % len(self._members)]
        return {name: tensor.clone() for name, tensor in member.items()}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["members"] = self._members
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "DeepEnsemblePosterior":
        return cls(state["members"])


_REGISTRY = {
    "DiagonalGaussianPosterior": DiagonalGaussianPosterior,
    "DeepEnsemblePosterior": DeepEnsemblePosterior,
}


def load_posterior(
    path: str | Path, device: str | torch.device = "cpu"
) -> WeightPosterior:
    """Load a posterior saved with :meth:`WeightPosterior.save`."""
    state = torch.load(Path(path), map_location=device, weights_only=False)
    kind = state["kind"]
    if kind not in _REGISTRY:
        raise ValueError(f"Cannot load posterior of type {kind!r}")
    return _REGISTRY[kind].from_state_dict(state)
