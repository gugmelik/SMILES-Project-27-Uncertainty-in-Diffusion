"""Small shared helpers: seeded randomness that survives device mismatches."""

from __future__ import annotations

import torch


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def randn_like_from(
    tensor: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    """``torch.randn_like`` that accepts a generator on a different device.

    ``torch.randn(..., generator=g, device=d)`` requires ``g.device == d``. Using
    a CPU generator everywhere keeps draws reproducible across CPU/GPU runs,
    which matters here: the common-random-numbers trick in the epistemic /
    aleatoric decomposition is only valid if the noise really is identical.
    """
    if generator is None:
        return torch.randn_like(tensor)
    if generator.device.type == tensor.device.type:
        return torch.randn(
            tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype
        )
    drawn = torch.randn(
        tensor.shape, generator=generator, device=generator.device, dtype=torch.float32
    )
    return drawn.to(device=tensor.device, dtype=tensor.dtype)


def randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generator-safe ``torch.randn`` for an explicit shape and device."""
    device = torch.device(device)
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    if generator.device.type == device.type:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    drawn = torch.randn(
        shape, generator=generator, device=generator.device, dtype=torch.float32
    )
    return drawn.to(device=device, dtype=dtype)


def randint(
    high: int, *, generator: torch.Generator | None = None, device: str = "cpu"
) -> int:
    if generator is None:
        return int(torch.randint(high, (1,)).item())
    return int(
        torch.randint(high, (1,), generator=generator, device=generator.device).item()
    )
