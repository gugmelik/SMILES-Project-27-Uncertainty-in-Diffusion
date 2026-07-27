"""Does the uncertainty score actually mean anything?

A heatmap that looks plausible is not evidence. These are the checks that
separate a useful score from a pretty one:

* **Correlation** with a per-sample error or quality measure.
* **AUROC** for separating deliberately corrupted / failed generations from
  clean ones.
* **Selective generation**: discard the most uncertain samples and verify that
  the average quality of what remains actually improves.

Everything here is numpy-only so it can run without a GPU session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayLike = "np.ndarray | list[float]"

# np.trapz was renamed in NumPy 2.0; the notebooks pin neither version.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _as_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Empty input")
    return array


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so tied values do not bias the rank correlation."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def pearson_corr(x, y) -> float:
    x, y = _as_array(x), _as_array(y)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt((x**2).sum() * (y**2).sum())
    return float(x @ y / denominator) if denominator > 0 else float("nan")


def spearman_corr(x, y) -> float:
    return pearson_corr(_rank(_as_array(x)), _rank(_as_array(y)))


def auroc(scores, labels) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    ``labels`` is 1 for the class the score is supposed to be *high* for
    (e.g. corrupted or out-of-distribution generations).
    """
    scores = _as_array(scores)
    labels = _as_array(labels)
    positive = labels > 0.5
    num_positive = int(positive.sum())
    num_negative = int((~positive).sum())
    if num_positive == 0 or num_negative == 0:
        return float("nan")

    ranks = _rank(scores)
    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative)
    )


@dataclass
class SelectiveCurve:
    """Quality of the retained set as the most uncertain samples are dropped."""

    coverage: np.ndarray
    """Fraction of samples kept, from most-certain-only to everything."""
    value: np.ndarray
    """Mean quality (or risk) over the retained set at each coverage."""
    area: float
    """Area under the curve; interpretation follows ``value``."""
    baseline: float
    """Value at full coverage — the no-filtering reference."""

    def improvement_at(self, coverage: float) -> float:
        index = int(np.searchsorted(self.coverage, coverage, side="left"))
        index = min(max(index, 0), len(self.value) - 1)
        return float(self.value[index] - self.baseline)


def selective_curve(uncertainty, value, *, higher_is_better: bool = True) -> SelectiveCurve:
    """Sort by ascending uncertainty and accumulate the mean ``value``.

    With a useful uncertainty score and ``higher_is_better=True``, the curve
    starts high (the most confident samples are the best ones) and decays toward
    the dataset mean. A flat curve means the score carries no information about
    quality.
    """
    uncertainty = _as_array(uncertainty)
    value = _as_array(value)
    if uncertainty.shape != value.shape:
        raise ValueError(f"Shape mismatch: {uncertainty.shape} vs {value.shape}")

    order = np.argsort(uncertainty, kind="mergesort")
    ordered = value[order]
    counts = np.arange(1, len(ordered) + 1, dtype=np.float64)
    running_mean = np.cumsum(ordered) / counts
    coverage = counts / len(ordered)

    return SelectiveCurve(
        coverage=coverage,
        value=running_mean,
        area=float(_trapezoid(running_mean, coverage)),
        baseline=float(ordered.mean()),
    )


def risk_coverage_curve(uncertainty, error) -> SelectiveCurve:
    """Selective curve for an error signal; lower area is better."""
    return selective_curve(uncertainty, error, higher_is_better=False)


@dataclass
class UncertaintyReport:
    """Summary of every validation signal available for one uncertainty score."""

    name: str
    num_samples: int
    pearson: float | None = None
    spearman: float | None = None
    auroc_corrupted: float | None = None
    aurc: float | None = None
    selective_gain_at_50: float | None = None
    curves: dict = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        parts = [f"{self.name} (n={self.num_samples})"]
        if self.spearman is not None:
            parts.append(f"spearman={self.spearman:+.3f}")
        if self.pearson is not None:
            parts.append(f"pearson={self.pearson:+.3f}")
        if self.auroc_corrupted is not None:
            parts.append(f"auroc={self.auroc_corrupted:.3f}")
        if self.aurc is not None:
            parts.append(f"aurc={self.aurc:.4f}")
        if self.selective_gain_at_50 is not None:
            parts.append(f"gain@50%={self.selective_gain_at_50:+.4f}")
        return " | ".join(parts)


def evaluate_uncertainty(
    uncertainty,
    *,
    name: str = "uncertainty",
    quality=None,
    error=None,
    corrupted_labels=None,
) -> UncertaintyReport:
    """Run every applicable validation check for one uncertainty score.

    ``quality`` should be higher-is-better (CLIP score, human rating);
    ``error`` lower-is-better (reconstruction error, LPIPS to a reference);
    ``corrupted_labels`` is 1 for samples known to be bad.
    """
    uncertainty = _as_array(uncertainty)
    report = UncertaintyReport(name=name, num_samples=len(uncertainty))

    target = error if error is not None else quality
    if target is not None:
        report.pearson = pearson_corr(uncertainty, target)
        report.spearman = spearman_corr(uncertainty, target)

    if corrupted_labels is not None:
        report.auroc_corrupted = auroc(uncertainty, corrupted_labels)

    if error is not None:
        curve = risk_coverage_curve(uncertainty, error)
        report.aurc = curve.area
        report.selective_gain_at_50 = curve.improvement_at(0.5)
        report.curves["risk_coverage"] = curve
    elif quality is not None:
        curve = selective_curve(uncertainty, quality)
        report.selective_gain_at_50 = curve.improvement_at(0.5)
        report.curves["selective_quality"] = curve

    return report
