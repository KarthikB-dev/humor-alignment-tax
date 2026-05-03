"""Alignment comparison: statistical tests for distributional shift
between base and aligned model outputs."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class AlignmentShiftResult:
    """Result of comparing a distributional metric between base and aligned models."""

    metric_name: str
    base_mean: float
    base_std: float
    aligned_mean: float
    aligned_std: float
    mean_shift: float  # aligned - base
    cohens_d: float  # effect size
    t_statistic: float
    p_value: float
    ks_statistic: float  # Kolmogorov-Smirnov
    ks_p_value: float
    n_base: int
    n_aligned: int


def compare_distributions(
    base_values: np.ndarray,
    aligned_values: np.ndarray,
    metric_name: str = "",
) -> AlignmentShiftResult:
    """
    Compare a distributional metric between base and aligned model.

    Uses Welch's t-test (unequal variance) and KS test.
    """
    base = base_values[np.isfinite(base_values)]
    aligned = aligned_values[np.isfinite(aligned_values)]

    base_mean = float(base.mean())
    base_std = float(base.std())
    aligned_mean = float(aligned.mean())
    aligned_std = float(aligned.std())

    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(base, aligned, equal_var=False)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(base, aligned)

    # Cohen's d
    pooled_std = np.sqrt((base.std() ** 2 + aligned.std() ** 2) / 2)
    cohens_d = (aligned_mean - base_mean) / pooled_std if pooled_std > 0 else 0.0

    return AlignmentShiftResult(
        metric_name=metric_name,
        base_mean=base_mean,
        base_std=base_std,
        aligned_mean=aligned_mean,
        aligned_std=aligned_std,
        mean_shift=aligned_mean - base_mean,
        cohens_d=float(cohens_d),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_p),
        n_base=len(base),
        n_aligned=len(aligned),
    )


def run_alignment_comparison(
    base_metrics: dict[str, np.ndarray],
    aligned_metrics: dict[str, np.ndarray],
) -> list[AlignmentShiftResult]:
    """Compare all metrics between base and aligned model."""
    results = []
    for metric_name in base_metrics:
        if metric_name not in aligned_metrics:
            continue
        result = compare_distributions(
            base_metrics[metric_name],
            aligned_metrics[metric_name],
            metric_name=metric_name,
        )
        results.append(result)
    return results


def print_comparison_table(results: list[AlignmentShiftResult]) -> None:
    """Print a formatted table of alignment comparison results."""
    header = (
        f"{'Metric':<25} {'Base μ':>8} {'Aligned μ':>10} "
        f"{'Shift':>8} {'Cohen d':>8} {'p-value':>10}"
    )
    print(header)
    print("-" * 75)
    for r in results:
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
        print(
            f"{r.metric_name:<25} {r.base_mean:>8.4f} {r.aligned_mean:>10.4f} "
            f"{r.mean_shift:>8.4f} {r.cohens_d:>8.4f} {r.p_value:>9.2e} {sig}"
        )


def save_results(results: list[AlignmentShiftResult], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved {len(results)} alignment comparison results to {output_path}")
