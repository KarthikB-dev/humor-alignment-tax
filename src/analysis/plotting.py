"""Plotting utilities for the humor alignment tax paper."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (activates styles on import)
import seaborn as sns


def setup_style():
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_distribution_comparison(
    base_values: np.ndarray,
    aligned_values: np.ndarray,
    metric_name: str,
    output_path: str,
) -> None:
    """Overlapping density plots: base vs aligned model."""
    setup_style()
    fig, ax = plt.subplots()
    sns.kdeplot(base_values, ax=ax, label="Base", fill=True, alpha=0.4)
    sns.kdeplot(aligned_values, ax=ax, label="Aligned (RLHF)", fill=True, alpha=0.4)
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {metric_name}")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_inverted_u(
    metric_values: np.ndarray,
    humor_scores: np.ndarray,
    metric_name: str,
    humor_dimension: str,
    output_path: str,
    n_bins: int = 15,
) -> None:
    """Binned scatter + quadratic fit for inverted-U visualization."""
    setup_style()
    mask = np.isfinite(metric_values) & np.isfinite(humor_scores)
    x, y = metric_values[mask], humor_scores[mask]

    fig, ax = plt.subplots()

    # Binned means
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers, bin_means, bin_sems = [], [], []
    for i in range(n_bins):
        in_bin = (x >= bins[i]) & (x < bins[i + 1])
        if in_bin.sum() >= 3:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(y[in_bin].mean())
            bin_sems.append(y[in_bin].std() / np.sqrt(in_bin.sum()))

    ax.errorbar(
        bin_centers, bin_means, yerr=bin_sems, fmt="o", capsize=3, label="Binned mean"
    )

    # Quadratic fit
    coeffs = np.polyfit(x, y, 2)
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = np.polyval(coeffs, x_smooth)
    ax.plot(x_smooth, y_smooth, "--", color="red", label="Quadratic fit", linewidth=1.5)

    ax.set_xlabel(metric_name)
    ax.set_ylabel(f"Humor ({humor_dimension})")
    ax.set_title(f"{humor_dimension} vs {metric_name}")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_temperature_sweep(
    generated_data: list[dict],
    metric_name: str,
    output_path: str,
) -> None:
    """Box plot of a metric across temperature configs."""
    setup_style()

    configs = sorted(set(d["config"] for d in generated_data))
    data_by_config = {c: [] for c in configs}
    for d in generated_data:
        if metric_name in d:
            data_by_config[d["config"]].append(d[metric_name])

    fig, ax = plt.subplots()
    positions = range(len(configs))
    bp = ax.boxplot(
        [data_by_config[c] for c in configs],
        positions=positions,
        patch_artist=True,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name} across decoding strategies")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")
