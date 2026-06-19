"""Plotting helpers for the local satire study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis.local_satire_analysis import RUBRIC_COLUMNS, aggregate_scores
from src.analysis.plotting import setup_style


def plot_average_total_score(scores_path: str, output_path: str) -> None:
    """Bar chart of average total score by model and prompt condition."""
    df = aggregate_scores(scores_path)
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="model",
        y="total_score",
        hue="prompt_condition",
        errorbar="se",
        ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Average total score")
    ax.set_title("Local satire total score by model and condition")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output_path)
    plt.close(fig)


def plot_rubric_dimensions(scores_path: str, output_path: str) -> None:
    """Grouped bars for the five rubric dimensions."""
    df = aggregate_scores(scores_path)
    long_df = df.melt(
        id_vars=["model", "prompt_condition"],
        value_vars=RUBRIC_COLUMNS,
        var_name="dimension",
        value_name="score",
    )
    long_df["model_condition"] = (
        long_df["model"].astype(str) + "\n" + long_df["prompt_condition"].astype(str)
    )

    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=long_df,
        x="model_condition",
        y="score",
        hue="dimension",
        errorbar="se",
        ax=ax,
    )
    ax.set_xlabel("Model / condition")
    ax.set_ylabel("Average score")
    ax.set_title("Local satire rubric dimensions")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output_path)
    plt.close(fig)


def plot_local_satire_scores(
    scores_path: str,
    total_out: str,
    dimensions_out: str | None = None,
) -> None:
    """Write local satire score plots."""
    Path(total_out).parent.mkdir(parents=True, exist_ok=True)
    plot_average_total_score(scores_path, total_out)
    if dimensions_out:
        Path(dimensions_out).parent.mkdir(parents=True, exist_ok=True)
        plot_rubric_dimensions(scores_path, dimensions_out)
