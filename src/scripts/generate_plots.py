"""Generate all figures for the alignment-tax presentation."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

OUT = Path("presentation/figures")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Llama-3.2-1B", "Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct-safety"]
LABELS = ["Base", "Instruct", "Safety-finetuned"]
COLORS = ["#2196F3", "#FF5722", "#4CAF50"]
METRICS = ["surprisals", "entropies", "distances"]
METRIC_LABELS = ["Punchline Surprisal", "Pre-punchline Entropy", "Embedding Distance"]
METRIC_SHORT  = ["Surprisal", "Entropy", "Distance"]

def load_model(name):
    return np.load(f"data/metrics/{name}.npz", allow_pickle=True)


# ── Plot 1: grouped bar chart of means ─────────────────────────────────────
def plot_means():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Metric Means Across Alignment Levels", fontsize=14, fontweight="bold", y=1.01)

    for ax, metric, label in zip(axes, METRICS, METRIC_LABELS):
        means, sems = [], []
        for m in MODELS:
            d = load_model(m)[metric].astype(float)
            means.append(d.mean())
            sems.append(stats.sem(d))

        x = np.arange(len(MODELS))
        bars = ax.bar(x, means, yerr=sems, color=COLORS, width=0.6,
                      capsize=4, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Mean value")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # annotate bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sems) * 0.5,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "metric_means.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "metric_means.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved metric_means")


# ── Plot 2: violin distributions ───────────────────────────────────────────
def plot_distributions():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Metric Distributions Across Models", fontsize=14, fontweight="bold", y=1.01)

    for ax, metric, label in zip(axes, METRICS, METRIC_LABELS):
        data = [load_model(m)[metric].astype(float) for m in MODELS]
        parts = ax.violinplot(data, positions=range(len(MODELS)), showmedians=True,
                               showextrema=False)

        for pc, color in zip(parts["bodies"], COLORS):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Value")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT / "distributions.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "distributions.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved distributions")


# ── Plot 3: shift arrows (base → instruct → safety) ────────────────────────
def plot_shifts():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Metric Progression: Base → Instruct → Safety-finetuned",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, metric, label, short in zip(axes, METRICS, METRIC_LABELS, METRIC_SHORT):
        means = [load_model(m)[metric].astype(float).mean() for m in MODELS]
        xs = [0, 1, 2]

        ax.plot(xs, means, "o-", color="#333333", linewidth=2, zorder=3, markersize=10)
        for x, y, color, lbl in zip(xs, means, COLORS, LABELS):
            ax.scatter([x], [y], color=color, s=120, zorder=4)
            ax.text(x, y + (max(means) - min(means)) * 0.05, f"{y:.4f}",
                    ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(LABELS, rotation=10, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(short)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # annotate shifts
        for i in range(len(means) - 1):
            shift = means[i + 1] - means[i]
            mid_x = (xs[i] + xs[i + 1]) / 2
            mid_y = (means[i] + means[i + 1]) / 2
            color = "#C62828" if shift > 0 else "#1B5E20"
            ax.annotate(f"{shift:+.4f}", xy=(mid_x, mid_y),
                        fontsize=8, color=color, ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUT / "shifts.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "shifts.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved shifts")


# ── Plot 4: temperature sweep ───────────────────────────────────────────────
def plot_temperature():
    ORDER = ["low_temp", "mid_temp", "default", "high_temp"]
    TEMP_LABELS = ["Low\n(T=0.3)", "Mid\n(T=0.7)", "Default\n(T=1.0)", "High\n(T=1.4)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Temperature Sweep: Surprisal & Entropy vs. Decoding Temperature",
                 fontsize=13, fontweight="bold", y=1.02)

    model_files = [
        ("Llama-3.2-1B",          "Base",    COLORS[0]),
        ("Llama-3.2-1B-Instruct", "Instruct", COLORS[1]),
    ]

    for ax, metric_key, metric_label in zip(axes,
            ["surprisal", "entropy"],
            ["Mean Punchline Surprisal", "Mean Pre-punchline Entropy"]):
        for fname, lbl, color in model_files:
            with open(f"data/generated/generated_{fname}.json") as f:
                jokes = json.load(f)

            by_config = {}
            for j in jokes:
                cfg = j["config"]
                by_config.setdefault(cfg, []).append(j[metric_key])

            means = [np.mean(by_config.get(c, [np.nan])) for c in ORDER]
            sems  = [stats.sem(by_config.get(c, [np.nan])) for c in ORDER]
            xs = range(len(ORDER))
            ax.errorbar(xs, means, yerr=sems, label=lbl, color=color,
                        marker="o", linewidth=2, capsize=4, markersize=7)

        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(TEMP_LABELS, fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT / "temperature_sweep.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "temperature_sweep.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved temperature_sweep")


# ── Plot 5: safety fine-tuning loss curve ──────────────────────────────────
def plot_training_loss():
    # Values from training output log
    train_steps  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    train_loss   = [3.718, 1.37, 0.92, 0.694, 0.545, 0.476, 0.447, 0.428, 0.432, 0.422, 0.422,
                    0.351, 0.353, 0.339, 0.348, 0.339]
    eval_epochs  = [1, 2, 3]
    eval_loss    = [0.493, 0.455, 0.453]
    # Map eval epochs to approximate steps (55 steps/epoch)
    eval_steps   = [55, 110, 165]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_steps, train_loss, color=COLORS[2], linewidth=2, label="Train loss", zorder=3)
    ax.scatter(eval_steps, eval_loss, color="#FF9800", s=80, zorder=4,
               label="Eval loss (end of epoch)", edgecolors="white", linewidth=1)

    for epoch, step, loss in zip(eval_epochs, eval_steps, eval_loss):
        ax.axvline(step, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.text(step + 2, loss + 0.05, f"Epoch {epoch}\n{loss:.3f}",
                fontsize=8, color="gray")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Safety Fine-tuning: Training Loss (AdvBench + HarmBench refusals)", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT / "training_loss.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "training_loss.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved training_loss")


if __name__ == "__main__":
    plot_means()
    plot_distributions()
    plot_shifts()
    plot_temperature()
    plot_training_loss()
    print(f"\nAll figures saved to {OUT}/")
