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


# ── Humor-score figures (Gemini judge over corpus + generated) ──────────────
JDIR = Path("data/judgments/gemini_bulk")
GEN_TIERS = [
    ("Llama-3.2-1B", "Base", COLORS[0]),
    ("Llama-3.2-1B-Instruct", "Instruct", COLORS[1]),
    ("Llama-3.2-1B-Instruct-safety", "Safety-finetuned", COLORS[2]),
]
TEMP_ORDER = ["low_temp", "mid_temp", "default", "high_temp", "very_high_temp"]
TEMP_VALS = [0.3, 0.7, 1.0, 1.3, 1.6]


def _load_judgments(path):
    with open(path) as f:
        return json.load(f)


def _overall(entries):
    return np.array(
        [e["judge_scores"]["overall"] for e in entries if "overall" in e.get("judge_scores", {})],
        dtype=float,
    )


# ── Plot 6: humor scores by alignment tier (the headline) ──────────────────
def plot_humor_by_tier():
    corpus = _overall(_load_judgments(JDIR / "judgments_corpus.json"))
    tiers = [("Corpus\n(human)", corpus, "#9C27B0")]
    for fname, lbl, color in GEN_TIERS:
        tiers.append((lbl, _overall(_load_judgments(JDIR / f"judgments_generated_{fname}.json")), color))

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(tiers))
    means = [t[1].mean() for t in tiers]
    sems = [stats.sem(t[1]) for t in tiers]
    bars = ax.bar(xs, means, yerr=sems, color=[t[2] for t in tiers], width=0.62,
                  capsize=4, edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[0] for t in tiers], fontsize=10)
    ax.set_ylabel("Mean overall humor (LLM-judge, 1–10)")
    ax.set_title("Humor Quality by Alignment Tier (Gemini judge)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for bar, m, n in zip(bars, means, [len(t[1]) for t in tiers]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{m:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    # flag the safety refusal effect
    ax.text(3, 1.0, "≈74% refusals", ha="center", fontsize=8, style="italic", color="white")

    plt.tight_layout()
    plt.savefig(OUT / "humor_by_tier.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "humor_by_tier.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved humor_by_tier")


# ── Plot 7: humor vs decoding temperature (Hypothesis 4) ───────────────────
def plot_humor_temperature():
    fig, ax = plt.subplots(figsize=(8, 5))
    for fname, lbl, color in GEN_TIERS:
        entries = _load_judgments(JDIR / f"judgments_generated_{fname}.json")
        by_cfg = {}
        for e in entries:
            if "overall" in e.get("judge_scores", {}):
                by_cfg.setdefault(e["config"], []).append(e["judge_scores"]["overall"])
        means = [np.mean(by_cfg[c]) if by_cfg.get(c) else np.nan for c in TEMP_ORDER]
        sems = [stats.sem(by_cfg[c]) if by_cfg.get(c) else 0 for c in TEMP_ORDER]
        ax.errorbar(TEMP_VALS, means, yerr=sems, label=lbl, color=color,
                    marker="o", linewidth=2, capsize=4, markersize=7)

    ax.set_xlabel("Decoding temperature")
    ax.set_ylabel("Mean overall humor (LLM-judge, 1–10)")
    ax.set_title("Humor vs. Temperature — no moderate-temp peak (Hypothesis 4)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT / "humor_temperature.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "humor_temperature.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved humor_temperature")


# ── Plot 8: generated-joke inverted-U fits (instruct tier) ─────────────────
def plot_generated_inverted_u():
    entries = _load_judgments(JDIR / "judgments_generated_Llama-3.2-1B-Instruct.json")
    rows = [(e.get("surprisal"), e.get("entropy"), e["judge_scores"]["overall"])
            for e in entries if "overall" in e.get("judge_scores", {})
            and isinstance(e.get("surprisal"), (int, float))
            and isinstance(e.get("entropy"), (int, float))]
    surp = np.array([r[0] for r in rows]); ent = np.array([r[1] for r in rows])
    hum = np.array([r[2] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Generated-joke humor vs. distributional metrics — Instruct "
                 "(no inverted-U)", fontsize=12, fontweight="bold", y=1.01)
    for ax, x, xlabel in zip(axes, [surp, ent], ["Punchline surprisal", "Pre-punchline entropy"]):
        ax.scatter(x, hum, s=14, alpha=0.35, color=COLORS[1], edgecolors="none")
        # quadratic fit line
        a, b, c = np.polyfit(x, hum, 2)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, a * xx**2 + b * xx + c, color="#222", linewidth=2,
                label=f"quad fit (a={a:+.3f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Overall humor (1–10)")
        ax.set_ylim(0, 10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT / "generated_inverted_u.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(OUT / "generated_inverted_u.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved generated_inverted_u")


if __name__ == "__main__":
    plot_means()
    plot_distributions()
    plot_shifts()
    plot_temperature()
    plot_training_loss()
    plot_humor_by_tier()
    plot_humor_temperature()
    plot_generated_inverted_u()
    print(f"\nAll figures saved to {OUT}/")
