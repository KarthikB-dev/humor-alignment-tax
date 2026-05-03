"""Run the full analysis pipeline: inverted-U fitting and alignment comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.alignment_comparison import (
    print_comparison_table,
    run_alignment_comparison,
    save_results as save_alignment_results,
)
from src.analysis.inverted_u import (
    run_inverted_u_analysis,
    save_results as save_inverted_u_results,
)


def load_metrics(metrics_dir: str, model_name: str) -> dict[str, np.ndarray]:
    """Load extracted metrics for a model from npz."""
    path = Path(metrics_dir) / f"{model_name}.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "punchline_surprisal": data["surprisals"],
        "pre_punchline_entropy": data["entropies"],
        "embedding_distance": data["distances"],
    }


def load_judgments(judgments_dir: str, model_name: str) -> dict[str, np.ndarray]:
    """Load humor judgments and return arrays for each dimension."""
    path = Path(judgments_dir) / f"judgments_{model_name}.json"
    with open(path) as f:
        judgments = json.load(f)

    dims = ["unexpectedness", "cleverness", "amusement", "overall"]
    result = {}
    for dim in dims:
        vals = []
        for j in judgments:
            if dim in j:
                vals.append(j[dim])
            else:
                vals.append(np.nan)
        result[dim] = np.array(vals, dtype=np.float32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inverted-U and alignment comparison analyses."
    )
    parser.add_argument("--metrics_dir", default="data/metrics")
    parser.add_argument("--judgments_dir", default="data/judgments")
    parser.add_argument("--output_dir", default="data/analysis")
    parser.add_argument(
        "--base_model",
        default="Llama-3.2-1B",
        help="Base model name (filename stem in metrics_dir)",
    )
    parser.add_argument(
        "--aligned_model",
        default="Llama-3.2-1B-Instruct",
        help="Aligned model name (filename stem in metrics_dir)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading metrics and judgments...", flush=True)
    base_metrics = load_metrics(args.metrics_dir, args.base_model)
    aligned_metrics = load_metrics(args.metrics_dir, args.aligned_model)

    # Use aligned model's judgments (instruct model is the judge)
    humor_scores = load_judgments(args.judgments_dir, args.aligned_model)

    # ---- Inverted-U analysis ----
    print("\n" + "=" * 60)
    print("INVERTED-U ANALYSIS")
    print("=" * 60)

    # Test on base model metrics (wider distributional range)
    iu_results = run_inverted_u_analysis(
        base_metrics, humor_scores, alpha=args.alpha
    )

    header = (
        f"{'Metric':<25} {'Humor Dim':<16} {'a (quad)':>8} "
        f"{'R²':>6} {'p-value':>10} {'Peak':>8} {'Inv-U?':>6}"
    )
    print(header)
    print("-" * 85)
    for r in iu_results:
        peak_str = f"{r.peak_x:.3f}" if r.peak_x is not None else "N/A"
        print(
            f"{r.metric_name:<25} {r.humor_dimension:<16} {r.quadratic_coeff:>8.4f} "
            f"{r.r_squared:>6.4f} {r.p_value_quadratic:>9.2e} "
            f"{peak_str:>8} {'YES' if r.is_inverted_u else 'no':>6}"
        )

    save_inverted_u_results(iu_results, str(out_dir / "inverted_u_results.json"))

    # ---- Alignment comparison ----
    print("\n" + "=" * 60)
    print("ALIGNMENT COMPARISON (base vs aligned)")
    print("=" * 60)

    alignment_results = run_alignment_comparison(base_metrics, aligned_metrics)
    print_comparison_table(alignment_results)
    save_alignment_results(
        alignment_results, str(out_dir / "alignment_comparison.json")
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_inverted_u = sum(1 for r in iu_results if r.is_inverted_u)
    print(f"Inverted-U confirmed: {n_inverted_u}/{len(iu_results)} combinations")

    sig_shifts = [r for r in alignment_results if r.p_value < args.alpha]
    print(f"Significant alignment shifts: {len(sig_shifts)}/{len(alignment_results)}")

    for r in sig_shifts:
        direction = "decreased" if r.mean_shift < 0 else "increased"
        print(
            f"  {r.metric_name}: {direction} by {abs(r.mean_shift):.4f} "
            f"(d={r.cohens_d:.3f})"
        )


if __name__ == "__main__":
    main()
