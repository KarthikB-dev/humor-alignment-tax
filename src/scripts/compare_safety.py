"""Three-way comparison: base vs instruct vs safety-instruct.

Loads pre-computed .npz metric files and prints a side-by-side table of
means, shifts, Cohen's d, and p-values for each metric.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import stats


def load_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "surprisal": data["surprisals"].astype(float),
        "entropy": data["entropies"].astype(float),
        "distance": data["distances"].astype(float),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((b.mean() - a.mean()) / pooled_std) if pooled_std > 0 else 0.0


def compare(name_a: str, a: dict, name_b: str, b: dict) -> None:
    print(f"\n{'─'*72}")
    print(f"  {name_a}  →  {name_b}")
    print(f"{'─'*72}")
    print(f"{'Metric':<14} {'A mean':>8} {'B mean':>8} {'Shift':>8} {'Cohen d':>9} {'p-value':>12}")
    print(f"{'─'*72}")
    for metric in ("surprisal", "entropy", "distance"):
        av, bv = a[metric], b[metric]
        shift = bv.mean() - av.mean()
        d = cohens_d(av, bv)
        _, p = stats.mannwhitneyu(av, bv, alternative="two-sided")
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"{metric:<14} {av.mean():>8.4f} {bv.mean():>8.4f} {shift:>+8.4f} {d:>+9.4f} {p:>10.2e} {stars}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way metric comparison.")
    parser.add_argument("--metrics_dir", default="data/metrics")
    parser.add_argument("--base",    default="Llama-3.2-1B")
    parser.add_argument("--instruct", default="Llama-3.2-1B-Instruct")
    parser.add_argument("--safety",   default="Llama-3.2-1B-Instruct-safety")
    args = parser.parse_args()

    d = Path(args.metrics_dir)
    base    = load_npz(d / f"{args.base}.npz")
    instruct = load_npz(d / f"{args.instruct}.npz")
    safety  = load_npz(d / f"{args.safety}.npz")

    print("\n" + "="*72)
    print("  THREE-WAY ALIGNMENT TAX COMPARISON")
    print("="*72)
    print(f"\n  base    n={len(base['surprisal'])}")
    print(f"  instruct n={len(instruct['surprisal'])}")
    print(f"  safety   n={len(safety['surprisal'])}")

    compare(args.base,    base,    args.instruct, instruct)
    compare(args.instruct, instruct, args.safety,  safety)
    compare(args.base,    base,    args.safety,   safety)

    # Summary: direction of each shift
    print(f"\n{'='*72}")
    print("  SUMMARY — direction of metric shifts")
    print(f"{'='*72}")
    header = f"{'Metric':<14} {'base→instruct':>16} {'instruct→safety':>17} {'base→safety':>13}"
    print(header)
    print("─" * 72)
    for metric in ("surprisal", "entropy", "distance"):
        def arrow(a, b):
            d = b.mean() - a.mean()
            return f"{'↑' if d > 0 else '↓'} {abs(d):.4f}"
        print(
            f"{metric:<14}"
            f" {arrow(base[metric], instruct[metric]):>16}"
            f" {arrow(instruct[metric], safety[metric]):>17}"
            f" {arrow(base[metric], safety[metric]):>13}"
        )


if __name__ == "__main__":
    main()
