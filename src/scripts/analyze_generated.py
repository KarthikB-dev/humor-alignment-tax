"""Inverted-U analysis on model-*generated* jokes (the on-target test of Hypothesis 1
and 4: does humor peak at moderate surprisal/entropy, and does temperature move jokes
along that curve?).

Generated jokes carry surprisal + entropy (computed at generation time) and Gemini
humor scores (`judge_scores`). NB: generated jokes have no embedding-distance metric —
`generate_jokes.py` only computes surprisal and entropy — so this fits over those two.

For each alignment tier and a pooled set, fits humor = a*z^2 + b*z + c per
metric x humor-dimension, and reports the temperature sweep (mean surprisal/entropy/
humor per decoding config). The instruct tier is the clean test — the base model barely
writes jokes and the safety model mostly refuses (see alignment_tax_report.md).

Reads:  data/judgments/gemini_bulk/judgments_generated_<tier>.json
Writes: data/analysis/generated_inverted_u.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.inverted_u import run_inverted_u_analysis

HUMOR_DIMS = ["unexpectedness", "cleverness", "amusement", "overall"]
METRICS = ["surprisal", "entropy"]


def _arrays(entries: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build {metric: array} and {humor_dim: array} from generated judgments.

    Failed/absent judge scores become NaN; fit_inverted_u masks non-finite pairs.
    """
    metrics = {m: [] for m in METRICS}
    humor = {d: [] for d in HUMOR_DIMS}
    for e in entries:
        sc = e.get("judge_scores", {})
        metrics["surprisal"].append(e.get("surprisal", np.nan))
        metrics["entropy"].append(e.get("entropy", np.nan))
        for d in HUMOR_DIMS:
            humor[d].append(sc[d] if d in sc else np.nan)
    metrics = {k: np.array(v, dtype=np.float32) for k, v in metrics.items()}
    humor = {k: np.array(v, dtype=np.float32) for k, v in humor.items()}
    # rename metric keys to match the report wording
    return {"punchline_surprisal": metrics["surprisal"],
            "pre_punchline_entropy": metrics["entropy"]}, humor


def _temperature_sweep(entries: list[dict]) -> list[dict]:
    """Per decoding config: n, mean surprisal/entropy/overall. Tests Hypothesis 4."""
    by_cfg: dict[tuple, list[dict]] = {}
    for e in entries:
        key = (e.get("temperature"), e.get("top_p"), e.get("config"))
        by_cfg.setdefault(key, []).append(e)

    rows = []
    for (temp, top_p, cfg), items in sorted(by_cfg.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0)):
        def mean(getter):
            vals = [getter(x) for x in items if getter(x) is not None and np.isfinite(getter(x))]
            return round(float(np.mean(vals)), 3) if vals else None
        rows.append({
            "config": cfg,
            "temperature": temp,
            "top_p": top_p,
            "n": len(items),
            "mean_surprisal": mean(lambda x: x.get("surprisal")),
            "mean_entropy": mean(lambda x: x.get("entropy")),
            "mean_overall": mean(lambda x: x.get("judge_scores", {}).get("overall")),
            "mean_amusement": mean(lambda x: x.get("judge_scores", {}).get("amusement")),
        })
    return rows


def _print_iu_table(title: str, results) -> None:
    print(f"\n{title}")
    print(f"{'Metric':<24} {'Humor Dim':<15} {'a(quad)':>8} {'R²':>6} {'p':>10} {'Peak':>8} {'Inv-U?':>6}")
    print("-" * 82)
    for r in results:
        peak = f"{r.peak_x:.3f}" if r.peak_x is not None else "N/A"
        print(f"{r.metric_name:<24} {r.humor_dimension:<15} {r.quadratic_coeff:>8.4f} "
              f"{r.r_squared:>6.3f} {r.p_value_quadratic:>9.2e} {peak:>8} "
              f"{'YES' if r.is_inverted_u else 'no':>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverted-U + temperature sweep on generated jokes.")
    parser.add_argument("--judgments_dir", default="data/judgments/gemini_bulk")
    parser.add_argument("--tiers", nargs="+", default=[
        "Llama-3.2-1B", "Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct-safety"])
    parser.add_argument("--output", default="data/analysis/generated_inverted_u.json")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    jdir = Path(args.judgments_dir)
    tier_entries: dict[str, list[dict]] = {}
    for tier in args.tiers:
        path = jdir / f"judgments_generated_{tier}.json"
        if not path.exists():
            print(f"  (skip {tier}: {path} not found)")
            continue
        with open(path) as f:
            tier_entries[tier] = json.load(f)

    output = {"inverted_u": {}, "temperature_sweep": {}}

    for tier, entries in tier_entries.items():
        metrics, humor = _arrays(entries)
        results = run_inverted_u_analysis(metrics, humor, alpha=args.alpha)
        _print_iu_table(f"INVERTED-U — {tier} (n={len(entries)})", results)
        output["inverted_u"][tier] = [r.__dict__ for r in results]

        sweep = _temperature_sweep(entries)
        output["temperature_sweep"][tier] = sweep

    # pooled across all tiers (widest metric range)
    pooled = [e for es in tier_entries.values() for e in es]
    metrics, humor = _arrays(pooled)
    pooled_results = run_inverted_u_analysis(metrics, humor, alpha=args.alpha)
    _print_iu_table(f"INVERTED-U — POOLED all tiers (n={len(pooled)})", pooled_results)
    output["inverted_u"]["pooled"] = [r.__dict__ for r in pooled_results]

    # temperature sweep for the clean tier (instruct), printed prominently
    instruct = tier_entries.get("Llama-3.2-1B-Instruct")
    if instruct:
        print("\nTEMPERATURE SWEEP — Llama-3.2-1B-Instruct (Hypothesis 4)")
        print(f"{'config':<16} {'temp':>5} {'top_p':>6} {'n':>4} {'surprisal':>10} {'entropy':>9} {'overall':>8} {'amuse':>7}")
        print("-" * 74)
        for r in output["temperature_sweep"]["Llama-3.2-1B-Instruct"]:
            print(f"{r['config']:<16} {str(r['temperature']):>5} {str(r['top_p']):>6} {r['n']:>4} "
                  f"{str(r['mean_surprisal']):>10} {str(r['mean_entropy']):>9} "
                  f"{str(r['mean_overall']):>8} {str(r['mean_amusement']):>7}")

    n_iu = sum(1 for r in pooled_results if r.is_inverted_u)
    print(f"\nInverted-U confirmed (pooled): {n_iu}/{len(pooled_results)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
