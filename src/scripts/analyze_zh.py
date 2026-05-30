"""Per-source distributional comparison for the Chinese datasets (Oogiri, Ruozhiba).

compare_safety.py pools all jokes in a metrics dir; this splits the pooled
data/metrics_zh arrays back out by source (via joke_metadata.json) so we can see
whether the base->instruct->safety widening holds in each dataset separately.
Writes data/analysis/zh_distribution.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS_DIR = Path("data/metrics_zh")
TIERS = {
    "base": "Llama-3.2-1B",
    "instruct": "Llama-3.2-1B-Instruct",
    "safety": "Llama-3.2-1B-Instruct-safety",
}
METRICS = {"surprisal": "surprisals", "entropy": "entropies", "distance": "distances"}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    s = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((b.mean() - a.mean()) / s) if s > 0 else 0.0


def main() -> None:
    meta = json.load(open(METRICS_DIR / "joke_metadata.json"))
    sources = np.array([m["source"] for m in meta])

    tier_data = {
        tier: np.load(METRICS_DIR / f"{name}.npz", allow_pickle=True)
        for tier, name in TIERS.items()
    }

    report: dict[str, dict] = {}
    for src in ["oogiri", "ruozhiba", "__pooled__"]:
        mask = np.ones(len(sources), bool) if src == "__pooled__" else (sources == src)
        n = int(mask.sum())
        report[src] = {"n": n, "metrics": {}}
        print(f"\n{'='*64}\n  {src}  (n={n})\n{'='*64}")
        print(f"{'metric':<10} {'base':>8} {'instruct':>9} {'safety':>8} "
              f"{'b→i d':>7} {'i→s d':>7} {'b→s d':>7}")
        print("-" * 64)
        for mname, key in METRICS.items():
            vals = {t: tier_data[t][key].astype(float)[mask] for t in TIERS}
            bi_d = cohens_d(vals["base"], vals["instruct"])
            is_d = cohens_d(vals["instruct"], vals["safety"])
            bs_d = cohens_d(vals["base"], vals["safety"])
            _, bi_p = stats.mannwhitneyu(vals["base"], vals["instruct"])
            _, is_p = stats.mannwhitneyu(vals["instruct"], vals["safety"])
            _, bs_p = stats.mannwhitneyu(vals["base"], vals["safety"])
            report[src]["metrics"][mname] = {
                "mean": {t: float(vals[t].mean()) for t in TIERS},
                "base_to_instruct": {"d": bi_d, "p": float(bi_p)},
                "instruct_to_safety": {"d": is_d, "p": float(is_p)},
                "base_to_safety": {"d": bs_d, "p": float(bs_p)},
            }
            print(f"{mname:<10} {vals['base'].mean():>8.3f} {vals['instruct'].mean():>9.3f} "
                  f"{vals['safety'].mean():>8.3f} {bi_d:>+7.3f} {is_d:>+7.3f} {bs_d:>+7.3f}")

    out = Path("data/analysis/zh_distribution.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
