"""Score a filled human-rating sheet against the LLM judge.

Reads data/human_eval/human_rating_sheet.csv (with the human_* columns filled
1-10) and data/human_eval/rating_key.json, then reports per-dimension
human<->LLM agreement: Pearson r, Spearman rho, mean absolute error, and
exact / within-1 hit rates. Writes data/human_eval/human_llm_agreement.json.

Rows with any blank human_* cell are skipped, so this can be run on a partially
filled sheet to get an interim read.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

OUT_DIR = Path("data/human_eval")
DIMS = ["unexpectedness", "cleverness", "amusement", "overall"]


def main() -> None:
    key = json.load(open(OUT_DIR / "rating_key.json"))
    sheet = OUT_DIR / "human_rating_sheet.csv"

    human: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(open(sheet)):
        rid = row["rating_id"]
        vals = {}
        ok = True
        for d in DIMS:
            cell = (row.get(f"human_{d}") or "").strip()
            if cell == "":
                ok = False
                break
            vals[d] = float(cell)
        if ok:
            human[rid] = vals

    n = len(human)
    if n == 0:
        print("No fully-rated rows found. Fill the human_* columns (1-10) first.")
        return
    print(f"Scoring {n} fully-rated jokes (of {len(key)} in sheet)\n")

    report: dict[str, dict] = {}
    for d in DIMS:
        h = np.array([human[r][d] for r in human])
        m = np.array([key[r]["llm_scores"][d] for r in human])
        diff = np.abs(h - m)
        pear = float(pearsonr(h, m)[0]) if np.std(h) > 0 and np.std(m) > 0 else float("nan")
        spear = float(spearmanr(h, m)[0]) if np.std(h) > 0 and np.std(m) > 0 else float("nan")
        report[d] = {
            "n": n,
            "pearson_r": pear,
            "spearman_rho": spear,
            "mae": float(diff.mean()),
            "exact_pct": float((diff == 0).mean() * 100),
            "within1_pct": float((diff <= 1).mean() * 100),
            "human_mean": float(h.mean()),
            "llm_mean": float(m.mean()),
        }

    hdr = f"{'dim':14s} {'n':>4s} {'pearson':>8s} {'spearman':>9s} {'MAE':>6s} {'exact%':>7s} {'<=1%':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for d in DIMS:
        s = report[d]
        print(
            f"{d:14s} {s['n']:>4d} {s['pearson_r']:>8.3f} {s['spearman_rho']:>9.3f} "
            f"{s['mae']:>6.2f} {s['exact_pct']:>6.1f}% {s['within1_pct']:>5.1f}%"
        )

    out = OUT_DIR / "human_llm_agreement.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
