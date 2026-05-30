"""Build a blind human-rating validation subset.

Samples a stratified subset of jokes (across alignment tiers x decoding configs,
plus the human corpus) from the existing Gemini judgments, and writes:

  * data/human_eval/human_rating_sheet.csv  -- BLIND sheet for a human to fill in
        (setup, punchline, and empty 1-10 rating columns; no model scores shown)
  * data/human_eval/rating_key.json         -- hidden key: id -> tier/config + LLM scores
  * (scoring is done separately by score_human_ratings.py once the sheet is filled)

The point is to validate the LLM judge: once a human fills in the sheet,
score_human_ratings.py computes human<->LLM correlation per dimension.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

JUDGMENTS_DIR = Path("data/judgments/gemini_bulk")
OUT_DIR = Path("data/human_eval")
DIMS = ["unexpectedness", "cleverness", "amusement", "overall"]

# Map each judgment file to a coarse tier label.
FILES = {
    "judgments_corpus.json": "corpus",
    "judgments_generated_Llama-3.2-1B.json": "base",
    "judgments_generated_Llama-3.2-1B-Instruct.json": "instruct",
    "judgments_generated_Llama-3.2-1B-Instruct-safety.json": "safety",
}


def load_records() -> list[dict]:
    """Flatten all judgment files into one list of normalized records."""
    records: list[dict] = []
    for fname, tier in FILES.items():
        path = JUDGMENTS_DIR / fname
        if not path.exists():
            print(f"  WARN: {path} missing, skipping")
            continue
        for r in json.load(open(path)):
            scores = r.get("judge_scores")
            if not scores or any(scores.get(d) is None for d in DIMS):
                continue
            # corpus rows use `source` as their sub-stratum; generated use `config`.
            config = r.get("config") or r.get("source") or "na"
            records.append(
                {
                    "tier": tier,
                    "config": config,
                    "stratum": f"{tier}/{config}",
                    "setup": r.get("setup", ""),
                    "punchline": r.get("punchline", ""),
                    "llm_scores": {d: float(scores[d]) for d in DIMS},
                }
            )
    return records


def stratified_sample(records: list[dict], target: int, seed: int) -> list[dict]:
    """Roughly even allocation across strata, capped by availability."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_stratum[r["stratum"]].append(r)

    strata = sorted(by_stratum)
    per = max(1, target // len(strata))
    picked: list[dict] = []
    for s in strata:
        pool = by_stratum[s]
        rng.shuffle(pool)
        picked.extend(pool[:per])

    # Top up toward `target` from the largest remaining pools if rounding left us short.
    if len(picked) < target:
        chosen_ids = {id(r) for r in picked}
        leftovers = [r for r in records if id(r) not in chosen_ids]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: target - len(picked)])

    rng.shuffle(picked)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=120, help="Approx total jokes to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    print(f"Loaded {len(records)} scored records across {len(FILES)} tiers")

    sample = stratified_sample(records, args.target, args.seed)
    print(f"Sampled {len(sample)} jokes")

    # Blind sheet + hidden key, sharing a stable rating_id.
    sheet_path = OUT_DIR / "human_rating_sheet.csv"
    key: dict[str, dict] = {}
    with open(sheet_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["rating_id", "setup", "punchline"]
            + [f"human_{d}" for d in DIMS]
            + ["notes"]
        )
        for i, r in enumerate(sample):
            rid = f"hr{i:03d}"
            w.writerow([rid, r["setup"], r["punchline"], "", "", "", "", ""])
            key[rid] = {
                "tier": r["tier"],
                "config": r["config"],
                "stratum": r["stratum"],
                "llm_scores": r["llm_scores"],
            }

    key_path = OUT_DIR / "rating_key.json"
    json.dump(key, open(key_path, "w"), indent=2, ensure_ascii=False)

    # Stratum coverage report.
    counts: dict[str, int] = defaultdict(int)
    for v in key.values():
        counts[v["stratum"]] += 1
    print("\nStratum coverage:")
    for s in sorted(counts):
        print(f"  {s:40s} {counts[s]}")
    print(f"\nWrote blind sheet -> {sheet_path}")
    print(f"Wrote hidden key  -> {key_path}")
    print("\nFill the human_* columns (1-10) in the sheet, then run:")
    print("  uv run python -m src.scripts.score_human_ratings")


if __name__ == "__main__":
    main()
