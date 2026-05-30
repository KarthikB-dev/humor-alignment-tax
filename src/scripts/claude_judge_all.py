"""Judge BOTH the human corpus and the model-generated jokes with one Claude judge.

Removes the two-judge confound: the existing pipeline judged the corpus with
Llama-1B and never judged the generated jokes. This re-judges everything with a
single consistent Claude judge.

  - Bulk pass: Haiku 4.5 scores every joke (corpus + all generated sets).
  - Agreement pass: Sonnet 4.6 re-judges a stratified subset; we report
    Haiku-vs-Sonnet agreement as a robustness check.

Outputs (under data/judgments/):
  claude_haiku/judgments_corpus.json
  claude_haiku/judgments_generated_<model>.json   (one per generated set)
  claude_sonnet/subset_agreement.json             (subset w/ both judges + deltas)

Requires ANTHROPIC_API_KEY (in .env or the environment).
"""

import argparse
import asyncio
import importlib
import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.dataset.joke import DatasetSource
from src.dataset.loader import load_all_jokes

load_dotenv()

# Provider -> (judge module, default bulk model, default subset model, out subdir prefix)
PROVIDERS = {
    "claude": ("src.evaluation.claude_judge", "claude-haiku-4-5", "claude-sonnet-4-6", "claude"),
    "gemini": ("src.evaluation.gemini_judge", "gemini-2.5-flash", "gemini-2.5-pro", "gemini"),
}


def _load_corpus_items(sources: list[str], limit: int | None) -> list[dict]:
    jokes = load_all_jokes(
        sources=[DatasetSource(s) for s in sources], limit_per_source=limit
    )
    return [
        {
            "kind": "corpus",
            "stratum": f"corpus/{j.source.value}",
            "joke_id": j.joke_id,
            "source": j.source.value,
            "language": j.language.value,
            "setup": j.setup,
            "punchline": j.punchline,
        }
        for j in jokes
    ]


def _load_generated_items(generated_dir: str, limit: int | None) -> dict[str, list[dict]]:
    """Return {gen_model: [items]} for each generated_*.json file."""
    by_model: dict[str, list[dict]] = {}
    for path in sorted(Path(generated_dir).glob("generated_*.json")):
        gen_model = path.stem.removeprefix("generated_")
        with open(path) as f:
            jokes = json.load(f)
        if limit:
            jokes = jokes[:limit]
        by_model[gen_model] = [
            {
                "kind": "generated",
                "stratum": f"generated/{gen_model}/{j.get('config')}",
                "gen_model": gen_model,
                "config": j.get("config"),
                "temperature": j.get("temperature"),
                "top_p": j.get("top_p"),
                "surprisal": j.get("surprisal"),
                "entropy": j.get("entropy"),
                "setup": j.get("setup"),
                "punchline": j.get("punchline"),
            }
            for j in jokes
        ]
    return by_model


def _stratified_sample(items: list[dict], n: int, seed: int) -> list[int]:
    """Pick ~n indices spread evenly across strata. Returns indices into `items`."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_stratum[it["stratum"]].append(i)

    strata = sorted(by_stratum)
    per = max(1, n // len(strata))
    chosen: list[int] = []
    for s in strata:
        idxs = by_stratum[s]
        chosen.extend(rng.sample(idxs, min(per, len(idxs))))
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge corpus + generated jokes with one LLM judge.")
    parser.add_argument("--provider", default="gemini", choices=list(PROVIDERS))
    parser.add_argument("--bulk_model", default=None, help="Override provider default bulk model")
    parser.add_argument("--subset_model", default=None, help="Override provider default subset model")
    parser.add_argument("--subset_size", type=int, default=120)
    parser.add_argument(
        "--sources", nargs="+", default=["r_jokes", "one_liner"],
        choices=[s.value for s in DatasetSource],
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap jokes per corpus source / generated file")
    parser.add_argument("--concurrency", type=int, default=None, help="Default: 8 (claude) / 4 (gemini)")
    parser.add_argument("--generated_dir", default="data/generated")
    parser.add_argument("--output_dir", default="data/judgments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    module_name, default_bulk, default_subset, prefix = PROVIDERS[args.provider]
    judge_mod = importlib.import_module(module_name)
    judge_jokes = judge_mod.judge_jokes
    scores_to_dict = judge_mod.scores_to_dict
    bulk_model = args.bulk_model or default_bulk
    subset_model = args.subset_model or default_subset
    concurrency = args.concurrency if args.concurrency is not None else (4 if args.provider == "gemini" else 8)

    out_dir = Path(args.output_dir)
    bulk_dir = out_dir / f"{prefix}_bulk"
    subset_dir = out_dir / f"{prefix}_subset"
    bulk_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    # ---- assemble all jokes (corpus + every generated set) ----
    corpus_items = _load_corpus_items(args.sources, args.limit)
    generated_by_model = _load_generated_items(args.generated_dir, args.limit)
    all_items = corpus_items + [it for items in generated_by_model.values() for it in items]
    print(
        f"Loaded {len(corpus_items)} corpus + "
        f"{sum(len(v) for v in generated_by_model.values())} generated = {len(all_items)} jokes",
        flush=True,
    )

    # ---- bulk pass: cheap model judges everything ----
    bulk_scores = asyncio.run(
        judge_jokes(all_items, model=bulk_model, concurrency=concurrency)
    )
    for it, sc in zip(all_items, bulk_scores):
        it["judge_scores"] = scores_to_dict(sc)

    n_ok = sum(1 for it in all_items if "error" not in it["judge_scores"])
    print(f"{bulk_model} judged {n_ok}/{len(all_items)} successfully", flush=True)

    # write corpus + per-generated-model judgment files
    with open(bulk_dir / "judgments_corpus.json", "w") as f:
        json.dump(corpus_items, f, indent=2, ensure_ascii=False)
    for gen_model, items in generated_by_model.items():
        with open(bulk_dir / f"judgments_generated_{gen_model}.json", "w") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Wrote {bulk_model} judgments to {bulk_dir}/", flush=True)

    # ---- agreement pass: stronger model re-judges a stratified subset ----
    subset_idx = _stratified_sample(all_items, args.subset_size, args.seed)
    subset = [all_items[i] for i in subset_idx]
    print(f"\nRe-judging a {len(subset)}-joke stratified subset with {subset_model}", flush=True)
    subset_scores = asyncio.run(
        judge_jokes(subset, model=subset_model, concurrency=concurrency)
    )

    agreement_rows = []
    deltas: list[float] = []
    for it, sc in zip(subset, subset_scores):
        strong = scores_to_dict(sc)
        bulk = it["judge_scores"]
        row = {k: it.get(k) for k in ("kind", "stratum", "setup", "punchline")}
        row["bulk"] = bulk
        row["strong"] = strong
        if "error" not in bulk and "error" not in strong:
            d = abs(bulk["overall"] - strong["overall"])
            row["overall_abs_delta"] = d
            deltas.append(d)
        agreement_rows.append(row)

    summary = {
        "subset_model": subset_model,
        "bulk_model": bulk_model,
        "n_subset": len(subset),
        "n_compared": len(deltas),
        "mean_abs_overall_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "exact_overall_agreement": (
            sum(1 for d in deltas if d == 0) / len(deltas) if deltas else None
        ),
        "within_1_agreement": (
            sum(1 for d in deltas if d <= 1) / len(deltas) if deltas else None
        ),
    }
    with open(subset_dir / "subset_agreement.json", "w") as f:
        json.dump({"summary": summary, "rows": agreement_rows}, f, indent=2, ensure_ascii=False)

    print(f"\n{bulk_model} vs {subset_model} agreement on the subset:")
    print(f"  mean |Δ overall|:        {summary['mean_abs_overall_delta']}")
    print(f"  exact overall agreement: {summary['exact_overall_agreement']}")
    print(f"  within-1 agreement:      {summary['within_1_agreement']}")
    print(f"Wrote agreement report to {subset_dir}/subset_agreement.json", flush=True)


if __name__ == "__main__":
    main()
