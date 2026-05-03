"""Run LLM-as-judge humor evaluation on all jokes."""

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from src.dataset.joke import DatasetSource
from src.dataset.loader import load_all_jokes
from src.evaluation.llm_judge import judge_joke
from src.utils.utils import get_model_path, load_model_and_tokenizer, model_name_from_path

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score jokes using LLM-as-judge on multiple humor dimensions."
    )
    parser.add_argument(
        "--judge_model",
        default="LLAMA_3_1_8B_INSTRUCT",
        help="Model key for the judge (must be instruction-tuned)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["r_jokes", "one_liner"],
        choices=[s.value for s in DatasetSource],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default="data/judgments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wandb.init(
        project="humor-alignment-tax",
        name=f"judge-humor-{uuid.uuid4().hex[:8]}",
        config=vars(args),
    )

    model_path = get_model_path(args.judge_model)
    model_name = model_name_from_path(model_path)
    model, tokenizer = load_model_and_tokenizer(model_path, device_map={"": 0})

    sources = [DatasetSource(s) for s in args.sources]
    jokes = load_all_jokes(sources=sources, limit_per_source=args.limit)
    print(f"\nJudging {len(jokes)} jokes with {model_name}", flush=True)

    results = []
    n_success = 0

    for joke in tqdm(jokes, desc="Judging", unit="joke"):
        judgment = judge_joke(model, tokenizer, joke.setup, joke.punchline)
        entry = {
            "joke_id": joke.joke_id,
            "source": joke.source.value,
            "language": joke.language.value,
        }
        if judgment is not None:
            entry.update(
                {
                    "unexpectedness": float(judgment.unexpectedness),
                    "cleverness": float(judgment.cleverness),
                    "amusement": float(judgment.amusement),
                    "overall": float(judgment.overall),
                    "rationale": judgment.rationale,
                }
            )
            n_success += 1
        else:
            entry["error"] = "failed_to_parse"
        results.append(entry)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / f"judgments_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {n_success}/{len(jokes)} judgments to {output_file}", flush=True)
    wandb.log({"n_judged": n_success, "n_failed": len(jokes) - n_success})
    wandb.finish()


if __name__ == "__main__":
    main()
