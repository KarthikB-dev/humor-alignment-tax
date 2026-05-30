"""Run LLM-as-judge humor evaluation on model-*generated* jokes.

Unlike judge_humor.py (which scores the human corpus from data/raw), this loads
the generated jokes in data/generated/ — produced by generate_jokes.py across the
decoding sweep — and attaches judge scores so each generated joke can be placed on
the inverted-U curve. Generated-joke judgments retain their decoding config and the
surprisal/entropy already computed at generation time.
"""

import argparse
import json
import uuid
from pathlib import Path

import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from src.evaluation.llm_judge import judge_joke
from src.utils.utils import get_model_path, load_model_and_tokenizer, model_name_from_path

load_dotenv()


def _model_name_from_generated_file(path: Path) -> str:
    """generated_Llama-3.2-1B-Instruct.json -> Llama-3.2-1B-Instruct."""
    return path.stem.removeprefix("generated_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score model-generated jokes using LLM-as-judge."
    )
    parser.add_argument(
        "--judge_model",
        default="LLAMA_3_2_1B_INSTRUCT",
        help="Model key for the judge (must be instruction-tuned)",
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        default=None,
        help="Specific generated_*.json files. Default: all in --generated_dir.",
    )
    parser.add_argument("--generated_dir", default="data/generated")
    parser.add_argument("--output_dir", default="data/judgments")
    parser.add_argument("--limit", type=int, default=None, help="Cap jokes per file")
    args = parser.parse_args()

    if args.input_files:
        gen_files = [Path(p) for p in args.input_files]
    else:
        gen_files = sorted(Path(args.generated_dir).glob("generated_*.json"))
    if not gen_files:
        raise SystemExit(f"No generated_*.json files found in {args.generated_dir}")

    wandb.init(
        project="humor-alignment-tax",
        name=f"judge-generated-{uuid.uuid4().hex[:8]}",
        config=vars(args),
    )

    judge_path = get_model_path(args.judge_model)
    judge_name = model_name_from_path(judge_path)
    print(f"Loading judge: {judge_name}", flush=True)
    model, tokenizer = load_model_and_tokenizer(judge_path, device_map={"": 0})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gen_file in gen_files:
        gen_model = _model_name_from_generated_file(gen_file)
        with open(gen_file) as f:
            jokes = json.load(f)
        if args.limit:
            jokes = jokes[: args.limit]

        print(f"\nJudging {len(jokes)} generated jokes from {gen_model}", flush=True)
        results = []
        n_success = 0

        for joke in tqdm(jokes, desc=gen_model, unit="joke"):
            judgment = judge_joke(model, tokenizer, joke["setup"], joke["punchline"])
            # Carry over generation-time fields so judgments sit on the inverted-U
            entry = {
                "gen_model": gen_model,
                "config": joke.get("config"),
                "temperature": joke.get("temperature"),
                "top_p": joke.get("top_p"),
                "surprisal": joke.get("surprisal"),
                "entropy": joke.get("entropy"),
                "setup": joke.get("setup"),
                "punchline": joke.get("punchline"),
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

        output_file = out_dir / f"judgments_generated_{gen_model}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {n_success}/{len(jokes)} judgments to {output_file}", flush=True)
        wandb.log(
            {
                f"{gen_model}/n_judged": n_success,
                f"{gen_model}/n_failed": len(jokes) - n_success,
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()
