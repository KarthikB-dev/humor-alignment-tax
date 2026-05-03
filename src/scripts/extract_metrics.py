"""Extract distributional metrics (surprisal, entropy, embedding distance)
for all jokes across base and aligned models."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from joblib import dump, load
from tqdm import tqdm

from src.dataset.joke import DatasetSource, DistributionalMetrics, JokeEntry
from src.dataset.loader import load_all_jokes
from src.metrics.embedding import compute_embedding_distance
from src.metrics.entropy import compute_entropy
from src.metrics.surprisal import compute_surprisal
from src.utils.utils import (
    MODEL_PAIRS,
    get_model_path,
    load_model_and_tokenizer,
    model_name_from_path,
)

load_dotenv()


def extract_metrics_for_model(
    model_path: str,
    jokes: list[JokeEntry],
    output_dir: str,
) -> dict[str, np.ndarray]:
    """Extract all three distributional metrics for a list of jokes."""
    model_name = model_name_from_path(model_path)
    print(f"\nLoading model: {model_name}", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_path, device_map={"": 0})

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}", flush=True)

    surprisals = []
    entropies = []
    distances = []
    all_token_surprisals = []
    all_entropy_trajectories = []

    for joke in tqdm(jokes, desc=f"  {model_name}", unit="joke"):
        # Surprisal
        mean_surp, token_surps = compute_surprisal(
            model, tokenizer, joke.setup, joke.punchline
        )
        surprisals.append(float(mean_surp))
        all_token_surprisals.append(token_surps)

        # Entropy
        pre_punch_ent, ent_traj = compute_entropy(
            model, tokenizer, joke.setup, joke.punchline
        )
        entropies.append(float(pre_punch_ent))
        all_entropy_trajectories.append(ent_traj)

        # Embedding distance
        emb_dist = compute_embedding_distance(
            model, tokenizer, joke.setup, joke.punchline
        )
        distances.append(float(emb_dist))

    # Save results
    out_path = Path(output_dir) / f"{model_name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        surprisals=np.array(surprisals, dtype=np.float32),
        entropies=np.array(entropies, dtype=np.float32),
        distances=np.array(distances, dtype=np.float32),
        joke_ids=np.array([j.joke_id for j in jokes], dtype=object),
    )
    print(f"  Saved metrics to {out_path}", flush=True)

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()

    return {
        "punchline_surprisal": np.array(surprisals, dtype=np.float32),
        "pre_punchline_entropy": np.array(entropies, dtype=np.float32),
        "embedding_distance": np.array(distances, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract distributional metrics for jokes across model pairs."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["r_jokes", "one_liner"],
        choices=[s.value for s in DatasetSource],
        help="Which joke datasets to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max jokes per source (for debugging)",
    )
    parser.add_argument("--output_dir", default="data/metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run these individual model keys instead of MODEL_PAIRS (e.g. LLAMA_3_2_1B_INSTRUCT_SAFETY)",
    )
    args = parser.parse_args()

    wandb.init(
        project="humor-alignment-tax",
        name=f"extract-metrics-{uuid.uuid4().hex[:8]}",
        notes=f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        config=vars(args),
    )

    np.random.seed(args.seed)

    sources = [DatasetSource(s) for s in args.sources]
    jokes = load_all_jokes(sources=sources, limit_per_source=args.limit)
    print(f"\nTotal jokes loaded: {len(jokes)}", flush=True)

    # Save joke metadata
    meta_path = Path(args.output_dir) / "joke_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(
            [
                {
                    "joke_id": j.joke_id,
                    "source": j.source.value,
                    "language": j.language.value,
                    "setup_len": len(j.setup),
                    "punchline_len": len(j.punchline),
                }
                for j in jokes
            ],
            f,
            indent=2,
        )

    if args.models:
        # Individual model mode — run each key independently
        for model_key in args.models:
            model_path = get_model_path(model_key)
            metrics = extract_metrics_for_model(model_path, jokes, args.output_dir)
            model_name = model_name_from_path(model_path)
            for metric_name, vals in metrics.items():
                wandb.log({f"{model_name}/{metric_name}/mean": float(vals.mean()),
                           f"{model_name}/{metric_name}/std": float(vals.std())})
    else:
        for base_key, aligned_key in MODEL_PAIRS:
            base_path = get_model_path(base_key)
            aligned_path = get_model_path(aligned_key)

            base_metrics = extract_metrics_for_model(base_path, jokes, args.output_dir)
            aligned_metrics = extract_metrics_for_model(
                aligned_path, jokes, args.output_dir
            )

            for metric_name in base_metrics:
                wandb.log(
                    {
                        f"base/{metric_name}/mean": float(base_metrics[metric_name].mean()),
                        f"base/{metric_name}/std": float(base_metrics[metric_name].std()),
                        f"aligned/{metric_name}/mean": float(
                            aligned_metrics[metric_name].mean()
                        ),
                        f"aligned/{metric_name}/std": float(
                            aligned_metrics[metric_name].std()
                        ),
                    }
                )

    print("\nDone. All metrics saved.", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
