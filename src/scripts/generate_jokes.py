"""Generate jokes under varying decoding strategies to test
how temperature/top-p shifts distributional properties."""

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset.joke import DECODING_CONFIGS, DecodingConfig
from src.metrics.entropy import compute_entropy
from src.metrics.surprisal import compute_surprisal
from src.utils.utils import get_model_path, load_model_and_tokenizer, model_name_from_path

load_dotenv()

GENERATION_PROMPT = """\
Write a short, funny joke. Include a setup and a punchline separated by a newline.
Only output the joke, nothing else."""


def generate_joke(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: DecodingConfig,
    max_new_tokens: int = 128,
) -> str | None:
    """Generate a single joke with the given decoding config."""
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": GENERATION_PROMPT}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = GENERATION_PROMPT
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if config.temperature == 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = config.temperature
        if config.top_p < 1.0:
            gen_kwargs["top_p"] = config.top_p
        if config.top_k > 0:
            gen_kwargs["top_k"] = config.top_k

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    del output_ids
    return text.strip() if text.strip() else None


def split_generated_joke(text: str) -> tuple[str, str] | None:
    """Try to split generated text into setup and punchline."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:])
    # Fallback: split at midpoint
    words = text.split()
    if len(words) < 4:
        return None
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate jokes with varying decoding strategies."
    )
    parser.add_argument(
        "--model_keys",
        nargs="+",
        default=["LLAMA_3_2_1B_BASE", "LLAMA_3_2_1B_INSTRUCT"],
    )
    parser.add_argument(
        "--n_jokes", type=int, default=50, help="Jokes to generate per config"
    )
    parser.add_argument("--output_dir", default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wandb.init(
        project="humor-alignment-tax",
        name=f"generate-jokes-{uuid.uuid4().hex[:8]}",
        config=vars(args),
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key in args.model_keys:
        model_path = get_model_path(model_key)
        model_name = model_name_from_path(model_path)
        print(f"\nModel: {model_name}", flush=True)

        model, tokenizer = load_model_and_tokenizer(model_path, device_map={"": 0})

        model_results = []

        for config in DECODING_CONFIGS:
            print(f"\n  Config: {config.label}", flush=True)
            jokes = []

            for i in tqdm(range(args.n_jokes), desc=f"    {config.label}", unit="joke"):
                text = generate_joke(model, tokenizer, config)
                if text is None:
                    continue

                split = split_generated_joke(text)
                if split is None:
                    continue

                setup, punchline = split

                # Extract metrics on the generated joke
                mean_surp, _ = compute_surprisal(model, tokenizer, setup, punchline)
                pre_ent, _ = compute_entropy(model, tokenizer, setup, punchline)

                jokes.append(
                    {
                        "text": text,
                        "setup": setup,
                        "punchline": punchline,
                        "config": config.label,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "surprisal": float(mean_surp),
                        "entropy": float(pre_ent),
                    }
                )

            model_results.extend(jokes)
            print(f"    Generated {len(jokes)}/{args.n_jokes} valid jokes", flush=True)

            wandb.log(
                {
                    f"{model_name}/{config.label}/n_valid": len(jokes),
                    f"{model_name}/{config.label}/mean_surprisal": float(
                        np.mean([j["surprisal"] for j in jokes])
                    )
                    if jokes
                    else 0.0,
                    f"{model_name}/{config.label}/mean_entropy": float(
                        np.mean([j["entropy"] for j in jokes])
                    )
                    if jokes
                    else 0.0,
                }
            )

        # Save all results for this model
        output_file = out_dir / f"generated_{model_name}.json"
        with open(output_file, "w") as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(model_results)} jokes to {output_file}", flush=True)

        del model
        torch.cuda.empty_cache()

    wandb.finish()


if __name__ == "__main__":
    main()
