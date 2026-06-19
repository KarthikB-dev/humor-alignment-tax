"""Generate source-grounded local satire headlines from manual source capsules."""

import argparse
import os
import re
import uuid
from datetime import datetime, timezone

from src.dataset.local_satire import LocalSatireGeneration
from src.dataset.local_satire_loader import (
    load_local_satire_sources,
    save_local_satire_generations,
)
from src.evaluation.local_satire_prompts import (
    PROMPT_CONDITIONS,
    build_local_satire_prompt,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _split_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _call_model(model: str, prompt: str, temperature: float) -> str:
    model_lower = model.lower()
    if model_lower.startswith("gemini") or "gemini" in model_lower:
        return _call_gemini(model, prompt, temperature)
    if model_lower.startswith("gpt") or model_lower.startswith("openai"):
        return _call_openai(model, prompt, temperature)
    raise RuntimeError(
        f"Do not know how to call model '{model}'. Use a Gemini model name or "
        "an OpenAI/GPT model name, or run with --dry-run."
    )


def _call_gemini(model: str, prompt: str, temperature: float) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY, or use --dry-run.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("Install google-genai to call Gemini models.") from e

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=1024,
    )
    if "flash" in model.lower() or "lite" in model.lower():
        config.thinking_config = types.ThinkingConfig(thinking_budget=0)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return (response.text or "").strip()


def _call_openai(model: str, prompt: str, temperature: float) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY, or use --dry-run.")
    try:
        from openai import BadRequestError, OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai to call OpenAI models.") from e

    client = OpenAI(api_key=api_key)
    request = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": 1024,
    }
    try:
        response = client.responses.create(**request)
    except BadRequestError as e:
        if "temperature" not in str(e).lower():
            raise
        request.pop("temperature", None)
        response = client.responses.create(**request)
    return response.output_text.strip()


def _parse_headline(raw_output: str, condition: str) -> str:
    text = raw_output.strip()
    if condition == "target_first_rank":
        match = re.search(r"Best headline:\s*(.+)", text, flags=re.IGNORECASE)
        if match:
            return _clean_headline(match.group(1))
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return _clean_headline(first_line)


def _clean_headline(text: str) -> str:
    text = re.sub(r"^[-*\d.)\s]+", "", text.strip())
    text = re.sub(r"^(?:best\s+headline|headline):\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'")
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local satire headlines from manual source capsules."
    )
    parser.add_argument("--sources", default="data/local_satire_sources.jsonl")
    parser.add_argument("--out", default="outputs/local_satire_generations.csv")
    parser.add_argument(
        "--models",
        required=True,
        help='Comma-separated model names, e.g. "gemini-flash,gpt-mini"',
    )
    parser.add_argument(
        "--conditions",
        default="generic_issue,source_grounded,target_first_rank",
        help=f"Comma-separated prompt conditions: {', '.join(PROMPT_CONDITIONS)}",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = load_local_satire_sources(args.sources)
    models = _split_arg(args.models)
    conditions = _split_arg(args.conditions)

    for condition in conditions:
        if condition not in PROMPT_CONDITIONS:
            raise ValueError(
                f"Unknown condition '{condition}'. Expected: {', '.join(PROMPT_CONDITIONS)}"
            )

    rows: list[LocalSatireGeneration] = []
    for item in sources:
        for model in models:
            for condition in conditions:
                prompt = build_local_satire_prompt(item, condition)
                if args.dry_run:
                    raw_output = ""
                    output_headline = ""
                    print(f"\n--- {item.id} | {model} | {condition} ---\n{prompt}")
                else:
                    raw_output = _call_model(model, prompt, args.temperature)
                    output_headline = _parse_headline(raw_output, condition)

                rows.append(
                    LocalSatireGeneration(
                        id=f"local_satire_{uuid.uuid4().hex[:10]}",
                        source_id=item.id,
                        model=model,
                        prompt_condition=condition,
                        prompt=prompt,
                        output_headline=output_headline,
                        raw_output=raw_output,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )

    if args.dry_run:
        print(f"\nDry run complete: printed {len(rows)} prompts; no CSV written.")
    else:
        save_local_satire_generations(args.out, rows)
        print(f"Saved {len(rows)} generations to {args.out}")


if __name__ == "__main__":
    main()
