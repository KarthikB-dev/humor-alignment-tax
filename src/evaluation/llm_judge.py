"""LLM-as-judge humor evaluation.

Uses a strong instruction-tuned model to rate jokes on multiple dimensions:
unexpectedness, cleverness, amusement, and overall funniness.
"""

import json
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset.joke import HumorJudgment

JUDGE_SYSTEM_PROMPT = """\
You are a humor critic. Rate the following joke on four dimensions, \
each on a scale from 1 (lowest) to 10 (highest):

1. **Unexpectedness**: How surprising is the punchline?
2. **Cleverness**: How well-crafted is the joke's logic or wordplay?
3. **Amusement**: How funny is the joke overall?
4. **Overall**: Your holistic humor rating.

Respond ONLY with valid JSON, no other text:
{"unexpectedness": <int>, "cleverness": <int>, "amusement": <int>, \
"overall": <int>, "rationale": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = "Setup: {setup}\nPunchline: {punchline}"


def _parse_judgment(text: str) -> HumorJudgment | None:
    """Extract JSON from model output, handling markdown fences."""
    text = text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    try:
        obj = json.loads(text)
        return HumorJudgment(
            unexpectedness=np.float32(obj["unexpectedness"]),
            cleverness=np.float32(obj["cleverness"]),
            amusement=np.float32(obj["amusement"]),
            overall=np.float32(obj["overall"]),
            rationale=obj.get("rationale", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def judge_joke(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    setup: str,
    punchline: str,
    max_new_tokens: int = 256,
) -> HumorJudgment | None:
    """Rate a single joke using the model as judge."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(setup=setup, punchline=punchline),
        },
    ]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    del output_ids
    torch.cuda.empty_cache()

    return _parse_judgment(response)


def judge_jokes_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    setups: list[str],
    punchlines: list[str],
    max_retries: int = 2,
) -> list[HumorJudgment | None]:
    """Judge multiple jokes sequentially with retries."""
    results: list[HumorJudgment | None] = []
    for setup, punchline in zip(setups, punchlines):
        judgment = None
        for _ in range(max_retries + 1):
            judgment = judge_joke(model, tokenizer, setup, punchline)
            if judgment is not None:
                break
        results.append(judgment)
    return results
