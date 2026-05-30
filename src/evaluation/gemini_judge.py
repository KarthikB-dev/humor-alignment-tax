"""LLM-as-judge humor evaluation via the Gemini API (google-genai).

Drop-in alternative to claude_judge with the same public API
(`HumorScores`, `judge_jokes`, `scores_to_dict`) so the runner can switch
providers via a flag. Reuses the *exact same* system prompt and four dimensions
(`llm_judge.JUDGE_SYSTEM_PROMPT` / `JUDGE_USER_TEMPLATE`) and Gemini's native
structured output (`response_schema`) so results stay comparable and parsing
never fails the way the local Llama-1B judge does.

Requires GEMINI_API_KEY (in .env or the environment). Free-tier keys are
rate-limited, so calls retry on 429 and default to low concurrency.
"""

import asyncio
import os
import re

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from src.evaluation.llm_judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE


class HumorScores(BaseModel):
    """Schema the judge is constrained to return (1-10 per dimension)."""

    unexpectedness: int
    cleverness: int
    amusement: int
    overall: int
    rationale: str


def _retry_delay(err_text: str, attempt: int) -> float:
    """Honor the API's requested wait on a 429, falling back to exponential backoff.

    Free-tier 429s carry a `retryDelay: '54s'` / 'retry in 54.1s' hint that is far
    longer than naive backoff; respecting it is what actually lets the call through.
    """
    m = re.search(r"retry(?:Delay)?['\":\s]+(?:in\s+)?([\d.]+)s", err_text)
    if m:
        return min(float(m.group(1)) + 1.0, 75.0)
    return float(2 ** attempt + 1)


def _config(model: str, max_tokens: int, disable_thinking: bool) -> types.GenerateContentConfig:
    cfg = types.GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=HumorScores,
        max_output_tokens=max_tokens,
    )
    # Thinking adds cost/latency and can crowd out the JSON under a tight token
    # cap. The scoring task doesn't need it. Flash/Lite support budget=0; Pro
    # has a minimum thinking budget, so leave its default alone.
    if disable_thinking and ("flash" in model or "lite" in model):
        cfg.thinking_config = types.ThinkingConfig(thinking_budget=0)
    return cfg


async def _judge_one(
    client: genai.Client,
    model: str,
    setup: str,
    punchline: str,
    semaphore: asyncio.Semaphore,
    config: types.GenerateContentConfig,
    max_retries: int = 4,
) -> HumorScores | None:
    """Score one joke, retrying on rate limits. None only if it ultimately fails."""
    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=JUDGE_USER_TEMPLATE.format(setup=setup, punchline=punchline),
                    config=config,
                )
                return resp.parsed  # a HumorScores instance (or None if unparsable)
            except errors.APIError as e:
                # 429 = rate limited (common on free tier), 503 = overloaded
                if getattr(e, "code", None) in (429, 503) and attempt < max_retries:
                    await asyncio.sleep(_retry_delay(str(e), attempt))
                    continue
                print(f"  judge error ({model}): {type(e).__name__}: {e}", flush=True)
                return None
            except Exception as e:  # noqa: BLE001 — bulk job: record failure, keep going
                print(f"  judge error ({model}): {type(e).__name__}: {e}", flush=True)
                return None
    return None


async def judge_jokes(
    items: list[dict],
    model: str,
    concurrency: int = 4,
    max_tokens: int = 1024,
    disable_thinking: bool = True,
) -> list[HumorScores | None]:
    """Judge a list of jokes concurrently, preserving input order.

    Each item must have "setup" and "punchline" keys. Concurrency defaults low
    because free-tier Gemini keys have tight RPM limits.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env or the environment.")

    client = genai.Client(api_key=api_key)
    config = _config(model, max_tokens, disable_thinking)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _judge_one(client, model, it["setup"], it["punchline"], semaphore, config)
        for it in items
    ]
    return await tqdm_asyncio.gather(*tasks, desc=f"judging ({model})")


def scores_to_dict(scores: HumorScores | None) -> dict:
    """Flatten a HumorScores (or a failure) into JSON-serialisable fields."""
    if scores is None:
        return {"error": "judge_failed"}
    return {
        "unexpectedness": float(scores.unexpectedness),
        "cleverness": float(scores.cleverness),
        "amusement": float(scores.amusement),
        "overall": float(scores.overall),
        "rationale": scores.rationale,
    }
