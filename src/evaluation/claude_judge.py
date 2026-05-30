"""LLM-as-judge humor evaluation via the Claude API.

A stronger, more reliable judge than the local Llama-1B judge. Reuses the *exact
same* system prompt and four dimensions (`llm_judge.JUDGE_SYSTEM_PROMPT` /
`JUDGE_USER_TEMPLATE`) so Claude and Llama scores are directly comparable, and
uses structured outputs (`output_format`) so the API guarantees a schema-valid
score object — no JSON-parse failures like the local judge.

Used to judge both the human corpus and the model-generated jokes with one
consistent judge, removing the two-judge confound.
"""

import asyncio

from anthropic import AsyncAnthropic
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from src.evaluation.llm_judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE


class HumorScores(BaseModel):
    """Schema the Claude judge is constrained to return (1-10 per dimension)."""

    unexpectedness: int
    cleverness: int
    amusement: int
    overall: int
    rationale: str


# The judge system prompt is short (~150 tokens), below the per-model minimum
# cacheable prefix (2048-4096 tokens), so this breakpoint won't actually cache
# today — it's harmless and future-proofs the call if the rubric grows.
_SYSTEM = [{"type": "text", "text": JUDGE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]


async def _judge_one(
    client: AsyncAnthropic,
    model: str,
    setup: str,
    punchline: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 512,
) -> HumorScores | None:
    """Score a single joke. Returns None only if the API call ultimately fails."""
    async with semaphore:
        try:
            resp = await client.messages.parse(
                model=model,
                max_tokens=max_tokens,
                system=_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_USER_TEMPLATE.format(setup=setup, punchline=punchline),
                    }
                ],
                output_format=HumorScores,
            )
            return resp.parsed_output
        except Exception as e:  # noqa: BLE001 — bulk job: record failure, keep going
            print(f"  judge error ({model}): {type(e).__name__}: {e}", flush=True)
            return None


async def judge_jokes(
    items: list[dict],
    model: str,
    concurrency: int = 8,
    max_tokens: int = 512,
) -> list[HumorScores | None]:
    """Judge a list of jokes concurrently, preserving input order.

    Each item must have "setup" and "punchline" keys. The SDK auto-retries 429s
    and 5xx with backoff; `concurrency` bounds simultaneous in-flight requests.
    """
    semaphore = asyncio.Semaphore(concurrency)
    async with AsyncAnthropic() as client:
        tasks = [
            _judge_one(client, model, it["setup"], it["punchline"], semaphore, max_tokens)
            for it in items
        ]
        return await tqdm_asyncio.gather(*tasks, desc=f"judging ({model})")


def scores_to_dict(scores: HumorScores | None) -> dict:
    """Flatten a HumorScores (or a parse failure) into JSON-serialisable fields."""
    if scores is None:
        return {"error": "judge_failed"}
    return {
        "unexpectedness": float(scores.unexpectedness),
        "cleverness": float(scores.cleverness),
        "amusement": float(scores.amusement),
        "overall": float(scores.overall),
        "rationale": scores.rationale,
    }
