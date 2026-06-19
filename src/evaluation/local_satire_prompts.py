"""Prompt templates for local satire headline generation."""

from src.dataset.local_satire import LocalSatireSourceItem

PROMPT_CONDITIONS = ("generic_issue", "source_grounded", "target_first_rank")


def build_local_satire_prompt(item: LocalSatireSourceItem, condition: str) -> str:
    """Build a prompt for a source item and prompt condition."""
    if condition == "generic_issue":
        return _generic_issue_prompt(item)
    if condition == "source_grounded":
        return _source_grounded_prompt(item)
    if condition == "target_first_rank":
        return _target_first_rank_prompt(item)
    raise ValueError(
        f"Unknown prompt condition '{condition}'. "
        f"Expected one of: {', '.join(PROMPT_CONDITIONS)}"
    )


def _generic_issue_prompt(item: LocalSatireSourceItem) -> str:
    return f"""Write one Onion-style local satire headline about this issue:
Issue: {item.issue_phrase}
The headline should be funny, locally specific, and have a clear satirical target.
Output only one headline."""


def _source_grounded_prompt(item: LocalSatireSourceItem) -> str:
    facts = "\n".join(f"* {fact}" for fact in item.factual_bullets)
    return f"""Write one Onion-style local satire headline based on the source below.

Source title:
{item.source_title}

Facts:
{facts}

Local tension:
{item.local_tension}

Requirements:
* clearly grounded in the source issue
* specific to UCSB, Isla Vista, or Santa Barbara when appropriate
* clear satirical target
* avoid random absurdity
* avoid inventing facts not implied by the source
* output only one headline."""


def _target_first_rank_prompt(item: LocalSatireSourceItem) -> str:
    facts = "\n".join(f"* {fact}" for fact in item.factual_bullets)
    return f"""Write source-grounded Onion-style local satire from this campus/local source capsule.

Source title:
{item.source_title}

Source date:
{item.source_date or "unknown"}

Issue phrase:
{item.issue_phrase}

Facts:
{facts}

Local tension:
{item.local_tension}

Step 1: identify the satirical target in one short phrase.
Step 2: generate 5 candidate Onion-style local satire headlines.
Step 3: select the best using source relevance, local specificity, topic nuance/target, satirical bite, and coherence/non-randomness.

Output only:
Target: ...
Best headline: ...
Brief reason: ..."""
