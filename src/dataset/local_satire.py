"""Data models for source-grounded local satire generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LocalSatireSourceItem:
    """A manually written source capsule for one local/campus news item."""

    id: str
    source_title: str
    source_url: str
    source_date: str | None
    issue_phrase: str
    factual_bullets: list[str] = field(default_factory=list)
    local_tension: str = ""
    satirical_target_guess: str = ""


@dataclass
class LocalSatireGeneration:
    """One generated satirical headline for one source item."""

    id: str
    source_id: str
    model: str
    prompt_condition: str
    prompt: str
    output_headline: str
    raw_output: str
    timestamp: str = ""


@dataclass
class LocalSatireScore:
    """Human rubric score for one generated local satire headline."""

    generation_id: str
    scorer: str
    source_relevance: int
    local_specificity: int
    topic_nuance_target: int
    satirical_bite: int
    coherence_nonrandomness: int
    failure_tags: list[str] = field(default_factory=list)
    notes: str = ""
