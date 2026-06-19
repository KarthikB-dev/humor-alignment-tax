"""Loaders and writers for the local satire qualitative study."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.dataset.local_satire import (
    LocalSatireGeneration,
    LocalSatireScore,
    LocalSatireSourceItem,
)


GENERATION_FIELDNAMES = [
    "id",
    "source_id",
    "model",
    "prompt_condition",
    "prompt",
    "output_headline",
    "raw_output",
    "timestamp",
]

SCORE_FIELDNAMES = [
    "generation_id",
    "source_id",
    "model",
    "prompt_condition",
    "output_headline",
    "source_relevance",
    "local_specificity",
    "topic_nuance_target",
    "satirical_bite",
    "coherence_nonrandomness",
    "failure_tags",
    "notes",
    "scorer",
]


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def load_local_satire_sources(path: str) -> list[LocalSatireSourceItem]:
    """Load manually authored source capsules from JSONL."""
    items = []
    for row in _read_jsonl(path):
        if not isinstance(row.get("factual_bullets", []), list):
            raise ValueError(f"{row.get('id', '<unknown>')}: factual_bullets must be a list")
        if "satirical_target_guess" not in row and "human_target_annotation" in row:
            row["satirical_target_guess"] = row["human_target_annotation"]
        allowed_fields = LocalSatireSourceItem.__dataclass_fields__
        filtered_row = {key: value for key, value in row.items() if key in allowed_fields}
        items.append(LocalSatireSourceItem(**filtered_row))
    return items


def save_local_satire_sources_template(path: str) -> None:
    """Write a small JSONL template for manually entering source capsules."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        LocalSatireSourceItem(
            id="example_001",
            source_title="EXAMPLE ONLY: Campus Committee Reviews Bike Path Signage",
            source_url="https://example.edu/local-news/example-001",
            source_date=None,
            issue_phrase="campus bike path signage confusion",
            factual_bullets=[
                "Example-only capsule; replace with a real source before analysis.",
                "Students say signs near a busy path are hard to interpret.",
            ],
            local_tension="Safety messaging competes with daily campus habits.",
            satirical_target_guess="bureaucratic signage solutions",
        ),
    ]
    with open(out_path, "w") as f:
        for item in examples:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def load_local_satire_generations(path: str) -> list[LocalSatireGeneration]:
    """Load generated headlines from CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [LocalSatireGeneration(**row) for row in reader]


def save_local_satire_generations(
    path: str, rows: list[LocalSatireGeneration]
) -> None:
    """Save generated headlines to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GENERATION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_local_satire_scores(path: str) -> list[LocalSatireScore]:
    """Load human score rows from CSV."""
    scores = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(
                LocalSatireScore(
                    generation_id=row["generation_id"],
                    scorer=row.get("scorer", ""),
                    source_relevance=_parse_score(row.get("source_relevance", "")),
                    local_specificity=_parse_score(row.get("local_specificity", "")),
                    topic_nuance_target=_parse_score(row.get("topic_nuance_target", "")),
                    satirical_bite=_parse_score(row.get("satirical_bite", "")),
                    coherence_nonrandomness=_parse_score(
                        row.get("coherence_nonrandomness", "")
                    ),
                    failure_tags=_parse_tags(row.get("failure_tags", "")),
                    notes=row.get("notes", ""),
                )
            )
    return scores


def _parse_score(value: str) -> int:
    if value == "":
        return 0
    score = int(value)
    if score < 1 or score > 5:
        raise ValueError(f"Scores must be 1-5, got {score}")
    return score


def _parse_tags(value: str) -> list[str]:
    if not value:
        return []
    return [tag.strip() for tag in value.replace(";", ",").split(",") if tag.strip()]
