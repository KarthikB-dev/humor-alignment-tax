"""Create a human scoring sheet and rubric for local satire generations."""

import argparse
import csv
from pathlib import Path

from src.dataset.local_satire_loader import (
    SCORE_FIELDNAMES,
    load_local_satire_generations,
)

RUBRIC = """# Local Satire Human Scoring Rubric

Score each metric from 1 (lowest) to 5 (highest).

## Metrics

Source relevance: Does the headline clearly respond to the source item?

Local specificity: Does it feel specific to UCSB/Isla Vista/Santa Barbara rather than generic college-town satire?

Topic nuance / satirical target: Does it identify the underlying institution, behavior, or social tension being mocked?

Satirical bite: Is there a clever incongruity or sharp comic turn?

Coherence / non-randomness: Does the absurdity make sense rather than being random?

## Failure Tags

Use comma-separated tags in the failure_tags column when applicable:

generic_onion_template
bland_news_summary
random_absurdity
hallucinated_local_detail
misses_source_tension
grounded_but_not_funny
funny_but_not_grounded
template_trap
refusal_or_safety_hedging
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a manual score sheet for local satire generations."
    )
    parser.add_argument(
        "--generations", default="outputs/local_satire_generations.csv"
    )
    parser.add_argument("--out", default="outputs/local_satire_scores_template.csv")
    parser.add_argument("--rubric-out", default="outputs/local_satire_rubric.md")
    args = parser.parse_args()

    generations = load_local_satire_generations(args.generations)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDNAMES)
        writer.writeheader()
        for gen in generations:
            writer.writerow(
                {
                    "generation_id": gen.id,
                    "source_id": gen.source_id,
                    "model": gen.model,
                    "prompt_condition": gen.prompt_condition,
                    "output_headline": gen.output_headline,
                    "source_relevance": "",
                    "local_specificity": "",
                    "topic_nuance_target": "",
                    "satirical_bite": "",
                    "coherence_nonrandomness": "",
                    "failure_tags": "",
                    "notes": "",
                    "scorer": "",
                }
            )

    rubric_path = Path(args.rubric_out)
    rubric_path.parent.mkdir(parents=True, exist_ok=True)
    rubric_path.write_text(RUBRIC)

    print(f"Wrote score sheet to {out_path}")
    print(f"Wrote rubric to {rubric_path}")


if __name__ == "__main__":
    main()

