"""Aggregate local satire human scores."""

import argparse
from pathlib import Path

from src.analysis.local_satire_analysis import (
    collect_failure_tag_counts,
    summarize_by_model_and_condition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local satire score CSV.")
    parser.add_argument("--scores", default="outputs/local_satire_scores_filled.csv")
    parser.add_argument("--out", default="outputs/local_satire_summary.csv")
    parser.add_argument("--tag-out", default="outputs/local_satire_failure_tags.csv")
    args = parser.parse_args()

    summary = summarize_by_model_and_condition(args.scores)
    tag_counts = collect_failure_tag_counts(args.scores)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    tag_counts.to_csv(args.tag_out, index=False)

    print(f"Wrote summary to {args.out}")
    print(f"Wrote failure tag counts to {args.tag_out}")


if __name__ == "__main__":
    main()

