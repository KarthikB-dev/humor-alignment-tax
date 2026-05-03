"""Prepare and validate joke datasets."""

import argparse
import json
from pathlib import Path

from src.dataset.joke import DatasetSource
from src.dataset.loader import load_all_jokes

PROCESSED_DIR = "data/processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and preprocess joke datasets.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["r_jokes", "one_liner"],
        choices=[s.value for s in DatasetSource],
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sources = [DatasetSource(s) for s in args.sources]
    jokes = load_all_jokes(sources=sources, limit_per_source=args.limit)

    print(f"\nTotal jokes loaded: {len(jokes)}")

    # Print source distribution
    from collections import Counter

    source_counts = Counter(j.source.value for j in jokes)
    lang_counts = Counter(j.language.value for j in jokes)

    print("\nBy source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    print("\nBy language:")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}")

    # Print length statistics
    setup_lens = [len(j.setup.split()) for j in jokes]
    punch_lens = [len(j.punchline.split()) for j in jokes]
    print(f"\nSetup length (words):     mean={sum(setup_lens)/len(setup_lens):.1f}")
    print(f"Punchline length (words): mean={sum(punch_lens)/len(punch_lens):.1f}")

    # Save processed data
    out_path = Path(PROCESSED_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    processed = []
    for j in jokes:
        processed.append(
            {
                "joke_id": j.joke_id,
                "text": j.text,
                "setup": j.setup,
                "punchline": j.punchline,
                "source": j.source.value,
                "language": j.language.value,
            }
        )

    output_file = out_path / "jokes.json"
    with open(output_file, "w") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(processed)} jokes to {output_file}")


if __name__ == "__main__":
    main()
