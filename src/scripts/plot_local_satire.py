"""Generate plots for local satire scores."""

import argparse

from src.analysis.local_satire_plotting import plot_local_satire_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot local satire score results.")
    parser.add_argument("--scores", default="outputs/local_satire_scores_filled.csv")
    parser.add_argument("--out", default="outputs/local_satire_total_scores.png")
    parser.add_argument(
        "--dimensions-out", default="outputs/local_satire_dimensions.png"
    )
    args = parser.parse_args()

    plot_local_satire_scores(args.scores, args.out, args.dimensions_out)
    print(f"Wrote total score plot to {args.out}")
    print(f"Wrote dimension plot to {args.dimensions_out}")


if __name__ == "__main__":
    main()

