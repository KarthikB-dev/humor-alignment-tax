"""Analysis helpers for local satire human scores."""

import pandas as pd

RUBRIC_COLUMNS = [
    "source_relevance",
    "local_specificity",
    "topic_nuance_target",
    "satirical_bite",
    "coherence_nonrandomness",
]


def aggregate_scores(scores_path: str) -> pd.DataFrame:
    """Load score CSV and add total/mean score columns."""
    df = pd.read_csv(scores_path)
    missing = [col for col in RUBRIC_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {', '.join(missing)}")
    for col in RUBRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["total_score"] = df[RUBRIC_COLUMNS].sum(axis=1, min_count=len(RUBRIC_COLUMNS))
    df["mean_score"] = df[RUBRIC_COLUMNS].mean(axis=1)
    return df


def summarize_by_model_and_condition(scores_path: str) -> pd.DataFrame:
    """Summarize rubric dimensions by model and prompt condition."""
    df = aggregate_scores(scores_path)
    group_cols = ["model", "prompt_condition"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("generation_id", "count"),
            mean_total_score=("total_score", "mean"),
            mean_score=("mean_score", "mean"),
            source_relevance=("source_relevance", "mean"),
            local_specificity=("local_specificity", "mean"),
            topic_nuance_target=("topic_nuance_target", "mean"),
            satirical_bite=("satirical_bite", "mean"),
            coherence_nonrandomness=("coherence_nonrandomness", "mean"),
        )
        .reset_index()
    )
    return summary


def collect_failure_tag_counts(scores_path: str) -> pd.DataFrame:
    """Count comma- or semicolon-separated failure tags."""
    df = pd.read_csv(scores_path)
    if "failure_tags" not in df.columns:
        raise ValueError("Missing failure_tags column")

    rows = []
    for _, row in df.iterrows():
        raw_tags = row.get("failure_tags")
        if pd.isna(raw_tags) or not str(raw_tags).strip():
            continue
        tags = [
            tag.strip()
            for tag in str(raw_tags).replace(";", ",").split(",")
            if tag.strip()
        ]
        for tag in tags:
            rows.append(
                {
                    "failure_tag": tag,
                    "model": row.get("model", ""),
                    "prompt_condition": row.get("prompt_condition", ""),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["failure_tag", "model", "prompt_condition", "count"]
        )

    tag_df = pd.DataFrame(rows)
    return (
        tag_df.groupby(["failure_tag", "model", "prompt_condition"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "failure_tag"], ascending=[False, True])
    )

