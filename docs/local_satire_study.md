# Source-Grounded Local Satire Study

This qualitative mini-study tests whether LLMs can generate Onion- or
Nexustentialism-style local satire from manually written campus/local news source
capsules.

It is not a RAG system. The scripts do not perform web search, retrieval, or live
source lookup. Source capsules are created manually so the comparison focuses on
how each model uses the supplied source facts rather than whether retrieval found
existing jokes or satire about the same topic.

Generation uses controlled manual source capsules only; evaluator annotations
are not fed to the model, and any future search-assisted explanation condition
should remain separate from generation.

## Research Question

Given a real local/campus news item, can an LLM generate a satirical headline
that preserves source relevance, local specificity, topic nuance / satirical
target, satirical bite, and coherence / non-randomness?

## Prompt Conditions

The main comparison is across three prompt conditions:

1. `generic_issue`: only the issue phrase.
2. `source_grounded`: source title, factual bullets, and local tension.
3. `target_first_rank`: full source capsule, target identification, five
   candidate headlines, and self-selection using the human rubric dimensions.

## Workflow

Create or edit manual source capsules:

```bash
uv run python - <<'PY'
from src.dataset.local_satire_loader import save_local_satire_sources_template
save_local_satire_sources_template("data/local_satire_sources.jsonl")
PY
```

Preview prompts without API calls:

```bash
uv run python -m src.scripts.generate_local_satire \
  --sources data/local_satire_sources.jsonl \
  --models gemini-3.5-flash,gpt-5.4-mini \
  --conditions generic_issue,source_grounded,target_first_rank \
  --dry-run
```

Generate headlines:

```bash
uv run python -m src.scripts.generate_local_satire \
  --sources data/local_satire_sources.jsonl \
  --out outputs/local_satire_generations_gemini35_oai54mini.csv \
  --models gemini-3.5-flash,gpt-5.4-mini \
  --conditions generic_issue,source_grounded,target_first_rank \
  --temperature 0.7
```

Create a manual score sheet and rubric:

```bash
uv run python -m src.scripts.create_local_satire_score_sheet \
  --generations outputs/local_satire_generations_gemini35_oai54mini.csv \
  --out outputs/local_satire_scores_template_gemini35_oai54mini.csv \
  --rubric-out outputs/local_satire_rubric.md
```

After filling the score sheet, aggregate results:

```bash
uv run python -m src.scripts.analyze_local_satire \
  --scores outputs/local_satire_scores_completed_with_human_baseline.csv \
  --out outputs/local_satire_summary_gemini35_oai54mini_human_baseline.csv \
  --tag-out outputs/local_satire_failure_tags_gemini35_oai54mini_human_baseline.csv
```

Optional plots:

```bash
uv run python -m src.scripts.plot_local_satire \
  --scores outputs/local_satire_scores_completed_with_human_baseline.csv \
  --out outputs/local_satire_total_scores_gemini35_oai54mini_human_baseline.png \
  --dimensions-out outputs/local_satire_dimensions_gemini35_oai54mini_human_baseline.png
```

## Current Artifacts

- `data/local_satire_sources.jsonl`: six manually written source capsules.
- `outputs/local_satire_generations_gemini35_oai54mini.csv`: 36 generated headlines from `gemini-3.5-flash` and `gpt-5.4-mini`.
- `outputs/local_satire_scores_template_gemini35_oai54mini.csv`: blank scoring template for the generated headlines.
- `outputs/local_satire_scores_completed_with_human_baseline.csv`: completed score sheet including human-baseline rows.
- `outputs/local_satire_scores_completed_with_human_baseline.xlsx`: spreadsheet copy of the completed scores.
- `outputs/local_satire_summary_gemini35_oai54mini_human_baseline.csv`: mean scores by model and prompt condition.
- `outputs/local_satire_failure_tags_gemini35_oai54mini_human_baseline.csv`: failure-tag counts by model and prompt condition.
- `outputs/local_satire_total_scores_gemini35_oai54mini_human_baseline.png`: total-score plot.
- `outputs/local_satire_dimensions_gemini35_oai54mini_human_baseline.png`: rubric-dimension plot.

## Evaluation

Human scoring separates grounding dimensions from humor dimensions. Each metric
is scored from 1 to 5:

- Source relevance
- Local specificity
- Topic nuance / satirical target
- Satirical bite
- Coherence / non-randomness

Failure tags capture interpretable failure modes such as generic Onion template,
random absurdity, hallucinated local detail, missing the source tension, or
safety/refusal hedging.
