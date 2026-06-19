# Do LLM Safety Constraints Restrict Humor and Creativity?

*Quantifying the alignment tax on humor through distributional analysis of language model outputs.*

A study into the effects of safety alignment on AI humor, measured through punchline surprisal, pre-punchline entropy, and embedding distance.

## Setup

Python 3.11 and `uv` required.

```bash
uv sync
cp .env.example .env   # fill in model paths and API keys
```

## Repository Structure

```
src/
├── dataset/
│   ├── joke.py              # Data models (JokeEntry, DecodingConfig, etc.)
│   └── loader.py            # Dataset loaders for r_jokes and one_liner corpora
├── metrics/
│   ├── surprisal.py         # Punchline surprisal extraction
│   ├── entropy.py           # Pre-punchline entropy computation
│   └── embedding.py         # Setup-punchline embedding distance
├── evaluation/
│   └── llm_judge.py         # LLM-as-judge humor scoring
├── analysis/
│   ├── inverted_u.py        # Inverted-U curve fitting and testing
│   ├── alignment_comparison.py  # Base vs aligned distributional shift tests
│   └── plotting.py          # Publication-quality figures
├── scripts/
│   ├── prepare_data.py      # Load and validate joke datasets
│   ├── extract_metrics.py   # Extract distributional metrics across models
│   ├── judge_humor.py       # LLM-as-judge humor evaluation
│   ├── generate_jokes.py    # Controlled generation with temperature sweep
│   ├── run_analysis.py      # Inverted-U + alignment comparison
│   ├── finetune_safety.py   # LoRA safety fine-tune on AdvBench + HarmBench
│   ├── compare_safety.py    # Three-way metric comparison (base/instruct/safety)
│   ├── generate_plots.py    # Generate all presentation figures
│   └── generate_odp.py      # Generate LibreOffice ODP presentation
└── utils/
    └── utils.py             # Model loading and path resolution
```

## Pipeline

### 1. Data Preparation

```bash
uv run python -m src.scripts.prepare_data --sources r_jokes one_liner
```

### 2. Extract Distributional Metrics

```bash
uv run python -m src.scripts.extract_metrics --sources r_jokes one_liner --limit 2000
```

### 3. LLM-as-Judge Humor Scoring

```bash
uv run python -m src.scripts.judge_humor --sources r_jokes one_liner --limit 2000
```

### 4. Controlled Generation (Temperature Sweep)

```bash
uv run python -m src.scripts.generate_jokes --n_jokes 50
```

### 5. Analysis

```bash
uv run python -m src.scripts.run_analysis
```

### 6. Safety Fine-tuning (Optional)

Fine-tune the instruct model on AdvBench + HarmBench refusal pairs, then re-extract metrics for three-way comparison:

```bash
uv run python -m src.scripts.finetune_safety
uv run python -m src.scripts.extract_metrics --models LLAMA_3_2_1B_INSTRUCT_SAFETY
uv run python -m src.scripts.compare_safety
```

### 7. Presentation

```bash
uv run python -m src.scripts.generate_plots
python3 src/scripts/generate_odp.py   # requires system odfpy
cd presentation && pdflatex alignment_tax.tex && pdflatex alignment_tax.tex
```

### 8. Source-Grounded Local Satire Study (Optional)

This independent qualitative study tests whether models can generate
Onion/Nexustentialism-style local satire from manually written campus/local news
source capsules. It is not RAG: the scripts do not do web search or retrieval,
and evaluator annotations are not fed to generation prompts.

```bash
uv run python -m src.scripts.generate_local_satire \
  --sources data/local_satire_sources.jsonl \
  --models gemini-3.5-flash,gpt-5.4-mini \
  --conditions generic_issue,source_grounded,target_first_rank \
  --dry-run

uv run python -m src.scripts.generate_local_satire \
  --sources data/local_satire_sources.jsonl \
  --out outputs/local_satire_generations_gemini35_oai54mini.csv \
  --models gemini-3.5-flash,gpt-5.4-mini \
  --conditions generic_issue,source_grounded,target_first_rank \
  --temperature 0.7

uv run python -m src.scripts.create_local_satire_score_sheet \
  --generations outputs/local_satire_generations_gemini35_oai54mini.csv \
  --out outputs/local_satire_scores_template_gemini35_oai54mini.csv

uv run python -m src.scripts.analyze_local_satire \
  --scores outputs/local_satire_scores_completed_with_human_baseline.csv \
  --out outputs/local_satire_summary_gemini35_oai54mini_human_baseline.csv \
  --tag-out outputs/local_satire_failure_tags_gemini35_oai54mini_human_baseline.csv
```

See `docs/local_satire_study.md` for the full workflow and rubric.

Current local-satire artifacts:

- `data/local_satire_sources.jsonl`: manual source capsules
- `outputs/local_satire_generations_gemini35_oai54mini.csv`: generated headlines
- `outputs/local_satire_scores_completed_with_human_baseline.csv`: completed rubric scores
- `outputs/local_satire_summary_gemini35_oai54mini_human_baseline.csv`: grouped score summary

## Hardware Requirements

Designed for a single **RTX 5070 Ti (16 GB VRAM)**:
- Llama 3.2 1B in BF16 fits comfortably in VRAM
- Batch size 1 for metric extraction (short joke sequences)
- Models loaded sequentially (base then aligned), with GPU memory freed between

## Key Hypotheses

1. **Inverted-U**: Humor quality peaks at moderate values of surprisal, entropy, and embedding distance
2. **Alignment shift**: RLHF compresses all three metrics (less surprise, less entropy, closer embeddings)
3. **Safety fine-tuning**: Additional safety training on AdvBench/HarmBench further shifts distributions
4. **Temperature control**: Increasing temperature moves generated jokes through the inverted-U curve
