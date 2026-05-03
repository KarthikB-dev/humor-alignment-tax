# Do LLM Safety Constraints Restrict Humor and Creativity?

*Quantifying the alignment tax on humor through distributional analysis of language model outputs.*

## Setup

Python 3.11 and `uv` required.

```bash
uv sync
```

## Repository Structure

```
src/
├── dataset/
│   ├── joke.py              # Data models (JokeEntry, DistributionalMetrics, etc.)
│   └── loader.py            # Dataset loaders for all four joke corpora
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
│   └── run_analysis.py      # Inverted-U + alignment comparison
└── utils/
    └── utils.py             # Model loading and path resolution
```

## Pipeline

### 1. Data Preparation

```bash
python -m src.scripts.prepare_data --sources r_jokes one_liner
```

### 2. Extract Distributional Metrics

```bash
python -m src.scripts.extract_metrics --sources r_jokes one_liner --limit 1000
```

### 3. LLM-as-Judge Humor Scoring

```bash
python -m src.scripts.judge_humor --sources r_jokes one_liner --limit 1000
```

### 4. Controlled Generation (Temperature Sweep)

```bash
python -m src.scripts.generate_jokes --n_jokes 50
```

### 5. Analysis

```bash
python -m src.scripts.run_analysis
```

## Hardware Requirements

Designed for a single **RTX 5070 Ti (16 GB VRAM)**:
- Llama 3.1 8B in BF16 fits in ~16 GB
- Batch size 1 for metric extraction (short joke sequences)
- Models loaded sequentially (base then aligned), with GPU memory freed between

## Key Hypotheses

1. **Inverted-U**: Humor quality peaks at moderate values of surprisal, entropy, and embedding distance
2. **Alignment shift**: RLHF compresses all three metrics toward the low end (less surprise, less entropy, closer embeddings)
3. **Temperature control**: Increasing temperature moves generated jokes through the inverted-U curve, with peak humor at moderate temperatures
