MAKEFLAGS += --no-print-directory

.PHONY: install clean format lint typecheck test \
        data metrics evaluate analyze all

# --------
# Setup
# --------
install:
	@echo "--- MAKE: install ---"
	@uv sync

clean:
	@echo "--- MAKE: clean ---"
	@rm -rf .ruff_cache .pytest_cache .ty_cache dist build outputs .venv

# --------
# Quality gates
# --------
format:
	@echo "--- MAKE: format ---"
	@uv run black .
	@uv run ruff check . --fix

lint:
	@echo "--- MAKE: lint ---"
	@uv run ruff check .

typecheck:
	@echo "--- MAKE: typecheck ---"
	@uv run ty check

test:
	@echo "--- MAKE: test ---"
	@export PYTHONPATH=$$PYTHONPATH:$(PWD)
	@uv run pytest

# --------
# Research pipeline
# --------
data:
	@echo "--- MAKE: data (load and preprocess jokes) ---"
	@make all
	@uv run python -m src.scripts.prepare_data

metrics:
	@echo "--- MAKE: metrics (extract distributional metrics) ---"
	@make all
	@uv run python -m src.scripts.extract_metrics

evaluate:
	@echo "--- MAKE: evaluate (LLM-as-judge humor scoring) ---"
	@make all
	@uv run python -m src.scripts.judge_humor

analyze:
	@echo "--- MAKE: analyze (inverted-U and alignment comparison) ---"
	@make all
	@uv run python -m src.scripts.run_analysis

generate:
	@echo "--- MAKE: generate (generate jokes with varying decoding) ---"
	@make all
	@uv run python -m src.scripts.generate_jokes

# --------
# Convenience
# --------
all:
	@make format lint typecheck
