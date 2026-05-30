# Pending Work

Forward-looking open items. Completed work is archived at the bottom; full findings
live in `alignment_tax_report.md`.

---

## Open

### 1. Human ratings on the validation subset  *(needs a human — you)*
The blind sheet is built and waiting: `data/human_eval/human_rating_sheet.csv` — 120
jokes stratified across all tiers × decoding configs (22 strata), no LLM scores shown.

- Fill the `human_unexpectedness / cleverness / amusement / overall` columns (1–10).
- Then: `uv run python -m src.scripts.score_human_ratings` → human↔LLM Pearson/Spearman/
  MAE per dimension (`data/human_eval/human_llm_agreement.json`). Runs on a partially
  filled sheet too, for an interim read.
- This is the proposal's "human ratings on a subset" deliverable and validates the judge.

### 2. Get the new findings into the figures + decks
The report has two results with **no plots and not in the beamer/ODP** yet:

- **§4d Chinese replication** (Oogiri/Ruozhiba widening; `data/analysis/zh_distribution.json`,
  `data/metrics_zh/`) — wants a per-source shift bar chart like the English `shifts.pdf`.
- **§4e reverse tax** (humor-FT vs safety refusal rates; `data/safety_eval/
  safety_eval_summary.json`) — wants a simple 3-bar refusal/ASR chart.
- Add `plot_*` fns to `generate_plots.py`, then add frames to `alignment_tax.tex` and
  slides via `generate_odp.py` (mirror how the §3/§4c humor figures were added).

### 3. Optional polish
- Rotate the **Gemini API key** in `.env` — it was pasted into a chat transcript.
- `generate_plots.py` `plot_temperature()` labels say "High (T=1.4)" but `high_temp`
  is T=1.3, and it omits the `very_high_temp` / nucleus configs — cosmetic.
- `judge_humor.py` (legacy local-judge path) still loads only the **corpus**; it never
  judged the generated jokes — superseded by the Gemini judge but left as-is.

### 4. Further-reach (net-new, not started)
- **Scale up the safety eval** beyond n=120 (full AdvBench+HarmBench ~900 prompts) and/or
  swap the substring heuristic for an LLM-judge refusal classifier — the ±few-prompt
  noise currently makes humor-FT vs instruct a statistical tie (§4e).
- **Probe the humor-FT model** on Chumor to complete the 4-tier probing picture (§5).

---

## Done

*2026-05-30:*
- **Safety-model probe** (§5) — peak stays late (base 11 → instruct 16 → safety 15), AUC
  intact (0.618 ≈ instruct); the tax is on generation/reading, not comprehension.
- **Oogiri & Ruozhiba** (§4d, was open #2) — fetched via `fetch_zh_datasets.py`, metrics
  to `data/metrics_zh/`, `analyze_zh.py`; English widening **replicates** cross-lingually.
  Gemini humor baseline: Oogiri 6.76 ≈ corpus, Ruozhiba 2.95.
- **Humor-FT → safety reverse experiment** (§4e, was open #4) — net-new `finetune_humor.py`
  (→ `models/Llama-3.2-1B-Instruct-humor`) + `eval_safety.py` (first AdvBench/HarmBench
  eval harness). **No reverse tax:** refusal ~93% vs instruct 90%, safety FT 100%.
  Gotcha: humor model uses curly apostrophes → ASCII refusal markers first inverted the
  headline; `is_refusal` now normalizes Unicode quotes.
- **Blind human-rating sheet built** (open #1, awaiting ratings) — `build_human_rating_subset.py`.

*Prior session:* Gemini judge over corpus + generated (replaced ~18%-valid Llama judge),
3-way humor + metric comparison, generated-joke inverted-U + temperature sweep, refreshed
beamer + ODP decks.
