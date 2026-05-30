# Alignment Tax on Humor — Findings Report

*Do LLM safety constraints restrict humor and creativity? Quantifying the alignment
tax on Llama 3.2 1B through distributional analysis, LLM-as-judge scoring, and linear
probing.*

Generated 2026-05-29. Models: **Llama 3.2 1B** (base), **Llama 3.2 1B Instruct**
(RLHF: SFT + RS + DPO), **Llama 3.2 1B Instruct-safety** (instruct + LoRA safety
fine-tune on AdvBench/HarmBench refusals).

---

## TL;DR

- **RLHF did *not* tax humor here — it improved it.** Instruct-model jokes are rated as
  funny as the human corpus; on the same human jokes RLHF *widened* (did not compress)
  surprisal, entropy, and embedding distance.
- **The tax is a *safety-fine-tuning* tax, not an RLHF tax.** The safety fine-tune makes
  the model **refuse 74% of benign joke prompts**, collapsing humor scores.
- **The inverted-U hypothesis is not supported** on the human corpus: distributional
  metrics explain almost none of the humor-rating variance (R² ≤ 0.09).
- **The widening replicates on Chinese humor.** On Oogiri and Ruozhiba, base→instruct
  again widens surprisal/entropy/embedding distance (d 0.43–0.83), not compresses (§4d).
- **The tax is asymmetric — there is no *reverse* tax.** Fine-tuning the model to be
  funnier does *not* erode AdvBench/HarmBench safety (refusal ~93%, tied with instruct;
  safety FT is 100%). Safety is cheap to keep while adding humor; humor is not cheap to
  keep while adding safety (§4e).
- **Methodological result:** the local Llama-1B judge is unusable on generated jokes
  (~18% valid output); a single Gemini judge over corpus + generated (99.7% valid)
  removes the two-judge confound.

---

## 1. Method summary

| Component | Detail |
|---|---|
| Distributional metrics | punchline surprisal, pre-punchline entropy, setup–punchline embedding distance (`src/metrics/`) |
| Corpus | r_jokes (1,000) + one_liner (1,000) = **2,000** human jokes (full raw sets are ~190k/3.2k; capped via `--limit`) |
| Generated jokes | 7-config decoding sweep (temp 0.3–1.6 + narrow/wide nucleus), 50/config → base 159, instruct 350, safety 350 valid |
| Humor judge | **Gemini** — `gemini-2.5-flash` (bulk) + `gemini-2.5-pro` (subset agreement), structured outputs, same 4-dim rubric as the legacy Llama judge (`src/evaluation/{gemini,claude}_judge.py`, `src/scripts/claude_judge_all.py`) |
| Probing | Linear probes on hidden states, Chumor (Chinese-joke explanation correctness), 5-fold GroupKFold (`src/scripts/probe_humor.py`) |

Dimensions scored 1–10: **unexpectedness, cleverness, amusement, overall**.

---

## 2. Judge reliability (methodological finding)

The local Llama-1B judge emits valid JSON for only a fraction of generated jokes;
failures concentrate on the noisiest (high-temp, safety) outputs, biasing any kept
subset. A single strong API judge over **both** corpus and generated jokes fixes this
and removes the original two-judge confound (corpus was judged by Llama; generated were
never judged).

| Judge | Valid scores on generated jokes |
|---|---|
| Llama-1B (local) | base 49/159, instruct 46/350, safety 56/350 → **151/859 ≈ 18%** |
| Gemini Flash (structured output) | **2,851/2,859 ≈ 99.7%** (corpus + generated) |

**Judge robustness:** on a 110-joke stratified subset, Flash vs Pro agreement is
*moderate* — mean |Δ overall| = **1.52**, exact 33%, within-1 66%. → Trust within-judge
**relative** tier comparisons; treat absolute scores as judge-dependent (±~1.5).

---

## 3. The alignment-tax headline (humor quality)

Mean LLM-judge scores (Gemini Flash), generated jokes by alignment tier vs the human
corpus reference:

| Tier | n | overall | unexpectedness | cleverness | amusement |
|---|---|---|---|---|---|
| **Corpus** (human jokes) | 1993 | **6.71** | 7.00 | 6.50 | 6.71 |
| base (no alignment) | 159 | **3.67** | 4.22 | 3.77 | 3.37 |
| instruct (RLHF) | 349 | **6.77** | 6.25 | 7.40 | 6.63 |
| safety (RLHF + safety FT) | 350 | **3.24** | 3.38 | 3.49 | 2.93 |

**The pattern is non-monotonic:** base → instruct *rises* (3.67 → 6.77), then safety FT
*collapses* it (→ 3.24).

- **Instruct ≈ human corpus** (6.77 vs 6.71). RLHF produced the best joke-writer.
- **Safety fine-tuning is where the tax appears**, and it is largely **refusal-driven**:
  **258/350 (74%)** of safety-model outputs are refusals/disclaimers ("can't help with
  that", "it could cause real harm") to entirely benign joke prompts. Refusal-like
  outputs average 3.03 overall; even the non-refusals reach only 3.83.

> **Caveat — the base tier is confounded.** A base *completion* model barely follows
> "write a joke," so its low 3.67 is partly incoherent/off-format output, not a pure
> humor-distribution effect. Base-vs-instruct is therefore not a clean alignment
> contrast; the trustworthy contrast is **instruct → safety**.

---

## 4. Distributional analysis

### 4a. Inverted-U hypothesis — **not supported** (0/12)

Fitting `humor = a·z² + b·z + c` on the human corpus (metrics from the base model,
wider range), per metric × humor dimension:

- **No** combination shows a significant *negative* quadratic (the inverted-U signature).
- Surprisal and embedding-distance trend positive/monotonic; entropy's quadratics are
  negative but non-significant (overall p = 0.058, just misses α = 0.05).
- **R² ≤ 0.089 everywhere** — the distributional metrics explain almost none of the
  humor-rating variance. (`data/analysis/inverted_u_results.json`)

### 4b. Distributional shift across alignment tiers (metric-only, judge-independent)

How each metric — assigned to the *same* 2,000 human jokes — shifts as alignment
increases (`compare_safety.py`; base→instruct also in `alignment_comparison.json`):

| Metric | base→instruct | instruct→safety | base→safety |
|---|---|---|---|
| punchline_surprisal | +0.339 *** | +0.092 * | +0.431 *** |
| pre_punchline_entropy | +0.210 (ns) | +0.438 *** | +0.648 *** |
| embedding_distance | +0.035 *** | +0.008 (ns) | +0.043 *** |

*** p<0.001, * p<0.05, ns = not significant; Cohen d ranges 0.05–0.45.

**Every metric increases monotonically at every stage** — the distribution *widens* with
more alignment, the **opposite of Hypothesis 2's compression prediction**, now confirmed
for the safety tier as well. The standout: safety FT pushes **entropy +0.44 over
instruct** — the safety model is markedly more *uncertain when reading* human jokes, even
as it largely *refuses to write* them (§3). Together: worse at modeling humor, unwilling
to produce it.

### 4c. Inverted-U on *generated* jokes + temperature sweep (Hypothesis 4)

Generated jokes carry surprisal + entropy and Gemini humor scores, so they can be placed
on the curve directly (no embedding distance — `generate_jokes.py` doesn't compute it).
`src/scripts/analyze_generated.py` → `data/analysis/generated_inverted_u.json`.

**Temperature sweep — Llama-3.2-1B-Instruct (the clean tier):**

| config | temp | top_p | surprisal | entropy | overall |
|---|---|---|---|---|---|
| low_temp | 0.3 | 1.0 | 0.79 | 2.24 | **7.22** |
| mid_temp | 0.7 | 1.0 | 1.05 | 2.91 | 6.98 |
| narrow_nucleus | 1.0 | 0.5 | 0.38 | 2.56 | **7.26** |
| default | 1.0 | 1.0 | 1.65 | 2.87 | 6.96 |
| wide_nucleus | 1.0 | 0.9 | 1.96 | 3.37 | 6.84 |
| high_temp | 1.3 | 1.0 | 2.34 | 3.42 | 6.34 |
| very_high_temp | 1.6 | 1.0 | 3.26 | 5.18 | **5.76** |

As temperature rises, surprisal and entropy climb as expected — but humor **declines
monotonically** (7.22 → 5.76). **Hypothesis 4 is not supported:** there is no moderate-
temperature peak; the instruct model is funniest at **low temperature / narrow nucleus**.

**Inverted-U fits on generated jokes:**

- **Instruct (clean tier): no inverted-U** (0/8 significant negative quadratics), though
  R² is much higher than the corpus (0.15–0.32 vs ≤0.09) — the metrics relate to humor,
  just not as a peak.
- **Safety tier: shows significant inverted-U** (surprisal×overall a=−0.15, R²=0.19,
  peak≈7.7; several entropy dims) — **likely an artifact** of the 74%-refusal mixture
  (low-surprisal refusals + a few real attempts + incoherent high-surprisal outputs
  manufacture a curve), not a genuine humor sweet spot.
- **Pooled across tiers: U-shaped (positive quadratic), not inverted** — tier mixing
  dominates.

### 4d. Distributional shift on the Chinese proposal datasets (Oogiri, Ruozhiba)

The proposal names **Oogiri** (crowd-sourced Japanese/Chinese witty responses; here the
`T2T` odai→boke pairs from `zhongshsh/CLoT-Oogiri-GO`) and **Ruozhiba** (absurd-logic
Q&A; `LooksJuicy/ruozhiba`). Both were fetched (`fetch_zh_datasets.py`, 500/source) and
run through the same metric extraction across all three tiers
(`data/metrics_zh/`, `analyze_zh.py` → `data/analysis/zh_distribution.json`).

**The English finding replicates cross-lingually.** Per source, every metric *widens*
base→instruct and base→safety (Cohen d shown):

| Source | metric | base | instruct | safety | b→i d | b→s d |
|---|---|---|---|---|---|---|
| Oogiri | surprisal | 4.50 | 5.32 | 5.48 | +0.61 | +0.73 |
| Oogiri | entropy | 8.66 | 9.41 | 9.35 | +0.74 | +0.70 |
| Oogiri | distance | 0.158 | 0.211 | 0.209 | +0.74 | +0.70 |
| Ruozhiba | surprisal | 2.34 | 2.55 | 2.72 | +0.43 | +0.75 |
| Ruozhiba | entropy | 7.19 | 8.32 | 8.59 | +0.50 | +0.62 |
| Ruozhiba | distance | 0.101 | 0.135 | 0.123 | +0.83 | +0.56 |

- **base→instruct widens all 3 metrics in both datasets** (medium effects, d 0.43–0.83) —
  again the **opposite of Hypothesis 2's compression** prediction, now confirmed on
  Chinese humor.
- **base→safety widens further** (d 0.53–0.75 everywhere). The instruct→safety step is
  small/mixed (distance plateaus or dips), exactly as in English (§4b).
- **Humor-quality baseline (Gemini judge, 250/source, isolated cheap run →
  `data/judgments_zh/`):** Oogiri scores **6.76** overall — as funny as the English
  corpus (6.71) — while Ruozhiba scores **2.95**: the judge cleanly separates genuine
  crowd-humor from absurd-logic Q&A. Flash↔Pro agreement on an 80-joke subset is
  consistent with English (mean |Δ overall| = 1.32, within-1 71%).

### 4e. Reverse direction — does fine-tuning for *humor* erode safety? (proposal "if time permits")

The proposal's optional experiment is the mirror image of the main study: push the model
toward **more** humor, then test whether **safety** degrades. Built net-new:
`finetune_humor.py` (LoRA SFT of instruct on 1,257 corpus-joke completions, same recipe
as the safety FT) → `Llama-3.2-1B-Instruct-humor`; and `eval_safety.py`, the first
AdvBench/HarmBench **evaluation** harness in the repo (those sets were previously only
training data) — greedy generation + the standard refusal-substring heuristic.

Refusal rate on 120 held-out harmful prompts (higher = safer; ASR = 1 − refusal):

| Model | refusal | ASR |
|---|---|---|
| Instruct (baseline) | **0.900** | 0.100 |
| Instruct-**humor** | **0.933** | 0.067 |
| Instruct-safety | 1.000 | 0.000 |

**Humor fine-tuning did *not* erode safety** — refusal stayed ~93%, statistically tied
with the instruct baseline (±~4 prompts at n=120), nowhere near a collapse. Qualitatively
the asymmetry is even stronger: the instruct baseline's 12 non-refusals are mostly
*genuine* soft-compliance (song lyrics, "I can provide a sample hate mail", E. coli
culturing hints), whereas the humor model's 8 non-refusals are mostly *degenerate
deflection* — "I'm a little teapot" in place of requested lyrics, a cow-and-robots joke
for "spread fake news" — i.e. it falls back to its joke distribution rather than
complying. **There is no symmetric "reverse alignment tax" at this scale:** safety is
cheap to keep while adding humor, but humor is expensive to keep while adding safety (§3).

> **Heuristic caveat.** The refusal classifier is the literature-standard substring match
> over n=120; treat ±a few points as noise. One bug worth recording: the humor model
> emits **curly apostrophes** ("I can't"), so the raw ASCII markers initially mis-scored
> 10 of its refusals as compliance and *inverted* the headline. `is_refusal` now
> normalizes Unicode quotes; always normalize before substring-matching model text.

---

## 5. Mechanistic probing (Chumor humor-explanation task)

Linear probes on hidden states predicting whether an explanation correctly accounts for
a joke (majority-class baseline 56.5%):

| Model | Best layer | Accuracy | AUC |
|---|---|---|---|
| Base | 11 (middle) | 58.4% | 0.608 |
| Instruct | 16 (final) | 59.3% | 0.619 |
| Instruct-safety | 15 (near-final) | 58.7% | 0.618 |

- All three beat baseline — a real but **weak** linear humor signal (~0.61–0.62 AUC).
- **RLHF pushes humor-relevant features toward the output layers** (peak shifts from
  middle layer 11 → final layer 16), consistent with alignment reshaping late-layer
  representations.
- **Safety FT keeps the signal late and intact, not flattened.** The safety model peaks
  at layer **15** (near-final, tied with 14/16), not back at the middle — alignment's
  late-layer reshaping *persists* through safety fine-tuning — and its peak AUC (0.618)
  is essentially identical to instruct (0.619), both above base (0.608). So the safety
  fine-tune that **collapses joke *generation*** (74% refusals, §3) and **widens
  reading-time *entropy*** (§4b) leaves the probe-measurable humor-*understanding* feature
  intact. The tax falls on production and distribution, not on this comprehension signal.
  (`data/probes/probe_results.md`)

---

## 6. Hypothesis scorecard

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | Inverted-U: humor peaks at moderate metric values | **Not supported** — corpus 0/12; instruct-generated 0/8 (R² up to 0.32). Significant inverted-U appears only in the safety tier, likely a refusal-mixture artifact |
| 2 | RLHF compresses surprisal/entropy/embedding distance | **Contradicted** — RLHF widened all three |
| 3 | Safety fine-tuning shifts distributions / degrades humor further | **Supported** — humor collapses (74% refusals); and safety FT *further widens* corpus metrics (entropy +0.44 over instruct), not compresses. Widening replicates on Chinese Oogiri/Ruozhiba (§4d) |
| 4 | Temperature moves generated jokes along the inverted-U | **Not supported** — for instruct, temp 0.3→1.6 *monotonically lowers* humor (7.22→5.76) as surprisal/entropy rise; funniest at low temp / narrow nucleus |
| 5 | Fine-tuning for humor erodes safety (reverse alignment tax) | **Not supported** — humor FT keeps refusal ~93% (tied with instruct 90%; safety FT 100%); the tax is asymmetric (§4e) |

---

## 7. Caveats

- **Single model scale (1B).** Findings may not generalize to larger models.
- **Absolute judge scores are judge-dependent** (Flash↔Pro ±~1.5); rely on relative
  ordering within one judge.
- **Base tier confounded** by instruction-following (§3).
- **Corpus capped at 2,000** (1,000/source) to match the original analysis scale and
  cost; the full datasets are ~190k.
- **Safety degradation is refusal-dominated** — partly a behavioral (refusal) effect,
  not purely a humor-quality effect.

---

## 8. Open / next steps

1. ~~**Probe the safety model** on Chumor.~~ **Done (§5).**
2. ~~**Oogiri / Ruozhiba distributional analysis.**~~ **Done (§4d):** fetched both, metrics
   replicate the widening result, Gemini humor baseline computed.
3. ~~**Reverse-direction humor-FT → safety experiment.**~~ **Done (§4e):** built the
   humor FT + the AdvBench/HarmBench eval harness; no reverse tax found.
4. **Human-rating validation subset** — 120-joke blind sheet built
   (`build_human_rating_subset.py` → `data/human_eval/human_rating_sheet.csv`, stratified
   across all tiers × configs). *Awaiting hand ratings*; `score_human_ratings.py` then
   reports human↔LLM correlation per dimension.
5. **Publication plots** for the new findings (per-tier humor bars, temperature-sweep
   curve, generated-joke inverted-U fits) via `generate_plots.py`.

*Done since first draft:* generated-joke inverted-U + temperature sweep (§4c); 3-way
metric comparison (§4b); safety-model probe (§5); Chinese-dataset replication +
humor baseline (§4d); humor-FT reverse-tax experiment (§4e); human-rating sheet (#4).

---

### Artifacts

- Humor judgments: `data/judgments/gemini_bulk/` (corpus + per-tier generated),
  `data/judgments/gemini_subset/subset_agreement.json`;
  Chinese: `data/judgments_zh/` (Oogiri/Ruozhiba bulk + subset agreement)
- Analysis: `data/analysis/inverted_u_results.json`, `alignment_comparison.json`,
  `generated_inverted_u.json`, `zh_distribution.json`
- Metrics: `data/metrics/*.npz` (English corpus), `data/metrics_zh/*.npz` (Oogiri/Ruozhiba)
- Probes: `data/probes/probe_results.md`
- Reverse experiment: `models/Llama-3.2-1B-Instruct-humor/`, `data/safety_eval/` (per-prompt
  responses + `safety_eval_summary.json`)
- Human-rating subset: `data/human_eval/` (blind sheet, hidden key, agreement output)
- Judge code: `src/evaluation/{gemini,claude}_judge.py`, `src/scripts/claude_judge_all.py`
- New scripts: `fetch_zh_datasets.py`, `analyze_zh.py`, `finetune_humor.py`,
  `eval_safety.py`, `build_human_rating_subset.py`, `score_human_ratings.py`
- Figures (`presentation/figures/`, via `generate_plots.py`): `humor_by_tier`,
  `humor_temperature`, `generated_inverted_u` (new humor figures), plus the existing
  `metric_means` / `distributions` / `shifts` / `temperature_sweep` / `training_loss`
