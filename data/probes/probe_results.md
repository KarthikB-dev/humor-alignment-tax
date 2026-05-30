# Linear Probe Results: Humor Understanding in Llama 3.2 1B

## Experiment

**Task:** Predict whether an explanation correctly accounts for why a Chinese joke (from Ruozhiba) is funny, using only a linear probe on the model's hidden states.

**Dataset:** MichiganNLP/Chumor (3,339 entries — 1,452 correct, 1,887 incorrect explanations across 1,932 unique jokes)

**Method:** For each (joke, explanation) pair, extract the last-token hidden state at every layer. Train logistic regression probes per layer using 5-fold GroupKFold (grouped by joke to prevent leakage). Features are StandardScaled before fitting.

**Majority-class baseline:** 56.5% (predicting all "incorrect")

## Results: Llama 3.2 1B Base

| Layer | Accuracy | Std | AUC | Std |
|-------|----------|-----|-----|-----|
| 0 (embed) | 0.5642 | 0.0204 | 0.4998 | 0.0027 |
| 1 | 0.5606 | 0.0229 | 0.5828 | 0.0174 |
| 2 | 0.5744 | 0.0165 | 0.5789 | 0.0155 |
| 3 | 0.5753 | 0.0274 | 0.5962 | 0.0212 |
| 4 | 0.5598 | 0.0100 | 0.5773 | 0.0189 |
| 5 | 0.5621 | 0.0146 | 0.5821 | 0.0162 |
| 6 | 0.5675 | 0.0131 | 0.5896 | 0.0087 |
| 7 | 0.5657 | 0.0165 | 0.5889 | 0.0126 |
| 8 | 0.5514 | 0.0176 | 0.5531 | 0.0176 |
| 9 | 0.5780 | 0.0057 | 0.6032 | 0.0173 |
| 10 | 0.5714 | 0.0132 | 0.5928 | 0.0178 |
| **11** | **0.5840** | **0.0076** | **0.6081** | **0.0035** |
| 12 | 0.5801 | 0.0074 | 0.6110 | 0.0070 |
| 13 | 0.5795 | 0.0103 | 0.6044 | 0.0133 |
| 14 | 0.5828 | 0.0179 | 0.6084 | 0.0199 |
| 15 | 0.5729 | 0.0180 | 0.5986 | 0.0208 |
| 16 | 0.5654 | 0.0203 | 0.5936 | 0.0165 |

**Best layer: 11** — 58.4% accuracy, 0.608 AUC

## Results: Llama 3.2 1B Instruct

| Layer | Accuracy | Std | AUC | Std |
|-------|----------|-----|-----|-----|
| 0 (embed) | 0.5642 | 0.0204 | 0.4998 | 0.0027 |
| 1 | 0.5861 | 0.0133 | 0.6036 | 0.0247 |
| 2 | 0.5648 | 0.0068 | 0.5795 | 0.0071 |
| 3 | 0.5744 | 0.0235 | 0.5937 | 0.0236 |
| 4 | 0.5657 | 0.0150 | 0.5769 | 0.0148 |
| 5 | 0.5529 | 0.0111 | 0.5698 | 0.0196 |
| 6 | 0.5702 | 0.0178 | 0.5924 | 0.0176 |
| 7 | 0.5654 | 0.0210 | 0.5734 | 0.0214 |
| 8 | 0.5594 | 0.0154 | 0.5749 | 0.0225 |
| 9 | 0.5723 | 0.0111 | 0.5866 | 0.0082 |
| 10 | 0.5744 | 0.0108 | 0.6012 | 0.0125 |
| 11 | 0.5789 | 0.0093 | 0.5968 | 0.0168 |
| 12 | 0.5714 | 0.0108 | 0.5876 | 0.0162 |
| 13 | 0.5738 | 0.0096 | 0.5957 | 0.0132 |
| 14 | 0.5816 | 0.0109 | 0.5989 | 0.0048 |
| 15 | 0.5789 | 0.0106 | 0.5959 | 0.0040 |
| **16** | **0.5927** | **0.0229** | **0.6191** | **0.0231** |

**Best layer: 16 (final)** — 59.3% accuracy, 0.619 AUC

## Results: Llama 3.2 1B Instruct-safety

(Instruct + LoRA safety fine-tune on AdvBench/HarmBench refusals.)

| Layer | Accuracy | Std | AUC | Std |
|-------|----------|-----|-----|-----|
| 0 (embed) | 0.5642 | 0.0204 | 0.4998 | 0.0027 |
| 1 | 0.5777 | 0.0132 | 0.5992 | 0.0225 |
| 2 | 0.5657 | 0.0061 | 0.5783 | 0.0127 |
| 3 | 0.5765 | 0.0112 | 0.5964 | 0.0109 |
| 4 | 0.5579 | 0.0159 | 0.5734 | 0.0167 |
| 5 | 0.5478 | 0.0116 | 0.5667 | 0.0109 |
| 6 | 0.5514 | 0.0109 | 0.5676 | 0.0098 |
| 7 | 0.5529 | 0.0105 | 0.5668 | 0.0122 |
| 8 | 0.5663 | 0.0109 | 0.5794 | 0.0125 |
| 9 | 0.5774 | 0.0194 | 0.5942 | 0.0272 |
| 10 | 0.5636 | 0.0204 | 0.5914 | 0.0159 |
| 11 | 0.5639 | 0.0050 | 0.5883 | 0.0074 |
| 12 | 0.5756 | 0.0120 | 0.5995 | 0.0054 |
| 13 | 0.5762 | 0.0094 | 0.6039 | 0.0142 |
| 14 | 0.5819 | 0.0221 | 0.6096 | 0.0186 |
| **15** | **0.5867** | 0.0153 | **0.6176** | 0.0184 |
| 16 (final) | 0.5849 | 0.0372 | 0.6076 | 0.0298 |

**Best layer: 15 (near-final)** — 58.7% accuracy, 0.618 AUC

## Key Observations

1. **All three models exceed majority-class baseline**, indicating a real but modest linear signal for humor explanation quality in the hidden representations.

2. **Peak layer moves toward the output with alignment: base 11 (middle) → instruct 16 (final) → safety 15 (near-final).** RLHF pushes humor-relevant features into the output layers; the safety fine-tune **keeps them late** (layer 15, essentially tied with 16/14), it does **not** revert the peak back to the middle. Alignment's late-layer reshaping persists through safety FT.

3. **Safety FT does not flatten the humor signal.** Peak AUC is essentially unchanged across the aligned models — instruct 0.619, safety 0.618 — and both exceed base (0.608). So the safety fine-tune that *collapses joke generation* (74% refusals) and *widens reading-time entropy* leaves the model's linear *humor-understanding* signal intact. The tax is on production and distribution, not on this probe-measurable comprehension feature.

4. **The signal is weak overall (~0.61–0.62 AUC) for all three.** This aligns with the Chumor paper's finding that LLMs perform near-random on humor explanation tasks. The models have some representation of humor understanding, but it is not strongly linearly separable.

5. **Layer 0 (embedding) AUC is ~0.50 for all three models** — pure token embeddings carry no humor signal, as expected.

6. **Layers 4–7 are a consistent dip across all three models**, suggesting a processing bottleneck or transition point in the network's internal representations (the earlier base/instruct "layer 8 dip" generalizes to a low-AUC band in the lower-middle layers, and is deepest in the safety model).
