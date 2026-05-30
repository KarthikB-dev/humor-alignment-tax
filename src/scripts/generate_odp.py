"""Generate an ODP presentation using odfpy (system Python).

Run with:  python3 src/scripts/generate_odp.py
"""

import os
from pathlib import Path

from odf.opendocument import OpenDocumentPresentation
from odf.style import Style, MasterPage, PageLayout, PageLayoutProperties, TextProperties
from odf.text import P, Span
from odf.draw import Page, Frame, TextBox, Image

OUT = Path("presentation")
FIG = Path("presentation/figures")


class ODP:
    W, H = "33.87cm", "19.05cm"

    def __init__(self):
        self.doc = OpenDocumentPresentation()
        self._styles()

    def _styles(self):
        d = self.doc

        pl = PageLayout(name="PL")
        d.automaticstyles.addElement(pl)
        pl.addElement(PageLayoutProperties(
            margin="0cm", pagewidth=self.W, pageheight=self.H,
            printorientation="landscape",
        ))

        self.mp = MasterPage(name="MP", pagelayoutname="PL")
        d.masterstyles.addElement(self.mp)

        for name, size, weight, color in [
            ("Heading", "32pt", "bold",   "#1565C0"),
            ("Body",    "20pt", "normal", "#1a1a2e"),
            ("Small",   "14pt", "normal", "#333333"),
            ("White",   "32pt", "bold",   "#FFFFFF"),
        ]:
            s = Style(name=name, family="text")
            s.addElement(TextProperties(
                fontsize=size, fontweight=weight, color=color,
                fontfamily="Liberation Sans",
            ))
            d.automaticstyles.addElement(s)

    # ── low-level helpers ─────────────────────────────────────────

    def _slide(self):
        p = Page(masterpagename="MP")
        self.doc.presentation.addElement(p)
        return p

    def _textbox(self, slide, x, y, w, h):
        f = Frame(width=f"{w}cm", height=f"{h}cm", x=f"{x}cm", y=f"{y}cm")
        slide.addElement(f)
        tb = TextBox()
        f.addElement(tb)
        return tb

    def _para(self, tb, text, style="Body"):
        p = P()
        tb.addElement(p)
        sp = Span(stylename=style)
        sp.addText(str(text))
        p.addElement(sp)

    def _image(self, slide, path, x, y, w, h):
        href = self.doc.addPicture(str(path))
        f = Frame(width=f"{w}cm", height=f"{h}cm", x=f"{x}cm", y=f"{y}cm")
        slide.addElement(f)
        img = Image(href=href, type="simple", show="embed", actuate="onLoad")
        f.addElement(img)

    # ── slide builders ────────────────────────────────────────────

    def title_slide(self, title, subtitle, author):
        s = self._slide()
        tb = self._textbox(s, 2, 4, 30, 4)
        self._para(tb, title, "Heading")
        tb2 = self._textbox(s, 2, 9, 30, 2)
        self._para(tb2, subtitle, "Body")
        tb3 = self._textbox(s, 2, 12, 30, 1.5)
        self._para(tb3, author, "Small")

    def text_slide(self, title, lines, style="Body"):
        s = self._slide()
        tb = self._textbox(s, 1, 0.4, 32, 2)
        self._para(tb, title, "Heading")
        tb2 = self._textbox(s, 1.5, 3.2, 31, 15.5)
        for line in lines:
            self._para(tb2, line, style)

    def two_col_slide(self, title, left, right, style="Body"):
        s = self._slide()
        tb = self._textbox(s, 1, 0.4, 32, 2)
        self._para(tb, title, "Heading")
        tbl = self._textbox(s, 1, 3.2, 15.5, 15.5)
        for line in left:
            self._para(tbl, line, style)
        tbr = self._textbox(s, 17.5, 3.2, 15.5, 15.5)
        for line in right:
            self._para(tbr, line, style)

    def image_slide(self, title, img_path, caption=None):
        s = self._slide()
        tb = self._textbox(s, 1, 0.4, 32, 2)
        self._para(tb, title, "Heading")
        p = Path(img_path)
        if p.exists():
            self._image(s, p, 1.5, 3, 31, 15.3)
        else:
            tb2 = self._textbox(s, 1.5, 8, 31, 3)
            self._para(tb2, f"[{p.name}]", "Small")
        if caption:
            tb3 = self._textbox(s, 1.5, 18.2, 31, 1)
            self._para(tb3, caption, "Small")

    def save(self, path):
        self.doc.save(str(path))
        print(f"Saved: {path}")


# ── Content ───────────────────────────────────────────────────

def build(odp: ODP):
    odp.title_slide(
        "Do Safety Constraints Tax Humor?",
        "Quantifying the Alignment Tax via Distributional Analysis",
        "Karthik Bhattaram  ·  2026",
    )

    odp.text_slide("The Core Question", [
        "Hypothesis: RLHF safety training compresses LLM outputs toward",
        "safe, predictable text — reducing the statistical properties",
        "that make jokes funny.",
        "",
        "Three metrics expected to fall under alignment:",
        "  • Punchline surprisal    — unexpected endings",
        "  • Pre-punchline entropy  — model uncertainty before punchline",
        "  • Embedding distance     — semantic leap from setup to punchline",
        "",
        "Three models tested:",
        "  Base  →  Instruct (RLHF)  →  Safety-finetuned (ours)",
    ])

    odp.two_col_slide("Datasets", [
        "Joke corpora:",
        "  r/Jokes",
        "    194,553 Reddit jokes",
        "  One-liners",
        "    3,200 short jokes",
        "  1,000 per source used",
        "  for metric extraction",
    ], [
        "Safety fine-tuning data:",
        "  AdvBench",
        "    520 harmful prompts",
        "  HarmBench",
        "    400 harmful prompts",
        "  920 total (prompt, refusal)",
        "  7 varied refusal templates",
    ])

    odp.text_slide("Models", [
        "Base     Llama 3.2-1B              Pre-training only",
        "         1.2B params",
        "",
        "Instruct Llama 3.2-1B-Instruct     + RLHF / instruction tuning",
        "         1.2B params",
        "",
        "Safety   Llama 3.2-1B-Instruct     + LoRA refusal fine-tune",
        "  (ours) 1.2B params               AdvBench + HarmBench",
        "",
        "Safety fine-tuning: LoRA r=16, 3 epochs, lr=2e-4,",
        "response-only loss. Training loss: 3.72 → 0.34,  accuracy: ~90%",
    ])

    odp.text_slide("Three Distributional Metrics", [
        "1. Punchline Surprisal",
        "   Mean −log P(punchline tokens given setup).",
        "   High = model didn't predict the punchline.",
        "",
        "2. Pre-punchline Entropy",
        "   Shannon entropy of next-token distribution at setup end.",
        "   High = model uncertain about what comes next.",
        "",
        "3. Setup–Punchline Embedding Distance",
        "   Cosine distance between sentence embeddings.",
        "   High = punchline is conceptually far from setup (semantic leap).",
        "",
        "Inverted-U hypothesis: humor quality peaks at MODERATE values of all three.",
    ])

    odp.image_slide("Safety Fine-tuning: Training Curve",
                    FIG / "training_loss.png",
                    "Loss converges 3.72 → 0.34 over 3 epochs (165 steps). Token accuracy ~90%.")

    odp.image_slide("Results: Metric Means", FIG / "metric_means.png")
    odp.image_slide("Results: Full Distributions", FIG / "distributions.png")
    odp.image_slide("Results: Metric Progression", FIG / "shifts.png")

    odp.text_slide("Results: Statistical Comparison", [
        "Base → Instruct:",
        "  Surprisal  +0.339   Cohen d=0.18   p = 2e-15  ***",
        "  Entropy    +0.210   Cohen d=0.08   p = 0.071",
        "  Distance   +0.035   Cohen d=0.41   p = 2e-27  ***",
        "",
        "Instruct → Safety:",
        "  Surprisal  +0.092   Cohen d=0.05   p = 0.016  *",
        "  Entropy    +0.439   Cohen d=0.16   p = 1e-7   ***",
        "  Distance   +0.008   Cohen d=0.07   p = 0.56   n.s.",
        "",
        "Base → Safety (cumulative):",
        "  Surprisal  +0.431   Cohen d=0.23   p = 3e-24  ***",
        "  Entropy    +0.648   Cohen d=0.24   p = 5e-14  ***",
        "  Distance   +0.043   Cohen d=0.45   p = 4e-25  ***",
    ], style="Small")

    odp.image_slide("Temperature Sweep", FIG / "temperature_sweep.png",
                    "Higher temperature raises both metrics. Instruct stays above Base at all temps.")

    # ── LLM-as-judge humor findings ───────────────────────────────
    odp.text_slide("Humor Quality: LLM-as-Judge", [
        "Distributional metrics measure form — but do the jokes actually land?",
        "Scored every joke (corpus + generated) with a single consistent judge",
        "(Gemini), 1-10 on unexpectedness, cleverness, amusement, overall.",
        "",
        "Why not the local 1B judge?",
        "  It returned valid scores for only ~18% of generated jokes",
        "  (parse failures concentrate on the noisiest outputs).",
        "  Gemini judge: 2,851 / 2,859 = 99.7% valid.",
        "",
        "One judge over corpus AND generated removes the two-judge confound.",
    ])

    odp.image_slide("Humor by Alignment Tier", FIG / "humor_by_tier.png",
                    "Non-monotonic: RLHF (Instruct) matches the human corpus; safety fine-tuning "
                    "collapses humor — ~74% of its outputs are refusals to benign joke prompts.")

    odp.image_slide("Humor vs. Temperature (Hypothesis 4)", FIG / "humor_temperature.png",
                    "No moderate-temperature peak: instruct humor declines monotonically (7.2 -> 5.8) "
                    "as surprisal/entropy rise. Funniest at low temperature / narrow nucleus.")

    odp.image_slide("Inverted-U on Generated Jokes? No.", FIG / "generated_inverted_u.png",
                    "Humor vs. metrics (instruct), with quadratic fits. No significant peak; metrics "
                    "relate to humor (R^2 up to 0.32) but monotonically, not as a sweet spot.")

    odp.text_slide("Key Finding: Disruption, Not Compression", [
        "All three metrics INCREASE at every alignment step.",
        "Safety training does NOT compress outputs toward predictable text.",
        "",
        "Two distinct mechanisms:",
        "",
        "  Base → Instruct (driven by surprisal & distance):",
        "    RLHF sensitises the model to harmful content.",
        "    Offensive punchlines become more surprising.",
        "    Clean classic jokes are better predicted (RLHF familiarity effect).",
        "",
        "  Instruct → Safety (driven by entropy):",
        "    Refusal fine-tuning disrupts next-token confidence across all text.",
        "    Collateral damage from a narrow 920-example training distribution.",
        "    Distance barely moves — semantic leap unchanged.",
    ])

    odp.two_col_slide("Example Jokes: Largest Divergences", [
        "Instruct >> Base surprisal:",
        "(offensive jokes)",
        "",
        '"What do you call a guy',
        ' who gets lots of blowjobs?"',
        "→ Successful",
        "Base: 14.2   Inst: 18.0   Δ=+3.84",
        "",
        '"What can you make with',
        ' epileptic lettuce?"',
        "→ A seizure salad",
        "Base: 7.2    Inst: 10.7   Δ=+3.58",
    ], [
        "Base >> Instruct surprisal:",
        "(classic clean jokes)",
        "",
        '"Why did the hungry baby',
        ' calf cross the road?"',
        "→ To get to the udder side.",
        "Base: 7.1    Inst: 5.7   Δ=−1.43",
        "",
        '"For Sale: French WWII Rifle"',
        "→ Never fired. Only dropped once.",
        "Base: 7.7    Inst: 6.5   Δ=−1.19",
    ], style="Small")

    odp.two_col_slide("Conclusion & Future Work", [
        "Findings:",
        "  Inverted-U not confirmed (1B, noisy)",
        "  Alignment shift confirmed —",
        "    direction is UPWARD",
        "  Two distinct mechanisms:",
        "    RLHF content-sensitisation",
        "    Safety entropy disruption",
        "  Humor (LLM-judge): RLHF keeps",
        "    humor; safety FT -> 74% refusals",
        "",
        "Limitations:",
        "  1B model is weak",
        "  Offensive jokes confound results",
        "  Only 920 refusal examples",
    ], [
        "Future work:",
        "  Filter to clean-only jokes",
        "  and re-test compression hypothesis",
        "",
        "  Scale to Llama 3.1-8B",
        "",
        "  Probe the safety model's",
        "  humor representations (Chumor)",
        "",
        "  Probe whether elevated entropy",
        "  improves joke generation quality",
    ])


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    odp = ODP()
    build(odp)
    odp.save(OUT / "alignment_tax.odp")
