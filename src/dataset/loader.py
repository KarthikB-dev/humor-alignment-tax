import csv
import json
import re
import uuid
from pathlib import Path

from src.dataset.joke import ChumorEntry, DatasetSource, JokeEntry, JokeLanguage

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def _make_id(source: DatasetSource) -> str:
    return f"{source.value}_{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------------ #
#  r/Jokes  (JSON lines: {"title": ..., "body": ...})
# ------------------------------------------------------------------ #
def load_r_jokes(path: str | None = None, limit: int | None = None) -> list[JokeEntry]:
    path = path or f"{DATA_DIR}/r_jokes.json"
    entries: list[JokeEntry] = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            setup = obj.get("title", "").strip()
            punchline = obj.get("body", "").strip()
            if not setup or not punchline:
                continue
            entries.append(
                JokeEntry(
                    text=f"{setup}\n{punchline}",
                    setup=setup,
                    punchline=punchline,
                    source=DatasetSource.R_JOKES,
                    language=JokeLanguage.ENGLISH,
                    joke_id=_make_id(DatasetSource.R_JOKES),
                )
            )
            if limit and len(entries) >= limit:
                break
    return entries


# ------------------------------------------------------------------ #
#  16k One-Liners  (one joke per line)
# ------------------------------------------------------------------ #
def _split_one_liner(text: str) -> tuple[str, str]:
    """Heuristic split: last clause after a dash, colon, or ellipsis is punchline."""
    for sep in [" — ", " - ", " – ", "... ", ": "]:
        if sep in text:
            parts = text.rsplit(sep, 1)
            if len(parts) == 2 and len(parts[1]) > 5:
                return parts[0] + sep.rstrip(), parts[1]

    # Fallback: split at roughly the last third
    words = text.split()
    split_idx = max(1, len(words) * 2 // 3)
    return " ".join(words[:split_idx]), " ".join(words[split_idx:])


def load_one_liners(
    path: str | None = None, limit: int | None = None
) -> list[JokeEntry]:
    path = path or f"{DATA_DIR}/one_liners.txt"
    entries: list[JokeEntry] = []
    with open(path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            setup, punchline = _split_one_liner(text)
            entries.append(
                JokeEntry(
                    text=text,
                    setup=setup,
                    punchline=punchline,
                    source=DatasetSource.ONE_LINER,
                    language=JokeLanguage.ENGLISH,
                    joke_id=_make_id(DatasetSource.ONE_LINER),
                )
            )
            if limit and len(entries) >= limit:
                break
    return entries


# ------------------------------------------------------------------ #
#  Oogiri  (JSON with prompt/response structure)
# ------------------------------------------------------------------ #
def load_oogiri(path: str | None = None, limit: int | None = None) -> list[JokeEntry]:
    path = path or f"{DATA_DIR}/oogiri.json"
    entries: list[JokeEntry] = []
    with open(path) as f:
        data = json.load(f)
    for item in data:
        setup = item.get("prompt", item.get("boke_text", "")).strip()
        punchline = item.get("response", item.get("tsukkomi_text", "")).strip()
        if not setup or not punchline:
            continue

        # Detect language
        lang = (
            JokeLanguage.CHINESE
            if re.search(r"[\u4e00-\u9fff]", setup)
            else JokeLanguage.ENGLISH
        )
        entries.append(
            JokeEntry(
                text=f"{setup}\n{punchline}",
                setup=setup,
                punchline=punchline,
                source=DatasetSource.OOGIRI,
                language=lang,
                joke_id=_make_id(DatasetSource.OOGIRI),
            )
        )
        if limit and len(entries) >= limit:
            break
    return entries


# ------------------------------------------------------------------ #
#  Ruozhiba  (JSON with question/answer structure, Chinese)
# ------------------------------------------------------------------ #
def load_ruozhiba(
    path: str | None = None, limit: int | None = None
) -> list[JokeEntry]:
    path = path or f"{DATA_DIR}/ruozhiba.json"
    entries: list[JokeEntry] = []
    with open(path) as f:
        data = json.load(f)
    for item in data:
        setup = item.get("question", item.get("title", "")).strip()
        punchline = item.get("answer", item.get("content", "")).strip()
        if not setup or not punchline:
            continue
        entries.append(
            JokeEntry(
                text=f"{setup}\n{punchline}",
                setup=setup,
                punchline=punchline,
                source=DatasetSource.RUOZHIBA,
                language=JokeLanguage.CHINESE,
                joke_id=_make_id(DatasetSource.RUOZHIBA),
            )
        )
        if limit and len(entries) >= limit:
            break
    return entries


# ------------------------------------------------------------------ #
#  Ruozhiba-raw  (kirp/ruozhiba-raw, 92k raw forum posts, HuggingFace)
# ------------------------------------------------------------------ #
def load_ruozhiba_raw(limit: int | None = None) -> list[JokeEntry]:
    from datasets import load_dataset  # lazy import — only needed for HF sources

    ds = load_dataset("kirp/ruozhiba-raw", split="train")
    entries: list[JokeEntry] = []

    for row in ds:
        # Schema: title (question/setup), detail (body, often None for short posts)
        setup = (row.get("title") or "").strip()
        punchline = (row.get("detail") or "").strip()

        if not setup:
            continue
        # Some posts have no separate body — treat full title as a one-liner
        if not punchline:
            setup, punchline = _split_one_liner(setup)

        entries.append(
            JokeEntry(
                text=f"{setup}\n{punchline}",
                setup=setup,
                punchline=punchline,
                source=DatasetSource.RUOZHIBA_RAW,
                language=JokeLanguage.CHINESE,
                joke_id=_make_id(DatasetSource.RUOZHIBA_RAW),
            )
        )
        if limit and len(entries) >= limit:
            break

    return entries


# ------------------------------------------------------------------ #
#  Chumor  (MichiganNLP/Chumor — joke + explanation + correctness label)
# ------------------------------------------------------------------ #
def load_chumor(
    split: str = "test", limit: int | None = None
) -> list[ChumorEntry]:
    """Load the Chumor humor-explanation dataset.

    Each entry is a (joke, explanation, label) triple where label indicates
    whether the explanation correctly accounts for why the joke is funny.
    Useful as probe labels: train a linear classifier on hidden states to
    predict explanation_correct.

    Args:
        split: HuggingFace dataset split — "test" by default (the labeled split).
        limit: cap the number of entries returned.
    """
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    try:
        ds = load_dataset("MichiganNLP/Chumor", split=split)
    except DatasetNotFoundError:
        raise RuntimeError(
            "Chumor is a gated dataset. Request access at "
            "https://huggingface.co/datasets/MichiganNLP/Chumor, "
            "then run: huggingface-cli login"
        ) from None
    entries: list[ChumorEntry] = []

    for i, row in enumerate(ds):
        joke = (row.get("Joke") or "").strip()
        explanation = (row.get("Explanation") or "").strip()
        raw_label = row.get("Label", "")

        if not joke or not explanation:
            continue

        correct = str(raw_label).lower() in {"good", "correct", "1", "true"}
        source = str(row.get("Source", "")).strip()

        entries.append(
            ChumorEntry(
                joke_id=f"chumor_{i:06d}",
                joke=joke,
                explanation=explanation,
                explanation_correct=correct,
                explanation_source=source,
                language=JokeLanguage.CHINESE,
            )
        )
        if limit and len(entries) >= limit:
            break

    return entries


# ------------------------------------------------------------------ #
#  Unified loader
# ------------------------------------------------------------------ #
LOADERS = {
    DatasetSource.R_JOKES: load_r_jokes,
    DatasetSource.ONE_LINER: load_one_liners,
    DatasetSource.OOGIRI: load_oogiri,
    DatasetSource.RUOZHIBA: load_ruozhiba,
    DatasetSource.RUOZHIBA_RAW: load_ruozhiba_raw,
    # CHUMOR omitted: returns ChumorEntry, not JokeEntry — use load_chumor() directly
}


def load_all_jokes(
    sources: list[DatasetSource] | None = None,
    limit_per_source: int | None = None,
) -> list[JokeEntry]:
    sources = sources or list(DatasetSource)
    all_jokes: list[JokeEntry] = []
    for source in sources:
        loader = LOADERS[source]
        try:
            jokes = loader(limit=limit_per_source)
            all_jokes.extend(jokes)
            print(f"Loaded {len(jokes)} jokes from {source.value}")
        except FileNotFoundError:
            print(f"Warning: data file not found for {source.value}, skipping.")
    return all_jokes
