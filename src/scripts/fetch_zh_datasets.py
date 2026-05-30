"""Fetch the two proposal datasets that were missing locally:

  * Ruozhiba  -> data/raw/ruozhiba.json   (LooksJuicy/ruozhiba; instruction/output Q&A)
  * Oogiri    -> data/raw/oogiri.json     (zhongshsh/CLoT-Oogiri-GO; T2T odai->boke pairs)

Both are streamed so we never pull the Oogiri image blobs. Output schemas match
the existing loaders (load_ruozhiba expects question/answer; load_oogiri expects
prompt/response).
"""

import argparse
import json
import os
from pathlib import Path

# These run before importing `datasets`; make sure we are ONLINE here
# (src.utils sets HF_HUB_OFFLINE=1, but we don't import it).
os.environ.pop("HF_HUB_OFFLINE", None)

RAW_DIR = Path("data/raw")


def fetch_ruozhiba(limit: int | None) -> None:
    from datasets import load_dataset

    print("Streaming LooksJuicy/ruozhiba ...", flush=True)
    ds = load_dataset("LooksJuicy/ruozhiba", split="train", streaming=True)
    out = []
    for row in ds:
        q = (row.get("instruction") or "").strip()
        a = (row.get("output") or "").strip()
        if not q or not a:
            continue
        out.append({"question": q, "answer": a})
        if limit and len(out) >= limit:
            break
    path = RAW_DIR / "ruozhiba.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"  Wrote {len(out)} Q&A pairs -> {path}", flush=True)


def fetch_oogiri(limit: int | None, max_scan: int) -> None:
    from datasets import load_dataset

    print("Streaming zhongshsh/CLoT-Oogiri-GO (T2T rows only) ...", flush=True)
    # cast image column away so streaming never decodes the blobs
    ds = load_dataset("zhongshsh/CLoT-Oogiri-GO", split="train", streaming=True)
    ds = ds.remove_columns([c for c in ["image"] if c in ds.column_names])
    out = []
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > max_scan:
            break
        if row.get("type") != "T2T":
            continue
        setup = (row.get("question") or "").strip()
        punch = (row.get("text") or "").strip()
        if not setup or not punch:
            continue
        out.append({"prompt": setup, "response": punch})
        if limit and len(out) >= limit:
            break
    path = RAW_DIR / "oogiri.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"  Wrote {len(out)} odai->boke pairs (scanned {scanned}) -> {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["ruozhiba", "oogiri", "both"], default="both")
    ap.add_argument("--limit", type=int, default=1000, help="Max pairs per dataset")
    ap.add_argument("--max_scan", type=int, default=200000,
                    help="Max Oogiri rows to scan for T2T pairs")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.which in ("ruozhiba", "both"):
        fetch_ruozhiba(args.limit)
    if args.which in ("oogiri", "both"):
        fetch_oogiri(args.limit, args.max_scan)


if __name__ == "__main__":
    main()
