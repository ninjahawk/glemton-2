"""Verified, streamable dataset sources for ton-2 (all openly licensed, no HF
token required, each exposes a ``text`` field). Shared by train_tokenizer.py and
prepare_data.py. Verified loadable 2026-06-28.
"""
from __future__ import annotations

SOURCES = {
    "fineweb_edu": dict(repo="HuggingFaceFW/fineweb-edu", config="sample-10BT", split="train", field="text"),
    "cosmopedia":  dict(repo="HuggingFaceTB/smollm-corpus", config="cosmopedia-v2", split="train", field="text"),
    "tinystories": dict(repo="roneneldan/TinyStories", config=None, split="train", field="text"),
    "openwebmath": dict(repo="open-web-math/open-web-math", config=None, split="train", field="text"),
}


def stream_texts(key: str):
    """Yield non-empty text documents from a source, streaming (no full download)."""
    from datasets import load_dataset

    s = SOURCES[key]
    if s["config"]:
        ds = load_dataset(s["repo"], s["config"], split=s["split"], streaming=True)
    else:
        ds = load_dataset(s["repo"], split=s["split"], streaming=True)
    field = s["field"]
    for ex in ds:
        t = ex.get(field)
        if isinstance(t, str) and t.strip():
            yield t
