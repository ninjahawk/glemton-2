"""G2: train the 24k byte-level BPE on a balanced sample across sources, then gate.

Run: .venv\\Scripts\\python.exe scripts\\train_tokenizer.py
Gate: plain text round-trips exactly, all special tokens present, vocab <= 65535.
"""
from __future__ import annotations

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

from ton2.tokenizer import SPECIAL_TOKENS, train_bpe  # noqa: E402
from data_sources import SOURCES, stream_texts  # noqa: E402

# BPE doesn't need much data; cap docs per source for a fast, balanced sample.
PER_SOURCE_DOCS = {"fineweb_edu": 150_000, "cosmopedia": 80_000, "tinystories": 80_000, "openwebmath": 50_000}


def sample_iter():
    for key in SOURCES:
        n = PER_SOURCE_DOCS.get(key, 50_000)
        cnt = 0
        try:
            for t in stream_texts(key):
                yield t
                cnt += 1
                if cnt >= n:
                    break
        except Exception as e:  # a single source failing must not kill the tokenizer
            print(f"[tok] source {key} failed: {e!r}; skipping", flush=True)
        print(f"[tok] sampled {cnt} docs from {key}", flush=True)


def main() -> int:
    tok = train_bpe(sample_iter(), vocab_size=24000)
    vs = tok.get_vocab_size()
    print("vocab_size", vs)
    assert vs <= 65535, "vocab exceeds uint16 packing limit"

    ok = True
    for s in [
        "Hello, world! 2 + 2 = 4.",
        "The mitochondria is the powerhouse of the cell.",
        'She said, "why not?" and left.',
    ]:
        dec = tok.decode(tok.encode(s).ids)
        print(repr(s), "->", repr(dec))
        if dec != s:
            ok = False

    for st in SPECIAL_TOKENS:
        assert tok.token_to_id(st) is not None, f"missing special token {st}"
    print("special_inline_ids", tok.encode("a<|eot|>b").ids)

    print("GATE_PASS" if ok else "GATE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
