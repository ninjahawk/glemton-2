"""G3: stream sources, tokenize with the 24k BPE, pack into uint16 .bin shards.

<|eot|> is appended at each document boundary. Output: data/tokenized/<source>/
shard_NNNNN.bin. Run BEFORE training, never during (host-RAM + thermal headroom).
Streaming is not resumable mid-source; a restart re-streams from the top.

Run (full):  .venv\\Scripts\\python.exe scripts\\prepare_data.py
Run (smoke): ... scripts\\prepare_data.py --sources tinystories --budget-scale 0.01
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from ton2.data import write_shard  # noqa: E402
from ton2.tokenizer import load as load_tok  # noqa: E402
from data_sources import SOURCES, stream_texts  # noqa: E402

SHARD_TOKENS = 100_000_000  # ~200 MB per shard (uint16)
ENCODE_BATCH = 1000         # docs per encode_batch call (Rust, multithreaded)

# Night-1 per-source token budgets.
DEFAULT_BUDGET = {
    "fineweb_edu": 5_000_000_000,
    "cosmopedia": 2_000_000_000,
    "tinystories": 500_000_000,
    "openwebmath": 700_000_000,
}


def pack_source(key, budget, tok, eot_id, out_root):
    out_dir = os.path.join(out_root, key)
    os.makedirs(out_dir, exist_ok=True)
    buf: list[int] = []
    written = 0
    shard_idx = 0
    total = 0
    t0 = time.time()
    docs: list[str] = []

    def emit(encs):
        nonlocal total
        for e in encs:
            buf.extend(e.ids)
            buf.append(eot_id)
            total += len(e.ids) + 1

    def maybe_flush():
        nonlocal shard_idx, written
        while len(buf) >= SHARD_TOKENS:
            arr = np.array(buf[:SHARD_TOKENS], dtype=np.uint16)
            write_shard(arr, os.path.join(out_dir, f"shard_{shard_idx:05d}.bin"))
            del buf[:SHARD_TOKENS]
            shard_idx += 1
            written += SHARD_TOKENS
            print(f"[data] {key}: {written/1e6:.0f}M tok ({time.time()-t0:.0f}s)", flush=True)

    try:
        for text in stream_texts(key):
            docs.append(text)
            if len(docs) >= ENCODE_BATCH:
                emit(tok.encode_batch(docs))
                docs = []
                maybe_flush()
                if total >= budget:
                    break
    except Exception as e:
        print(f"[data] {key} stream error: {e!r}", flush=True)

    if docs:
        emit(tok.encode_batch(docs))
        maybe_flush()
    if buf:
        write_shard(np.array(buf, dtype=np.uint16), os.path.join(out_dir, f"shard_{shard_idx:05d}.bin"))
        written += len(buf)
    print(f"[data] {key}: DONE {written/1e6:.1f}M tokens", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="fineweb_edu,cosmopedia,tinystories,openwebmath")
    ap.add_argument("--out", default="data/tokenized")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="scale all budgets (e.g. 0.01 to smoke-test)")
    args = ap.parse_args()

    tok = load_tok()
    eot_id = tok.token_to_id("<|eot|>")
    assert eot_id is not None, "tokenizer missing <|eot|>; train it first"

    grand = 0
    for key in [k.strip() for k in args.sources.split(",") if k.strip()]:
        if key not in SOURCES:
            print("unknown source", key, flush=True)
            continue
        budget = int(DEFAULT_BUDGET.get(key, 1_000_000_000) * args.budget_scale)
        grand += pack_source(key, budget, tok, eot_id, args.out)
    print(f"[data] GRAND TOTAL {grand/1e9:.2f}B tokens", flush=True)


if __name__ == "__main__":
    main()
