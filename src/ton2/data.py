"""Data loading for ton-2.

Corpora are tokenized into contiguous ``uint16`` ``.bin`` shards (flat token
streams). The trainer mmaps them and draws random ``seq_len + 1`` windows to form
(input, target) pairs. Shards live under ``data/tokenized/<source>/*.bin``; the
parent directory name is the source key used for ``source_weights``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch


def write_shard(token_ids: np.ndarray, out_path: str) -> None:
    assert token_ids.dtype == np.uint16, "vocab must fit uint16 (<= 65535)"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    token_ids.tofile(out_path)


def open_shard(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


class PackedDataset(torch.utils.data.IterableDataset):
    """Yields (input, target) windows of shape (seq_len,) from packed shards.

    Random shard (weighted by token count × optional per-source multiplier),
    then random offset. No epoch boundary.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int,
        seed: int = 0,
        source_weights: Optional[dict[str, float]] = None,
    ):
        super().__init__()
        self.shard_paths = sorted(str(p) for p in Path(shard_dir).rglob("*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .bin shards found in {shard_dir}")
        self.seq_len = seq_len
        self.seed = seed
        self.source_weights = source_weights or {}

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (worker.id if worker else 0))
        shards = [open_shard(p) for p in self.shard_paths]
        sizes = np.array([len(s) for s in shards], dtype=np.int64)
        mults = np.ones(len(shards), dtype=np.float64)
        for i, p in enumerate(self.shard_paths):
            src = Path(p).parent.name
            if src in self.source_weights:
                mults[i] = float(self.source_weights[src])
        weights = (sizes * mults).astype(np.float64)
        weights /= weights.sum()
        while True:
            i = int(rng.choice(len(shards), p=weights))
            arr = shards[i]
            max_offset = len(arr) - self.seq_len - 1
            if max_offset <= 0:
                continue
            off = int(rng.integers(0, max_offset))
            chunk = arr[off : off + self.seq_len + 1].astype(np.int64)
            yield torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def make_dataloader(
    shard_dir: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 2,
    source_weights: Optional[dict[str, float]] = None,
):
    ds = PackedDataset(shard_dir, seq_len=seq_len, source_weights=source_weights)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
