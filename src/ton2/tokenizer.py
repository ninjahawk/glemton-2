"""Byte-level BPE tokenizer for ton-2 (HuggingFace ``tokenizers``).

Byte-level so it never produces ``<unk>`` and round-trips any input. Special
tokens (chat turns, calculator tool, BOS/EOT, spares) are reserved up front so
they exist in the vocab from pretraining on — no "added later and untrained"
problem. Vocab 24k keeps the tied embedding from eating too much of a ~170M model
and stays under the uint16 (65535) packing limit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Order matters: these get the lowest ids. Keep stable across the project.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eot|>",        # end of turn / document
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<calc>",
    "</calc>",
    "<|reserved_0|>",
    "<|reserved_1|>",
    "<|reserved_2|>",
    "<|reserved_3|>",
]

DEFAULT_PATH = "data/tokenizer/ton2-bpe-24k.json"


def train_bpe(
    text_iter: Iterable[str],
    vocab_size: int = 24000,
    out_path: str = DEFAULT_PATH,
    min_frequency: int = 2,
) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iter, trainer=trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)
    return tok


def load(path: str = DEFAULT_PATH) -> Tokenizer:
    return Tokenizer.from_file(path)
