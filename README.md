# Glemton-2

A small (~180M-param) language model built **from scratch** on a single consumer
GPU (RTX 5070, 12 GB) — and taken through the full modern chat pipeline
(tokenizer → pretraining → SFT → DPO → tool-use → chat UI) so you can actually
talk to it. Same family as Glemton-1, opposite philosophy.

**The twist:** a distinctive, *trained-in, measurable* personality — **witty &
deadpan, grounded in honesty.** Not a system prompt bolted on at inference;
baked into the weights, and evaluated to prove it stuck.

> **Status: planning.** No code or weights yet.
> - Full scope → [`PLAN.md`](PLAN.md)
> - The voice → [`docs/character_spec.md`](docs/character_spec.md)
> - Project rules → [`CLAUDE.md`](CLAUDE.md)

It is a toy next to GPT-3.5 (~1000× smaller), and the docs say so plainly. The
point isn't to beat anything — it's to reproduce the whole ChatGPT-style recipe,
from zero, on one consumer GPU. A reproducibility study with a personality.

**License:** Apache-2.0 · **Author:** ninjahawk
