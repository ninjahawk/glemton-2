# glemton-2

A small model trained from scratch on a consumer GPU.

This repo is a full-stack, from-scratch implementation of a small chat language
model — custom tokenizer, pretraining, supervised finetuning, preference tuning
(DPO), calculator tool use, and a chat web UI — in one clean, minimal, hackable,
dependency-light codebase that runs end to end on a single RTX 5070 (12GB).

ton-2 (a.k.a. Glemton-2) is the sequel to Glemton-1, and its inversion.
Glemton-1 was a pure base model trained on ~2B tokens of movie subtitles with no
finetuning — which is exactly why you couldn't really talk to it. ton-2 runs the
whole modern pipeline, so you can.

The one twist worth pointing at: **ton-2's personality is trained into the
weights, not bolted on with a system prompt.** The voice is dry, deadpan, and
honest — it answers first and jokes second, and it tells you when it doesn't
know. We also measure that the personality took: a blind judge should be able to
pick ton-2 out of a lineup of similar-size models. That's the small research
result hiding inside the engineering.

## Scale, honestly

ton-2 is ~170M parameters — GPT-2 small/medium size class. The whole point is what
one person can build from scratch on hardware they already own; the constraint is
the project. Read it the way you'd read nanochat or a GPT-2 reproduction: a
research artifact and a complete, legible pipeline, end to end.

## The pipeline

```
tokenizer → pretraining → midtraining → SFT → DPO → tool use → eval → chat UI
```

This is the InstructGPT recipe (SFT → reward model → RL), modernized to
**SFT → DPO** so it fits on one GPU — the same sequence of stages that turned
GPT-3 into ChatGPT, and the same stages nanochat uses, scaled down to consumer
hardware. Pretraining leans on high-quality data (FineWeb-EDU + synthetic
textbooks + math) rather than raw web scale, because at this size the tokens have
to count.

## Status

Early build. Environment validated on the RTX 5070 (Blackwell / sm_120), model +
dataloader implemented, pipeline scoped end to end.

- [x] env + GPU validated · model + dataloader · plan + runbook
- [ ] tokenizer · data · nano proof · base pretrain · SFT · DPO · tool use · eval · UI

## Read more

- `PLAN.md` — the full plan and the reasoning behind every choice
- `RUNBOOK.md` — exact, gated execution steps
- `docs/character_spec.md` — the voice, defined precisely
- `docs/hardware_safety.md` — keeping the GPU safe on long unattended runs

Trained on a single RTX 5070. License: Apache-2.0. Author: ninjahawk.
