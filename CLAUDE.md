# Glemton-2 — project rules for Claude

> Read `PLAN.md` first. This file is the stable operating discipline; `PLAN.md`
> is the actual technical plan. A `WAKE_UP.md` will hold live run state once we
> start building (we are not building yet — we are in planning).

## What Glemton-2 is

A from-scratch, small (~150–360M param) decoder-only language model trained on a
single RTX 5070 (12 GB) that you can genuinely **chat** with — coherent, follows
simple instructions, knows common facts, does basic math (with a calculator
tool), holds a back-and-forth. Same model *family* as Glemton-1, but a clean new
repo and a deliberately **inverted philosophy**: where Glemton-1 was a pure base
model with no fine-tuning, Glemton-2 follows the full modern pipeline
(pretrain → SFT → preference-tuning), which is the only way to get the
"talk-to-it-like-early-ChatGPT" feel the project is about.

**The original twist:** a distinctive, *trained-in, measurable* personality — witty
& deadpan, grounded in honesty (spec: `docs/character_spec.md`). The
architecture/recipe is the standard well-trodden path; the originality is the data
we create, the character, the analysis, and the writeup.

Positioning: a **reproducibility study** — can one person rebuild the
ChatGPT-style recipe (the InstructGPT 3-step, modernized) on one consumer GPU,
from the ground up, and produce something genuinely nice to talk to? The
constraint *is* the achievement.

## Core working rules (read first)

### Honesty
Be **completely honest** with ninjahawk at all times. No flattery, no "great
idea!", no softening bad news, no overstating what's working. If a benchmark is
mediocre, say it's mediocre. If a claim of novelty/capability is wrong, retract
it and fix the docs. Set expectations on capability *down*, not up — a 200M
model is a toy next to GPT-3.5, and we say so plainly while making it the best
toy we can.

### Novelty/claims require research, not assumption
Every "novel / first / best" claim must be verified (arXiv, HF, GitHub,
leaderboards) **before** acting on it. Document prior art honestly in
`docs/prior_art.md`.

### Attribution
Author/co-author attribution is **ninjahawk only** on every commit, PR, and
release note. Never add `Co-Authored-By: Claude`, `Generated with Claude Code`,
or any AI attribution footer.

### Training-data provenance (decided — see PLAN.md §8)
- **Do not train on Claude/GPT/Gemini API outputs.** Not because a 200M toy is a
  "competing model," but because (a) it's a cleaner, more impressive story for
  lab applications to own the whole stack, and (b) it removes all ToS ambiguity.
- Teacher for any synthetic/distilled data = the **local Qwen3.6-27B** (already
  installed, offline, zero ToS issues) and/or **openly-licensed synthetic
  datasets** (Cosmopedia, SmolTalk, etc.). Track license per source in
  `docs/data.md`.
- Only permissively-licensed / public-domain / openly-licensed corpora. No
  unlicensed web scrapes, no Reddit bulk, no ToS-restricted sources.

## Hardware envelope (training host — do not exceed without explicit approval)
- GPU: RTX 5070, 12 GB VRAM, Blackwell sm_120, CUDA 12.8/12.9 nightly toolchain.
- CPU: Ryzen 5 9600X (6c/12t). RAM: 31 GB. Disk: ~240 GB free on C:.
- Single GPU only. No cloud, no rented H100s, no multi-node. The whole point is
  the consumer-GPU constraint.
- Pretraining GPU-burn budget: target ≤ ~2 weeks/run, ≤ ~4 weeks total.
- Never run heavy CPU/IO data-prep jobs while the GPU is training (Glemton-1 lost
  a run to host-RAM exhaustion this way).

## Documentation discipline (this is half the portfolio value)
Every non-trivial decision (arch pick, data filter, hyperparam, eval result)
gets a dated entry in `docs/`. Required artifacts at release:
`MODEL_CARD.md`, `docs/architecture.md`, `docs/training.md`, `docs/data.md`,
`docs/evals.md`, `docs/prior_art.md`, `docs/journal/YYYY-MM-DD.md` (lab notebook,
one entry per session). No undocumented magic. The paper trail is the proof of
capability for AI-lab applications, as much as the weights.

## Conventions
- Python 3.11, type-hinted, `ruff` + `black`. `uv` for envs. Configs in YAML.
- PyTorch nightly cu128/cu129 (NOT cu130 — Blackwell support still patchy).
- Every training run: 6-char run ID + committed config snapshot beside the ckpt.
- Decoupled long runs via Windows Task Scheduler (NOT harness-launched processes
  — they are not detached and die when the tool call ends; Glemton-1 lesson).
- Commit messages: imperative, present tense, ninjahawk attribution only.
