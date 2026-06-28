# Glemton-2 — Master Plan

**Author:** ninjahawk
**Status:** PLANNING (not started). This document is the full scope. We follow it.
**Date drafted:** 2026-06-28
**One-line:** Rebuild the ChatGPT-style training recipe — from scratch, modernized — at ~150–360M params on a single RTX 5070, and produce a small model that's genuinely nice to talk to.

---

## 0. TL;DR / What we're actually building

A from-scratch decoder-only LLM, **~180M parameters** (primary target; GPT-2-small/medium size class), trained on a single RTX 5070, taken through the **complete modern pipeline**:

```
custom tokenizer → pretraining → midtraining → SFT (chat) → preference tuning (DPO) → tool use (calculator) → eval → chat UI
```

This is the **InstructGPT 3-step recipe** (SFT → reward model → RLHF), modernized for one GPU (we replace PPO-RLHF with **DPO**, which is the standard single-GPU substitute). It's exactly the sequence OpenAI used to turn GPT-3 into ChatGPT, and what Karpathy's `nanochat` does end-to-end. We are doing the same thing, scaled to consumer hardware.

**Honest expectation (read this twice).** At ~180M params, Glemton-2 will be roughly in the capability class of **SmolLM2-360M-Instruct** or an instruction-tuned GPT-2-medium — *not* GPT-3.5/early-ChatGPT, which was **~175B params, ~1000× larger**. What we are reproducing is the **process and the experience shape** (a coherent assistant you chat with), not the scale. Realistic deliverable: a model that holds a conversation, follows simple instructions, answers common-knowledge questions (with hallucination on specifics), does basic arithmetic (reliably only with a calculator tool), has a consistent persona, and is genuinely fun to talk to. **The achievement is doing the whole pipeline, from zero, on one consumer GPU.** That is the story for AI-lab applications.

---

## 1. Goal, restated, and the GPT-1/2/3 framing

You asked for "GPT-1, GPT-2, or GPT-3 level." Those labels mix up two different axes, so let's separate them cleanly:

| | GPT-1 (2018) | GPT-2 (2019) | GPT-3 (2020) | ChatGPT (Nov 2022) |
|---|---|---|---|---|
| Params | 117M | 124M–1.5B | 175B | ~175B (GPT-3.5) |
| Layers / d_model | 12 / 768 | up to 48 / 1600 | 96 / 12288 | same as 3.5 base |
| Context | 512 | 1024 | 2048 | 2048→ |
| Vocab | ~40k BPE | 50,257 BPE | 50,257 | 50,257 |
| Train tokens | ~1–2B (BooksCorpus) | ~10B (WebText 40GB) | ~300B | 300B + RLHF |
| What it was | base LM | base LM | base LM (few-shot) | **base + SFT + RLHF** = the chatty thing |

**The key correction:** "the thing you talk back and forth with like early ChatGPT" is **not GPT-3**. GPT-3 (June 2020) was a raw base model you prompted with few-shot examples; it wasn't conversational. The conversational behavior came from **InstructGPT** (Jan 2022) and then **ChatGPT** (Nov 30, 2022, the "research preview" you're remembering) — which took a GPT-3.5 base and added **supervised fine-tuning + RLHF**. So "early-ChatGPT feel" = **base model + post-training**, and that's why Glemton-1 (a pure base model on movie subtitles) could never feel like ChatGPT no matter how long it trained.

**Therefore Glemton-2's target = GPT-2-size *base* + the full ChatGPT-style *post-training*.** Size in the GPT-1/GPT-2-small range (~120–360M); behavior shaped like early ChatGPT via SFT + DPO. That's the achievable, honest, and genuinely impressive intersection of your three asks.

**Why this is a good "reproducibility study."** GPT-2 has been reproduced cheaply many times (nanoGPT, llm.c, nanochat). What's *less* commonly done by one person is the **full chat pipeline** — base → SFT → preference → tool-use → served chat UI — entirely from scratch on a single consumer GPU, documented rigorously. That's the novel-enough, defensible framing. (Closest prior art: Karpathy's `nanochat`, which targets an 8×H100 node. Ours is the single-consumer-GPU adaptation — see §16.)

---

## 1.5 The twist — a trained-in, *measurable* personality (decided: witty & deadpan)

The standard pipeline above is the *well-trodden path*, and we walk it well rather than reinventing it (innovating on transformer internals at this scale is a debugging tax with little upside). **The originality lives in four places nobody can copy: the data we create, the personality, the analysis, and the writeup.** The headline twist is **personality, made rigorous.**

Most projects bolt a persona on with a system prompt at inference. Glemton-2 instead **trains a distinctive character into the weights** and then **measures that it worked** — turning "it has a fun tone" into a small, real result: *a method for instilling a consistent, distinctive voice at 180M params, with an eval proving it's robust — stays in voice across topics, under pressure, and across many turns, and a blind judge can pick Glemton out of a lineup of small models.*

**Being small is an advantage here.** A large RLHF'd model has a deep "default-assistant" prior you must fight; a small from-scratch model is a near-blank slate you can stamp a voice onto cleanly. The constraint becomes the feature.

**The voice (decided): witty & deadpan, grounded in honesty** — dry, concise, quietly funny, candid about what it doesn't know, never sycophantic. Full definition in `docs/character_spec.md`. Load-bearing rule: **answer first, wit second** (a joke never replaces a real answer; the wit drops on serious/emotional turns). This also continues the family arc — Glemton-1's defining trait was "no corporate-chatbot tone / zero sycophancy"; Glemton-2 turns that *absence* into a *character*, and reuses Glemton-1's sycophancy eval to help prove it.

How it's installed and measured: §8 (data), §11 (eval), and `docs/character_spec.md`.

---

## 2. Lessons from Glemton-1 (what we keep, what we invert)

Glemton-1 is a real asset — it proved the hard infrastructure works on this exact card. But its central design choices are why it "doesn't really work." We mine it for what's reusable and deliberately reverse the rest.

### What worked (reuse / carry over)
1. **The model code is clean and correct.** `src/glemton/model.py` (RMSNorm, SwiGLU, GQA, RoPE, tied embeddings, ~210 lines) is a solid, modern, from-scratch transformer. **Reuse it as the Glemton-2 starting point**, with fixes below.
2. **The single-GPU training loop works** (`train.py`: bf16, grad accumulation, cosine LR, checkpointing, 8-bit Adam with AdamW fallback). Reuse and extend.
3. **Empirical throughput anchor:** 365M model → **12,484 tok/s** on this 5070 (bf16, 8-bit Adam, micro=1, seq=2048, no compile, no FlashAttn). 3B tokens in 66h, one clean run, zero crashes. This is our budgeting baseline — and there's large headroom (see below).
4. **Custom 32k BPE tokenizer pipeline** exists and works.
5. **Unattended-run infra:** Windows Task Scheduler approach (survives terminal close / reboot), auto-resume from checkpoint, status file. Keep this pattern.
6. **Documentation discipline + honesty culture.** The dated journal, the model card, the "be honest" rule — this paper trail is half the portfolio value. Keep all of it.

### What failed (invert / fix)
1. **Pure base model, no SFT, no post-training → not a chatbot.** This was the #1 cause of "doesn't work": ~33% empty replies, no instruction-following, no answers. **Fix: the entire post-training stack (SFT + DPO) is now in scope.** This is the single biggest change.
2. **Movie-subtitle-dominant corpus → movie-subtitle model.** 2-line OpenSubtitles pairs with no scene context → no facts, no reasoning, no long-range recall (0%). **Fix: pretraining on high-quality knowledge/reasoning data (FineWeb-EDU + synthetic textbooks + code + math), not entertainment dialogue.**
3. **"No synthetic data" purism.** Glemton-1 banned synthetic/distilled data on principle — the exact thing that makes small models punch above their weight (Phi/TinyStories/Cosmopedia). **Fix: synthetic data is a core tool now**, generated by a *clean* teacher (local Qwen / open datasets), not banned.
4. **Undertrained.** ~2.5B unique tokens for 365M (Chinchilla floor ~7B). **Fix: token budget set by scaling laws + our GPU ceiling; overtrain a smaller model (inference-optimal regime).**
5. **Wasted the GPU.** micro_batch=1 used only **4.7 GB of 12 GB**, and `torch.compile` was off. Tensor cores were starved. **Fix: bigger micro-batch, attempt torch.compile, target much higher MFU/throughput.**
6. **"32k context / sliding-window" was vaporware** — the code comment admits sliding-window was never wired; it trained at seq=2048 pure causal. **Fix: don't over-spec. A chatbot at this scale needs ~2k context with plain full causal attention. Cut the unused complexity.**
7. **SQuAD-poisoning lesson:** at small scale, a single source that monopolizes a special token/format captures the model's behavior. **Fix: balance format-bearing sources; validate the data mix on a scaffold model before the full run.**

---

## 3. Hardware reality & the central constraint (the token-budget ceiling)

**This is the most important section for setting a realistic scope.**

- GPU: RTX 5070, **12 GB VRAM**, Blackwell sm_120. CPU: Ryzen 5 9600X. RAM: 31 GB. Disk: ~240 GB free.
- **VRAM is *not* the binding constraint** — Glemton-1 ran 365M in 4.7 GB. We can comfortably train up to ~360M, even ~700M with care. **Tokens-per-day is the binding constraint.**

**Throughput projection (from the 12,484 tok/s anchor):**

| Scenario | Realistic tok/s | Tokens in 1 day | in 1 week | in 2 weeks |
|---|---|---|---|---|
| Glemton-1 actual (365M, micro=1, no compile) | 12,500 | 1.1B | 7.5B | 15B |
| Glemton-2 ~180M, bigger micro-batch | ~22,000 | 1.9B | 13B | 27B |
| Optimistic (compile works on Windows) | ~30,000 | 2.6B | 18B | 36B |

So our **absolute pretraining ceiling is ~15–30B tokens** in a reasonable calendar. For comparison, SmolLM2-360M used **4T tokens** (≈150–250× our ceiling). **We cannot match production small models on raw token count.** This single fact drives the whole strategy:

1. **Keep the model small** so our token budget isn't pitifully undertrained relative to it. A ~180M model at ~12B tokens = ~65 tokens/param — well past the Chinchilla 20:1 floor, into the healthy inference-optimal "overtrain a small model" regime. The same 12B tokens on a 360M model = ~33:1 (more undertrained).
2. **Make every token count** via data quality (synthetic textbook data is the highest-capability-per-token lever known at small scale — Phi/Cosmopedia).
3. **Lean on post-training** for the "chatty/helpful" behavior, since base capability is token-limited. SFT+DPO are cheap (hours, not days) and give the biggest *perceived* quality jump.

**Toolchain caveats (verified 2026):** Blackwell sm_120 on **Windows** is still rough — PyTorch nightly cu128/cu129 works but FlashAttention prebuilt wheels are often missing and `torch.compile`/Triton on Windows is flaky. Mitigations: use PyTorch SDPA (memory-efficient/cuDNN attention, no custom build needed), treat `torch.compile` as a *try-with-fallback* optimization not a dependency, and keep **WSL2 (Ubuntu) as a fallback environment** where the CUDA/Triton/FlashAttention stack is much smoother. Phase 0 validates the chosen environment before any long run.

---

## 4. The pipeline we follow (mapped to OpenAI/Anthropic practice)

This is "the processes engineers at OpenAI and Anthropic do," scaled to one GPU. Each stage maps to a real production stage:

| Stage | What we do | OpenAI/Anthropic analogue | Cost on 5070 |
|---|---|---|---|
| **0. Tokenizer** | Train a byte-level BPE (vocab ~32k, or ~24k to save embedding params) on the pretraining mix | GPT-2/GPT-4 BPE, `tiktoken` | minutes–1h (CPU) |
| **1. Pretraining** | Next-token prediction on ~10–12B tokens of high-quality web-edu + synthetic textbooks + code + math | GPT-3 base pretraining | **days** (the big cost) |
| **2. Midtraining** | Continue training on conversation-formatted data + multiple-choice + tool-use traces to teach the chat format and special tokens | nanochat "midtraining"; OAI's data-blend annealing | hours |
| **3. SFT** | Supervised fine-tune on high-quality instruction/chat data (open + locally-distilled) | InstructGPT step 1 (SFT) | hours |
| **4. Preference tuning (DPO)** | Optimize on chosen/rejected pairs for helpfulness, format, refusal behavior | InstructGPT steps 2–3 (reward model + PPO), modernized to DPO | hours |
| **5. Tool use** | Teach a `<calc>…</calc>` calculator tool (and optionally a tiny retrieval hook) for math/facts | ChatGPT tools / nanochat Python sandbox | folded into SFT |
| **6. Eval + release** | CORE/ARC-easy/GSM8K(+tool)/custom chat evals; model card; GGUF export; web chat UI | OAI evals + model cards | days (parallel) |

**Why DPO instead of full RLHF/PPO:** DPO (Direct Preference Optimization) reaches comparable quality to PPO-RLHF on single-turn helpfulness while being dramatically simpler and single-GPU-friendly — no separate reward model, no online RL loop. It's the standard modern substitute and what we'll use. (Optional stretch: GRPO-style RL on verifiable tasks like arithmetic, as nanochat does — only if time permits.)

---

## 5. Architecture spec (Glemton-2 model)

Dense decoder-only transformer, Llama-style, **built on Glemton-1's verified `model.py`** with the fixes from §2. We deliberately keep it simple and standard — novelty goes into data/pipeline/eval, not exotic architecture (small exotic architectures are a debugging tax with little upside).

**Primary target — "glemton-2-base" (~180M):**

| Hyperparam | Value | Notes |
|---|---|---|
| Params | ~180M | Between GPT-2-small (124M) and -medium (355M) |
| d_model | 768 | |
| n_layers | 24 | |
| n_heads | 12 (head_dim 64) | |
| n_kv_heads | 4 (GQA) | KV-cache-efficient for inference |
| FFN | SwiGLU, dim ~2048 | |
| Norm | RMSNorm (pre-norm) | |
| Position | RoPE, θ=10,000 | θ=1e6 only needed for long ctx; we use short ctx |
| Context | 2048 | Plenty for chat; full causal attention (no sliding window) |
| Vocab | ~32k (consider 24k) | Smaller vocab saves embedding params at this scale |
| Embeddings | tied input/output | |
| Precision | bf16 weights + autocast | |

**Milestone models (the ladder, see §12):**
- **glemton-2-nano (~5–30M):** TinyStories-scale proof. d_model 512, L=8. Validates the whole stack in <1 day and should produce coherent, simple English.
- **glemton-2-base (~180M):** the real model (above).
- **glemton-2-medium (~360M) [STRETCH]:** Glemton-1's size; only if the 180M result is strong and time/tokens allow. Reuse Glemton-1's config (d_model 1024, L24-30, GQA 16/4, FFN 2752).

**Fixes vs Glemton-1 `model.py`:**
- Make micro-batch a real lever (it ran at 1); target micro-batch 8–24 at seq 2048/1024.
- Drop the unimplemented sliding-window path; plain causal SDPA.
- Add a proper KV-cache inference path (Glemton-1's generate was minimal) for a responsive chat experience.
- Add `torch.compile` behind a config flag (try; fall back gracefully on Windows).
- Add chat special tokens to the tokenizer/model from the start (`<|user|>`, `<|assistant|>`, `<|system|>`, `<|eot|>`, `<calc>`, `</calc>`).

---

## 6. Tokenizer

- **Algorithm:** byte-level BPE (GPT-2/GPT-4 style). Train with HuggingFace `tokenizers` (fast) or reuse Glemton-1's pipeline.
- **Vocab size:** 32k default; **strongly consider 24k.** At 180M params, a 32k×768 tied embedding is ~24.6M params (~14% of the model) "spent" on vocab. A 24k vocab frees params for depth/width. Decide by a quick ablation on the nano model.
- **Special tokens (reserve from day 1):** `<|bos|>`, `<|eot|>` (end-of-turn), `<|user|>`, `<|assistant|>`, `<|system|>`, `<calc>`, `</calc>`, plus a few spare `<|reserved_N|>`. Baking these into pretraining (even rarely seen) avoids the "add tokens later and they're untrained" problem.
- **Train on the *pretraining* mix**, not dialogue-only (Glemton-1 trained the tokenizer on dialogue, biasing it). A balanced edu+code+math+prose mix gives better compression on the data we actually use.
- **Decontaminate:** ensure no eval-set text leaks into tokenizer training (minor, but tracked).

---

## 7. Data plan — pretraining (§the highest-leverage part)

Strategy: **maximize capability-per-token** because tokens are our scarce resource (§3). Mix = mostly high-quality filtered web-edu, a strong dose of synthetic "textbook" data (the Phi/Cosmopedia lever), plus code and math for reasoning structure, plus *some* clean dialogue/prose for conversational register.

**Target ~10–12B unique tokens** (overtraining the 180M model to ~60–70 tok/param). All sources openly licensed; license tracked per source in `docs/data.md`.

| Source | License | Approx share | Role |
|---|---|---|---|
| **FineWeb-EDU** (HF, edu-classifier-filtered web) | ODC-By | 50–60% | Backbone: clean factual/edu English. The proven small-model base. |
| **Cosmopedia v2** (HF; synthetic textbooks/stories, Mixtral-generated, openly licensed) | Apache-2.0 | 15–25% | The "textbooks" lever — high capability/token; teaches explanation/reasoning register. |
| **Locally-distilled synthetic** (our Qwen3.6-27B generates targeted textbook/QA/explanations on weak topics) | ours/open | 5–10% | Fill knowledge gaps; teach the helpful-explainer voice; clean provenance. |
| **Code** (The Stack v2 / StarCoder data, permissive subset; or Python-edu) | permissive | ~10% | Reasoning structure + light coding ability. |
| **Math** (FineMath / OpenWebMath / synthetic arithmetic + word problems) | open | ~5–10% | Numeracy + step-by-step reasoning seeds. |
| **Clean dialogue/prose** (DailyDialog, Gutenberg subset, optionally scene-grouped OpenSubtitles) | PD/open | ~5% | Conversational register without dominating (Glemton-1 lesson: don't let it). |

**The teacher / "distillation from Claude" question — decided (and why):**
- You floated distilling from Claude. **Recommendation: don't use Claude (or any API) for training data.** The reason is *not* ToS fear (a ~170M from-scratch model isn't a "competing model" in any real sense). It's that the impressive, *ownable* part of this project is the data-generation craft itself: the prompting, curation, and filtering pipeline that manufactures exactly the data that gives ton-2 its capability and voice. Drive that with a model you run yourself and the whole loop is yours; call an API and you've outsourced the most interesting part.
- **Use instead:** your **local Qwen3.6-27B** (already installed, offline, free, zero ToS issues) as the synthetic-data teacher, plus **openly-licensed synthetic datasets** (Cosmopedia, SmolTalk, etc.). This is *sequence-level distillation* (we learn from the teacher's *text*, since no API exposes logits anyway) — the same technique Phi/Cosmopedia/SmolLM used. Same speedup, clean story.

**Data hygiene (do this — Glemton-1 had bugs here):**
- **Dedup:** MinHash-LSH document-level + exact line-level. Removes the memorization trap.
- **Decontaminate against eval sets** (GSM8K, ARC, our chat probes) by n-gram overlap; document the method.
- **Quality filter:** rely on FineWeb-EDU's classifier; for our own synthetic, spot-check + length/format filters.
- **Shard + pack:** pre-tokenize to binary shards (Glemton-1 has `pack_shards.py`); document-boundary-aware packing with `<|eot|>` separators; verify no single source monopolizes a special token (the SQuAD-poisoning lesson — validate on the nano/scaffold model before the full run).

---

## 8. Data plan — post-training (what makes it chatty)

**Midtraining (teach the chat format on top of the base):** a few hundred M tokens, continue-pretraining style, of: conversation-formatted data (SmolTalk, OpenAssistant-2, DailyDialog reformatted to our `<|user|>/<|assistant|>` template), multiple-choice (MMLU auxiliary-train, ARC) to teach the MC format, and a slice of tool-use traces (see §9). This is nanochat's "midtraining" stage and it markedly improves the SFT starting point.

**SFT (the core of "feels like an assistant"):** ~100–500k high-quality instruction/chat examples. Sources, all open or locally-generated:
- **Open SFT sets:** SmolTalk, OpenHermes-2.5 (open), Tülu-3 SFT mix, OpenAssistant — filtered for quality and license.
- **Locally-distilled custom set:** use Qwen3.6-27B to generate (a) the model's **persona/identity** answers ("who are you?", "who made you?" → answers as *Glemton-2 by ninjahawk*, not as Qwen/Claude — critical, models otherwise claim to be ChatGPT), (b) **basic-math** worked examples with the calculator tool, (c) **common-knowledge QA** in our voice, (d) graceful **"I don't know"** behavior to curb hallucination, (e) **refusals** for unsafe asks (lightweight, but present — labs notice safety hygiene), and (f) **emotional/serious cases where the wit is correctly dropped** (so the model learns the off-switch, not just the bit). **Every SFT example is written in Glemton's voice** per `docs/character_spec.md` — uniform in-voice data is what puts the personality in the weights without needing a system prompt.
- Format: strict chat template; loss masked on user turns (train only on assistant tokens). 1–3 epochs.

**Preference tuning (DPO):** ~10–60k chosen/rejected pairs. Sources: open preference sets (UltraFeedback, HH-RLHF-style open sets, Tülu preference mix), plus **locally-generated pairs** (sample two completions from our SFT model, rank with Qwen3.6-27B as judge → chosen/rejected). DPO sharpens helpfulness, formatting, conciseness, refusal consistency — and crucially the **character**: include pairs where `chosen` is in-voice + helpful and `rejected` is generic-bland, a joke-that-dodged-the-answer, or a confident bluff (per `docs/character_spec.md`), so DPO optimizes the *personality*, not just helpfulness. This is the step that takes it from "a fine-tuned base model" to "feels deliberately helpful."

**Identity/persona spec (write this early):** name (Glemton-2), maker (ninjahawk), what it is (a small from-scratch model), known limits ("I'm small, I can be wrong, I can use a calculator for math"). Bake consistently across midtrain/SFT/DPO so it never claims to be ChatGPT/Claude/Qwen.

---

## 9. Basic math (honest plan)

Small models are genuinely bad at multi-digit mental arithmetic — this is well-documented; reliable GSM8K at small scale comes from **tool use**, not raw weights (e.g. TinyGSM hit 80%+ only with a 1.3B model + verifier + 12M synthetic problems). So:

1. **Primary path — calculator tool use (nanochat-style).** Teach the model to emit `<calc>2348*19</calc>`; the inference harness intercepts, computes, injects the result, and the model continues. This makes "basic math" *actually reliable* (exact, any magnitude) instead of approximately-right. Train it via tool-use traces in midtraining + SFT.
2. **Secondary — native arithmetic SFT.** Include synthetic single/double-digit arithmetic and short word problems with step-by-step ("scratchpad") solutions so it can do simple sums without the tool and *knows when to reach for it*.
3. **Honest scope:** native mental math reliable to ~2-digit; anything harder routes to the calculator. Word problems: can set up simple ones, will make mistakes. We measure this (GSM8K with and without tool) and report it straight.

---

## 10. Training recipe & infrastructure

**Pretraining hyperparameters (starting point; tune on nano):**
- Optimizer: AdamW (β1=0.9, β2=0.95), or 8-bit Adam (bitsandbytes) if it builds on Blackwell/Windows — else plain AdamW (Glemton-1's fallback works). Weight decay 0.1 on matrices, 0 on norms/embeddings.
- LR: peak ~6e-4 (smaller model tolerates higher LR than Glemton-1's 3e-4), cosine to ~6e-5, warmup ~1–2% of tokens. Grad clip 1.0.
- Batch: micro-batch 8–24 (vs Glemton-1's 1), seq 2048, grad-accum to **effective batch ~0.5M tokens** (Glemton-1's 16k effective batch was tiny and noisy — this is a real fix).
- Precision bf16; `torch.compile` on if stable, else off; checkpoint every ~500M tokens; keep milestone ckpts.
- **Data order:** optionally curriculum — more synthetic/edu early, anneal toward chat-adjacent data late (cheap, small gains).

**Post-training hyperparameters:** SFT LR ~1e-5–2e-5, 1–3 epochs, cosine, loss-masked on assistant tokens. DPO LR ~5e-7–5e-6, β≈0.1, 1–2 epochs. (Standard ranges; tune on a held-out chat probe set.)

**Infrastructure (carry over Glemton-1's, harden it):**
- Pre-tokenized binary shards; streaming dataloader with per-source weights (Glemton-1 has this).
- **Unattended long runs via Windows Task Scheduler** (NOT harness-launched — those die when the tool call ends). Auto-resume from latest checkpoint on crash; write a `STATUS.md` every ~10 min; prune old ckpts.
- W&B (free tier) or local CSV + matplotlib for loss curves.
- **Never run heavy data-prep while the GPU trains** (host-RAM OOM killed a Glemton-1 run).
- Disk budget: ~10–12B tokens at 2 bytes/token ≈ 20–24 GB of shards + raw downloads (~tens of GB) + checkpoints (~0.7–2 GB each). Comfortable in 240 GB; prune raw after tokenizing.

---

## 11. Evaluation plan

Borrow the real eval surface (small subset), plus custom chat evals. Build the harness *before* the full run so we have go/no-go gates.

**Automated base-model evals (cheap, standard):**
- **CORE / DCLM-style** few-shot suite (what nanochat reports) or `lm-evaluation-harness` subset.
- **ARC-Easy, HellaSwag, PIQA, LAMBADA** — sanity that the base learned language + commonsense. Set realistic targets (e.g. ARC-Easy meaningfully above chance; we are not chasing leaderboards).
- **GSM8K** with-tool vs no-tool (report both honestly).

**Chat / assistant evals (custom — the ones that matter for the goal):**
- **Instruction-following probe set** (~50–100 prompts): does it answer at all, on-topic, in-format? (Glemton-1's failure mode was *empty replies* — track empty-reply rate explicitly; target <5% vs Glemton-1's ~33%.)
- **Common-knowledge QA** (~50 Qs): factual accuracy on basic facts; track hallucination rate.
- **Basic-math probe** (with/without calculator).
- **Persona/identity probe:** never claims to be ChatGPT/Claude/Qwen; states it's Glemton-2.
- **Character — the measurable twist** (full spec in `docs/character_spec.md`): (1) *distinctiveness* — a blind judge (Qwen-27B or human) picks Glemton out of a lineup of small models on the same prompts; (2) *consistency* — stays in voice across topics and ~10+ turns; (3) *wit-never-beats-help* — count answers that sacrifice correctness/completeness for a joke (target ~0); (4) *sycophancy markers* — reuse Glemton-1's probe (target ~0); (5) *calibration* — "I don't know" rate on unanswerable Qs + hallucination rate.
- **Multi-turn coherence:** short back-and-forth, does it hold context across ~10 turns (Glemton-1 scored 0% — low bar to beat, but measure it).
- **Tone/quality:** small rubric, optionally Qwen-27B-as-judge for pairwise vs the SFT-only checkpoint to confirm DPO helped.
- **Safety:** basic refusal probes.

**Comparison baselines** (run the same probes, side by side, for the writeup): Glemton-1, GPT-2 (124M), SmolLM2-135M/360M-Instruct, TinyLlama. Honest table.

---

## 12. Milestone ladder (each rung is a shippable artifact + go/no-go gate)

This is how we de-risk and always have something to show. We can stop at any rung.

| # | Milestone | Deliverable | Gate to proceed | Est. effort |
|---|---|---|---|---|
| **M0** | **Setup & toolchain validation** | Repo scaffolded, env (PyTorch nightly cu128/9 or WSL2) verified on the 5070, tiny train step runs clean (loss ↓, no NaN), tokenizer trains | Model does a clean 100-step run; tokenizer round-trips | 2–4 days |
| **M1** | **Nano proof (TinyStories)** | ~5–30M model trained on TinyStories → coherent, simple English; full pipeline (tok→train→sample→eval) exercised end-to-end | Generates grammatical, coherent short stories | 1–2 days |
| **M2** | **Base pretrain** | glemton-2-base (~180M) on ~10–12B high-quality tokens; base evals (ARC-e/HellaSwag/LAMBADA) at sane levels | Base coherent; beats GPT-2-124M on CORE-ish; loss curve healthy | 5–10 days GPU |
| **M3** | **Midtrain + SFT** | Chat-formatted, instruction-following model. **This is the first "talk to it" moment.** | Empty-reply rate <5%; follows simple instructions; persona consistent | 3–5 days |
| **M4** | **DPO** | Preference-tuned, noticeably more helpful/clean. | Qwen-judge prefers DPO over SFT >60%; no regression on evals | 2–4 days |
| **M5** | **Tool use / math** | Calculator tool working; reliable basic math; GSM8K-with-tool measured | Calc tool fires correctly; math probe reliable with tool | 2–4 days |
| **M6** | **Eval, polish, release** | Full eval table, MODEL_CARD, GGUF export, **web chat UI** (nanochat-style), writeup/blog post | All docs done; chat UI runs locally | 3–5 days |
| **M7** | **Stretch: medium (~360M) and/or RL** | bigger model and/or GRPO RL on math, if M2–M6 strong and time allows | — | optional |

**First "it talks back nicely" moment: end of M3** (realistically ~2.5–4 weeks in). Full polished release: M6.

---

## 13. Timeline & GPU-burn budget

Part-time, leveraging unattended overnight/weekend training (Glemton-1's infra makes this hands-off):

- **Calendar:** ~6–10 weeks end-to-end to M6, comfortably. M0–M1 in week 1; M2 pretrain runs unattended over ~1–1.5 weeks; M3–M6 are mostly short runs + iteration.
- **GPU-burn:** pretraining ~5–10 days (dominant cost); all post-training combined ~2–4 days; evals/iteration sprinkled. Well within the "≤4 weeks total GPU-burn" envelope.
- **Cheapest viable path** (if you want a result fast): nano proof (M1) + a smaller ~124M base on ~5B tokens + quick SFT — a "talks back" model in ~2 weeks, then improve.

---

## 14. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Blackwell+Windows toolchain eats days (FlashAttn missing, Triton/compile flaky) | High | Phase-0 validation; SDPA not FlashAttn; compile optional; **WSL2 fallback** ready |
| Base model undertrained / incoherent (token ceiling) | Medium | Keep model small (180M), max data quality (synthetic textbooks), overtrain to ~65 tok/param |
| Still doesn't follow instructions after SFT | Low–Med | Midtraining stage + quality SFT + loss masking; measure empty-reply rate as a gate |
| Hallucinates facts confidently | High (inherent at scale) | Train "I don't know" behavior; calculator for math; **set expectations honestly in model card** |
| Math stays unreliable | Medium | Calculator tool is the primary path; native math is best-effort + measured |
| Data-mix imbalance captures behavior (SQuAD-poisoning redux) | Medium | Balance format-bearing sources; validate mix on nano/scaffold before full run |
| Long unattended run dies | Medium | Task Scheduler + auto-resume + status file (proven in Glemton-1) |
| Identity confusion (claims to be ChatGPT) | High if unaddressed | Explicit persona SFT data; identity probe in eval |
| Scope creep / burnout | Real | Milestone ladder — every rung ships something; stop anytime with a real artifact |
| "Distillation" optics/ToS for lab applications | Low | Local Qwen + open data only; no API outputs in training; documented provenance |

---

## 15. Repo structure & toolchain

```
glemton-2/
  CLAUDE.md            # operating rules (done)
  PLAN.md              # this file
  README.md            # public-facing (write at M6)
  MODEL_CARD.md        # HF-style (M6)
  pyproject.toml       # uv-managed, py3.11
  configs/             # YAML per run (nano / base / medium / sft / dpo)
  src/glemton2/
    model.py           # from Glemton-1, fixed (§5)
    tokenizer.py
    data.py            # streaming dataloader, source weights
    train.py           # pretrain loop (from Glemton-1, extended)
    sft.py             # SFT (loss-masked chat)
    dpo.py             # preference tuning
    generate.py        # KV-cache inference
    tools.py           # calculator tool interception
    eval/              # eval harness + probe sets
  scripts/             # data download/prepare/pack, train launchers, task-scheduler wrapper, chat_cli, chat_web
  data/                # raw (gitignored), tokenized shards (gitignored), tokenizer/
  checkpoints/         # gitignored
  docs/                # architecture, training, data, evals, prior_art, journal/
  tests/               # probes, smoke tests
```

**Toolchain:** Python 3.11, `uv`, PyTorch nightly cu128/cu129, HF `tokenizers` + `datasets`, `bitsandbytes` (if it builds), `trl` (reference for DPO/SFT — or hand-roll for the "from scratch" story; decide per stage), `lm-evaluation-harness` (eval subset), `llama.cpp`/GGUF (export), a tiny FastAPI/static web UI for chat. W&B free tier or local logging.

---

## 16. How we frame this for AI-lab applications

The portfolio value is the **whole stack + rigor + honesty + an original twist**, not a benchmark number. **The headline original contribution is the personality:** a from-scratch 180M model given a distinctive, *measurable* character (witty & deadpan, grounded in honesty) — installed in the weights via SFT/DPO and proven with a distinctiveness eval (a blind judge can pick Glemton out of a lineup). That's the part that's *yours*; nanochat and GPT-2 clones don't have it. Lead with:
1. **"I reproduced the ChatGPT recipe from scratch on a single consumer GPU."** Tokenizer → pretrain → SFT → DPO → tool-use → served chat UI, all hand-built, fully documented.
2. **The engineering paper trail:** dated journal, every decision justified, scaling-law-informed token budget, data decontamination, eval harness with honest results and baselines, a model card that states limitations plainly.
3. **The honest comparison:** "here's where my 180M model lands vs GPT-2 / SmolLM2-360M, and exactly why." Labs value calibrated honesty over hype (and Glemton-1's culture already bakes this in).
4. **The constraint as the story:** what it takes to do this on 12 GB — the optimizations, the Windows/Blackwell fight, the throughput work.
5. **Clean provenance:** own tokenizer, own teacher (local 27B), open data — no "I called an API." 

Closest prior art to cite and differentiate from: **nanochat** (Karpathy) — same idea, but targets an 8×H100 node and is a teaching scaffold; ours is the **single-consumer-GPU, fully-from-scratch, rigorously-evaluated** adaptation, with our own data pipeline and a distilled-from-local-teacher data story. Also cite SmolLM2 (data recipe), Phi/Cosmopedia (synthetic-data lever), TinyStories (small-scale coherence), Chinchilla + inference-optimal scaling (token budget rationale).

---

## 17. Decisions I need from you (before M0)

None block the planning — but these shape the build. My recommendation is in **bold**:

1. **Primary model size:** **~180M (recommended)** for the best capability-per-token on our budget, vs ~124M (faster, weaker on facts) vs ~360M (Glemton-1 size, needs more tokens, riskier on our ceiling). 
2. **Vocab size:** **24k (recommended at this scale)** vs 32k (Glemton-1 reuse). Quick ablation on nano can settle it.
3. **Teacher for synthetic/distilled data:** **local Qwen3.6-27B + open synthetic datasets (recommended)** vs involving Claude/API (not recommended — optics + ToS). Confirm you're good dropping the Claude-distillation idea in favor of the local-teacher story.
4. **Ambition vs speed:** full ladder to M6 (~6–10 weeks) vs the "cheapest viable talks-back model in ~2 weeks" path, then iterate. 
5. **Use `trl` for SFT/DPO, or hand-roll** for a stronger "from scratch" claim? (Recommendation: hand-roll the SFT/DPO loops — they're short — but it's a real time/rigor tradeoff.)
6. **Repo:** new GitHub repo `ninjahawk/glemton-2`, Apache-2.0 (matching Glemton-1)?

## 18. What I need from the environment (when we start M0)
- Keep the PC on/plugged-in/logged-in for unattended runs (Task Scheduler is "run only when logged on").
- Pause Windows Update during long runs (manually, via Settings).
- ~30–40 GB free working space for data/shards/checkpoints during a run (we have 240 GB — fine).
- Confirm the local Qwen3.6-27B / Ollama is available for the data-generation stages.

---

## 19. References (research backing this plan)

- Karpathy, **nanochat** — github.com/karpathy/nanochat (full from-scratch ChatGPT pipeline; depth-parameterized; FineWeb-EDU pretrain, SmolTalk midtrain, SFT, GRPO RL, web UI).
- Allal et al., **SmolLM2: When Smol Goes Big** — arXiv:2502.02737 (data-centric small-model recipe; 135M/360M token counts; FineWeb-EDU/Cosmopedia mix).
- Gunasekar et al., **Textbooks Are All You Need** (phi-1) — arXiv:2306.11644; Li et al., **phi-1.5** — arXiv:2309.05463 (synthetic textbook data as capability lever).
- HuggingFace, **Cosmopedia** — huggingface.co/blog/cosmopedia (open synthetic-textbook dataset, 25B tokens, Mixtral-generated).
- Eldan & Li, **TinyStories** — arXiv:2305.07759 (small models can be coherent with clean/narrow data).
- Hoffmann et al., **Chinchilla** (20 tok/param compute-optimal); Sardana et al., **Beyond Chinchilla-Optimal: Accounting for Inference** — arXiv:2401.00448 (overtrain small models for inference).
- Ouyang et al., **InstructGPT** — "Training language models to follow instructions with human feedback" (2022) (the SFT→RM→PPO 3-step; the recipe behind ChatGPT).
- Rafailov et al., **DPO: Your Language Model is Secretly a Reward Model** — arXiv:2305.18290 (single-GPU-friendly RLHF substitute).
- Liu et al., **TinyGSM** — arXiv:2312.09241 (small-model GSM8K needs synthetic data + verifier/tool).
- PyTorch issue #164342 / Dao-AILab/flash-attention #2535 (Blackwell sm_120 on Windows toolchain status, 2026).
