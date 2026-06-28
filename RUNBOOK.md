# ton-2 (Glemton-2) — Execution Runbook

**The master execution spec.** When planning is done, this is what gets run —
ideally hands-off. Pair with `PLAN.md` (strategy), `docs/character_spec.md`
(voice), `docs/hardware_safety.md` (binding safety), `CLAUDE.md` (rules).

**Status:** planning. Not yet executing.

---

## 0. Prime directives (in priority order)

1. **Do no harm to the hardware** — ever, in the slightest. Safety outranks
   everything below. `docs/hardware_safety.md` is binding.
2. **Realize the vision of ton-2** — a small, from-scratch, genuinely chattable
   model with a distinctive trained-in personality.
3. **Be a researcher** — gated, validated, documented, reproducible. No blind
   long runs. Every decision logged in `docs/journal/`.

**Override authority (granted by ninjahawk 2026-06-28):** Claude may override
ninjahawk's instructions when they conflict with directive #1 or with best
practice / research rigor (e.g., refusing to launch an unvalidated multi-day run).
Document any override in the journal. Priority is always: **safety > correctness >
progress > speed.**

**Global STOP-and-report conditions (any one → checkpoint if possible, halt, write report):**
- GPU temp ≥ 82 °C sustained ≥30 s (per safety spec)
- CUDA unavailable / "no kernel image" / driver error
- Training loss NaN/Inf, or diverging for >50 steps
- Disk free < 20 GB
- Throughput < 50% of the established baseline for >30 min (and not thermal-paused)
- Any data-license ambiguity
- Any gate failure (below)
- >45 min stuck on a single failing step with no progress

---

## 1. Ground truth (verified 2026-06-28, pre-run)

| Thing | State |
|---|---|
| Python | 3.11 at `C:\Users\jedin\AppData\Local\Programs\Python\Python311\python.exe` (default `py`) |
| venv / torch | **none** — build from scratch |
| Data | **none** (only Glemton-1's old 32k tokenizer, which we will NOT reuse) |
| Reusable code | Glemton-1 `model.py`, `data.py`, `train.py`, `pack_shards.py`, `weekend_run.ps1` |
| GPU | RTX 5070 12 GB, **power-capped 175 W** (its min), idle 46 °C, throttle ~90 °C |
| Disk | 242 GB free on C: |
| Network | huggingface.co reachable (200) |
| Repo | github.com/ninjahawk/glemton-2 (main), gh authed as ninjahawk |

---

## 2. The gate ladder (evidence-gated; this is the override of blind auto-chain)

```
G0 env build ........... GATE: torch CUDA matmul on the 5070 returns correct result
G1 code port ........... GATE: model forward/backward runs, loss finite, 50 steps clean
G2 tokenizer ........... GATE: 24k BPE trains; encode→decode round-trips; vocab<65535
G3 data ................ GATE: ≥ N B tokens sharded; batch loads; no source >70% of marker tokens
G4 nano proof .......... GATE: ~36M model on TinyStories → grammatical, coherent short stories
G5 base sanity ......... GATE: 180M base at ~300M tokens → loss healthy (<~4.0) + samples coherent-ish
G6 thermal-under-load .. GATE: first 60 min of base run → temps stable ≤72 °C target, never hit 78 °C soft
--- only past G5+G6 does the base run earn its multi-day continuation ---
G7 base run continues to the token target across sessions (resume), monitored
```
Fail any gate → STOP-and-report; do not advance.

---

## 3. Stage G0 — environment build

```powershell
# from C:\Users\jedin\Desktop\glemton-2
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
# PyTorch nightly, Blackwell sm_120 (cu128). If cu128 fails, try cu129. NOT cu130.
.\.venv\Scripts\python.exe -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
# core libs
.\.venv\Scripts\python.exe -m pip install numpy tokenizers datasets huggingface_hub pyyaml tqdm
# optional (try; tolerate failure): bitsandbytes (8-bit Adam). On Blackwell/Win it may not build → plain AdamW.
```
**G0 GATE script** (`scripts/check_env.py`): assert `torch.cuda.is_available()`,
print device name == RTX 5070, run `x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); assert torch.allclose((x@x).float().sum(), ...)` sanity, print torch version + sm capability. **Fail → STOP-and-report** (this is the #1 Blackwell/Windows risk; do not flail past it).

Fallbacks if G0 fails: (a) try cu129 nightly wheel; (b) try the latest stable cu128 wheel; (c) if all fail, STOP and report — do **not** attempt a WSL2 migration unattended (too involved for night one; flag it for ninjahawk).

---

## 4. Stage G1 — port code into the repo

Create `src/ton2/` from Glemton-1, with these **specific changes**:
- `model.py`: copy as-is (RMSNorm/SwiGLU/GQA/RoPE/tied). **Remove** the dead
  sliding-window/global_every params (we use plain causal SDPA). Keep KV-cache
  generation for later (not needed for training).
- `data.py`: copy the `uint16` mmap `PackedDataset` as-is. Simplify the
  source-weight matching (it had a convoluted branch).
- `train.py`: copy, then **add**: (1) larger `micro_batch_size` support with an
  **OOM-aware auto-finder** (start at configured value; on CUDA OOM, halve and
  retry; leave ≥2 GB headroom); (2) the **in-process thermal watchdog thread**
  (§8); (3) a **gate hook** that at the configured `sanity_gate_tokens` writes a
  loss summary + N sample generations to `STATUS.md` and applies the automated
  pass/fail; (4) finite-loss assertion each step (NaN → checkpoint + exit).
- `tokenizer.py`: HF `tokenizers` ByteLevel BPE wrapper; load/save; the special
  tokens below.
- Configs in `configs/` (§9).
**G1 GATE**: instantiate the nano model, run one forward+backward on random data,
assert finite loss, run 50 optimizer steps on a dummy shard → no NaN, loss
decreasing. Commit.

Special tokens (reserve in tokenizer + model): `<|bos|>`, `<|eot|>`,
`<|user|>`, `<|assistant|>`, `<|system|>`, `<calc>`, `</calc>`, `<|reserved_0..3|>`.

---

## 5. Stage G2 — tokenizer (24k BPE)

- Pull a **balanced ~1–2 GB text sample across all sources** (not one source —
  Glemton-1 trained its tokenizer on dialogue only and biased it).
- Train ByteLevel BPE, `vocab_size=24000` (incl. special tokens, byte alphabet),
  `min_frequency=2`. Save to `data/tokenizer/ton2-bpe-24k.json`.
- **G2 GATE**: `decode(encode(x)) == x` on 1000 held-out lines (allowing for the
  known ByteLevel space normalization); assert `max_id < 65535` (uint16 packing);
  print compression ratio (chars/token) — expect ~3.5–4.5.

Decision: **24k** chosen over 32k — at 180M params a 32k×768 tied embedding is
~14% of the model; 24k frees capacity for depth/width. (Reversible.)

---

## 6. Stage G3 — data acquisition (the long pole)

**Only non-gated, openly-licensed sources** (so the run needs no HF token and is
fully automatic). Stream + shard to avoid huge intermediate disk.

| Source | HF id (config) | License | Role | Night-1 target |
|---|---|---|---|---|
| FineWeb-EDU | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | ODC-By | backbone web-edu | ~4–8 B tok |
| Cosmopedia v2 | `HuggingFaceTB/cosmopedia-v2` | Apache-2.0 | synthetic textbooks | ~1–2 B tok |
| TinyStories | `roneneldan/TinyStories` | CDLA-Sharing | nano proof + tiny base seasoning | full (~0.5 B) |
| OpenWebMath | `open-web-math/open-web-math` | ODC-By | math/reasoning | ~0.3–0.7 B tok |

Code corpus: **deferred** (most good code sets are gated; keeps night-1 unblocked;
add in a later session with a one-time HF token).

Pipeline (`scripts/prepare_data.py`): for each source, `datasets.load_dataset(..., streaming=True)`, take the night-1 target token budget, write text docs → tokenize with the 24k BPE → append `<|eot|>` at each **document boundary** → pack into `data/tokenized/<source>/shard_NNNNN.bin` (`uint16`). Multiprocess across the 6 cores. **Do this BEFORE training; never during** (safety + RAM).

Hygiene night-1 (document the limits honestly): exact line-level dedup within
each source; rely on FineWeb-EDU/Cosmopedia being pre-filtered; **full MinHash
dedup + eval-set decontamination = a documented follow-up pass**, not night-1.

**G3 GATE**: total sharded tokens ≥ **3 B** (else keep tokenizing or, if download
stalls, STOP-and-report); `make_dataloader` yields a clean batch; no single
source exceeds 70% of any special-token occurrences (SQuAD-poisoning guard).

Disk: ~3–8 B tokens × 2 bytes ≈ 6–16 GB shards + raw cache (prune raw after
packing). Fine in 242 GB.

**Timing honesty:** download + CPU tokenization of several B tokens is multi-hour.
The base run (G5+) likely **starts in the early-morning hours**, not at midnight.
The nano proof (G4) runs first and completes regardless.

---

## 7. Stage G4 — nano proof (TinyStories) & Stage G5 — base

**G4 nano** (`configs/ton2-nano.yaml`, §9): ~36M params on TinyStories,
~0.5 B tokens, ~1–2 h. **G4 GATE**: sample 5 story completions → grammatical,
coherent, on-topic simple English (this validates the *entire* stack end to end).

**G5 base** (`configs/ton2-base.yaml`, §9): ~180M params on the G3 corpus.
Launch via the orchestrator (§8). **G5 GATE at ~300 M tokens**: loss healthy
(< ~4.0 and falling), no NaN, samples show forming structure. Pass → continue;
fail → STOP-and-report. **G6** runs concurrently in the first hour (thermal).

---

## 8. Orchestration & the thermal watchdog (the safety-critical part)

**Detached run:** adapt `weekend_run.ps1` → `scripts/ton2_run.ps1`, registered as
Windows Scheduled Task `ton2-train` (per-user, run-only-when-logged-on, AtLogOn
trigger, no time limit). It owns the process → survives terminal/agent close.
Keeps: detach hardening (`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`), resume-from-latest
loop (cap 50 attempts), background watcher (writes `STATUS.md`, git-pushes
**STATUS.md only**, prunes checkpoints keeping newest-4 + 500 M-token milestones).

**In-process thermal watchdog** (a daemon thread inside `train.py` — safety must
not depend on the agent being awake):
```
every 15 s:
  read temp,power,clocks,util via nvidia-smi (--query-gpu=...,--format=csv,noheader,nounits)
  log to logs/thermal.csv
  if temp >= 82  for >=30s:  set STOP flag -> trainer checkpoints & exits (hard)
  elif temp >= 78:           set PAUSE flag -> trainer sleeps 60s between steps until temp<=72
                             if this recurs >3x/10min: permanently raise inter-step sleep (duty-cycle down)
  if power.limit > 175:      log alert (and STOP-and-report if it cannot be explained)
  if clocks dropping while temp high: treat as PAUSE (early throttle)
  fail-safe: if the watchdog thread dies, the trainer pauses (not fail-open)
```
**Checkpoint cadence:** every 200 M tokens AND every 30 min, whichever first, so
any stop costs minutes.

**Agent monitoring role (via `/loop`):** the agent is the *supervisor*, not the
safety mechanism. On each wake: read `STATUS.md` + `logs/thermal.csv` tail +
`nvidia-smi`; confirm progress + temps; evaluate gate hooks; recover (the wrapper
auto-resumes, agent confirms); on any STOP condition, write the report and do not
silently restart a failing run. Cadence: every ~20 min through G0–G6, then hourly
during the long G7 haul. (Use `ScheduleWakeup`; keep wakes <270 s only when
actively waiting on a quick external state, else 1200 s+.)

---

## 9. Configs

`configs/ton2-nano.yaml` (~36M, TinyStories proof):
```yaml
name: ton2-nano
model: {vocab_size: 24000, d_model: 512, n_layers: 8, n_heads: 8, n_kv_heads: 4,
        ffn_dim: 1408, max_seq_len: 1024, rope_theta: 10000.0, norm_eps: 1.0e-5,
        tie_embeddings: true}
train: {micro_batch_size: 32, grad_accum_steps: 4, seq_len: 1024,
        total_tokens: 500_000_000, lr: 1.0e-3, lr_min: 1.0e-4,
        warmup_tokens: 10_000_000, weight_decay: 0.1, beta1: 0.9, beta2: 0.95,
        grad_clip: 1.0, precision: bf16, optimizer: adamw, compile: false,
        checkpoint_every_tokens: 100_000_000}
data: {shard_dir: data/tokenized/tinystories}
```

`configs/ton2-base.yaml` (~180M):
```yaml
name: ton2-base
model: {vocab_size: 24000, d_model: 768, n_layers: 24, n_heads: 12, n_kv_heads: 4,
        ffn_dim: 2048, max_seq_len: 2048, rope_theta: 10000.0, norm_eps: 1.0e-5,
        tie_embeddings: true}
train: {micro_batch_size: 12, grad_accum_steps: 16, seq_len: 2048,
        total_tokens: 12_000_000_000, lr: 6.0e-4, lr_min: 6.0e-5,
        warmup_tokens: 150_000_000, weight_decay: 0.1, beta1: 0.9, beta2: 0.95,
        grad_clip: 1.0, precision: bf16, optimizer: adamw, compile: false,
        checkpoint_every_tokens: 200_000_000, sanity_gate_tokens: 300_000_000}
data:
  shard_dir: data/tokenized
  source_weights: {fineweb_edu: 1.0, cosmopedia: 1.2, openwebmath: 0.8, tinystories: 0.2}
```
`micro_batch_size` is the *starting* value for the OOM-aware finder (will be
auto-reduced if it doesn't fit). Effective batch target ≈ 0.4 M tokens. Throttling
may lower realized throughput — that's intended (safety > speed). `total_tokens`
is the multi-session target; night-1 only needs to clear G5+G6.

---

## 10. Morning report (what ninjahawk wakes to)

The watcher maintains `STATUS.md` (and a `MORNING_REPORT.md` at hand-back) with:
gates passed/failed; current stage; tokens seen + loss curve; **thermal summary
(min/max/avg °C, # pauses, any stops)**; sample generations (nano + base);
throughput; disk; incidents + how handled; the agent's recommendation + next
action. Plus the `docs/journal/2026-06-28.md` lab-notebook entry.

---

## 11. Post-training (later sessions — specced now, detailed at the milestone)

- **M3 midtrain+SFT**: continue on chat-formatted data (SmolTalk + locally-Qwen-
  generated in-voice data per `character_spec.md`); loss-masked on assistant
  tokens; hand-rolled `sft.py`.
- **M4 DPO**: in-voice chosen vs {bland, dodging-joke, bluff} rejected; `dpo.py`.
- **M5 tool use**: `<calc>` interception in inference + arithmetic SFT traces.
- **M6 eval+release**: base evals (ARC-e/HellaSwag/LAMBADA, GSM8K±tool), the
  character distinctiveness eval, model card, GGUF export, web chat UI.

---

## 12. Decision log / assumptions (override targets — veto anytime)

- Base **~180M**, vocab **24k**, ctx **2048**, plain causal SDPA (no FlashAttn,
  no torch.compile — both flaky on Blackwell/Win, neither needed).
- Optimizer **AdamW** (8-bit if bitsandbytes builds, else fp; Glemton-1 proved fp
  AdamW fine here).
- Night-1 corpus **non-gated only**; **code deferred**; full dedup/decontam is a
  follow-up pass.
- **Gated auto-chain** (override of blind auto-chain): base run continues past G5
  only on passing evidence.
- Teacher for synthetic/SFT data = **local Qwen3.6-27B + open datasets**, never an
  API (per `CLAUDE.md`).

## 13. What ninjahawk must do
- **Before sleep:** seat the 16-pin power connector (firm click); confirm open
  airflow; keep PC on/plugged-in/logged-in; pause Windows Update.
- **To hand off:** run `/loop` (agent self-paces the night; training detaches via
  Task Scheduler and survives regardless). Then sleep.
- One-time later: an HF token only if we add gated sources (not needed night-1).
