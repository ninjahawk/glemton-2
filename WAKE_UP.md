# WAKE_UP — live handoff (read FIRST on a new session)

**Updated:** 2026-06-28 ~15:50. **State: base pretrain RUNNING, healthy.**

## Right now
- **Base run training:** ton2-base — **346M / 6B tokens (5.8%)**, loss **3.41** (from ~9), **26.4k tok/s**, GPU **68°C** (max 71, **0 thermal pauses**), 175W cap. ETA to 6B: ~2.5 days.
- Latest checkpoint: `checkpoints/ton2-base/step_1309_tokens_343M.pt`.
- Launched as a background `python -m ton2.train configs\ton2-base.yaml` + a background checkpoint pruner. **These were launched through the Claude session and may DIE when the session restarts** (harness processes aren't fully detached — Glemton-1 lesson). That's fine — resume instructions below.

## ON RESTART — do exactly this:
1. **Check if the run survived:** `nvidia-smi --query-compute-apps=pid --format=csv,noheader`. If a python pid is listed AND `STATUS.md` timestamp is recent → it's alive; just read `STATUS.md` and monitor. Done.
2. **If it died** (no python on GPU / STATUS.md stale) — resume from the latest checkpoint as a BACKGROUND job:
   ```powershell
   Set-Location C:\Users\jedin\Desktop\glemton-2
   $env:PYTHONPATH="src"; $env:HF_HUB_DISABLE_PROGRESS_BARS="1"; $env:FOR_DISABLE_CONSOLE_CTRL_HANDLER="1"
   $latest = (Get-ChildItem checkpoints\ton2-base\step_*.pt | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
   & ".venv\Scripts\python.exe" -m ton2.train configs\ton2-base.yaml --resume $latest
   ```
   Loses ≤30 min (checkpoints land ~every 30 min). Then relaunch the pruner (hourly loop keeping newest 4 + every-1B-token checkpoint).
3. **Monitor** via `STATUS.md` (tokens/loss/tok-s/temp, written every 10 steps). Surface to ninjahawk only on milestones/issues — **conserve usage / 5-hr windows**.

## Key facts (don't re-learn these)
- **Throughput:** mb=4 + seq=1024 + VRAM **headroom** = 26k tok/s. mb=8 maxed VRAM (12MB free) → allocator thrash → 5k. On this 12GB/Windows card, LEAVE HEADROOM. FlashAttention + `expandable_segments` are unsupported here.
- **VRAM hog:** the logits tensor (mb×seq×vocab×4) — keep micro_batch small.
- **Launcher:** run training via DIRECT `python -m ton2.train` in background. `run_train.ps1` detached had a stdio bug (fixed to `*>>`, still unproven detached).
- **Safety:** in-process ThermalGuard (78°C pause / 82°C stop+sentinel), binding regardless of the agent. GPU 175W-capped. Validated: hours at max 71°C, 0 pauses.
- **Config (ton2-base.yaml):** mb=4, accum=64, seq=1024, total 6B, sanity_gate 300M (loss<6.0 — PASSED at 3.42), tied embeds, vocab 24k.

## Gates
G0 env ✅ · G1 code ✅ · G2 tokenizer ✅ · G3 data ✅ (7.85B tok) · G4 nano ✅ (coherent stories) · G5 sanity ✅ (loss 3.42) · G6 thermal ✅ → base run finishing **G7** (to 6B).

## Next phase (after base completes)
**M3: SFT** — turn the fluent base into the chatty, witty-deadpan ton-2 (`docs/character_spec.md`). Then M4 DPO → M5 calculator tool → M6 eval → M7 on-device iOS app. SFT data-gen needs the GPU, so it waits for the base run to finish.
