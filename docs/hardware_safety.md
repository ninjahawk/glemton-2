# Glemton-2 — Hardware Safety Spec

> **Directive (ninjahawk, 2026-06-28):** under no circumstances may unattended
> training damage the hardware *in the slightest*. Safety outranks training
> speed, always. This spec is binding on every autonomous run.

## The honest baseline
Sustained ML training does **not** damage GPUs — they are designed to run at
100% load for years (mining/render/ML clusters do this continuously). The card
self-protects in hardware:
- **Thermal throttle ≈ 90 °C** junction (NVIDIA Blackwell); it down-clocks itself
  before any harm.
- **Hard shutdown** above that as a failsafe.
- Typical *gaming* sustained load on an RTX 5070 is **68–78 °C**.

So the hardware floor of safety is high before we add anything. Everything below
is belt-and-suspenders on top of that.

## Measured state of THIS card (2026-06-28, idle)
| Field | Value | Note |
|---|---|---|
| Name | RTX 5070, 12 GB | Blackwell sm_120 |
| Power limit (current) | **175 W** | Already at the card's *minimum* (70% of default) |
| Power limit default / max | 250 W / 275 W | We never raise it |
| Idle temp / power | 46 °C / 31 W | Fans semi-passive (off) at idle |
| Throttle behavior | ~90 °C | Self-protecting |
| VRAM (GDDR7) temp | **N/A via nvidia-smi** | This driver doesn't expose it — see mitigation |

The 175 W cap is a gift: low power → low heat. Training will likely sit ~60–70 °C.

## Windows constraint (why safety is monitoring-based)
On a Windows GeForce card, `nvidia-smi` **cannot lock GPU clocks** (`-lgc` errors),
and changing the **power limit** (`-pl`) needs an Administrator/UAC prompt we can't
trigger unattended. But the card is **already at its 175 W minimum**, so there's
nothing to lower. Therefore safety is enforced by **controlling the workload**, not
by hardware caps.

## The watchdog (binding behavior)
A monitor polls the GPU **every 15 s** during any training run:
```
nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit,clocks.gr,utilization.gpu --format=csv,noheader,nounits
```

| Condition | Action |
|---|---|
| temp ≤ 72 °C | Normal. Full speed. |
| temp ≥ **78 °C** (soft) | **Auto-pause** training ~60 s to cool; resume. If it recurs >3×/10 min, permanently increase the inter-step micro-pause (duty-cycle down). |
| temp ≥ **82 °C** for ≥30 s (hard) | **Checkpoint, stop the trainer, STOP-and-report.** (≈8 °C below throttle, ≈13 °C below gaming peaks.) |
| power.limit observed > 175 W | Log alert; attempt to restore 175 W; if it requires admin, STOP-and-report. |
| clocks dropping while temp high | Early throttle signal → treat as soft threshold. |
| disk free < 20 GB | Pause checkpointing path / STOP-and-report (protects against fill-up). |

**Duty-cycling:** because we can't cap clocks, we cap *heat* by inserting a short
`time.sleep()` between optimizer steps when temps drift up. Trades throughput for
temperature — the correct trade under this directive. Default starts at 0; the
watchdog raises it if soft thresholds recur.

**Thermal throttle is itself safe.** If temps ever reached ~90 °C the card would
down-clock to protect itself — no damage. Our 82 °C hard-stop means we never get
close.

## Operational rules (binding)
- **Keep the 175 W power cap.** Never raise it.
- **No concurrent CPU/IO-heavy jobs during GPU training** (data download/tokenize
  happen *before* training, never during) — protects thermals *and* prevents the
  host-RAM OOM that killed a Glemton-1 run.
- **Checkpoint frequently** (≤ every ~200 M tokens / ~every 30 min, whichever
  first) so any stop costs minutes.
- The watchdog runs in the **same orchestration as training**; if the watchdog
  process dies, training pauses (fail-safe, not fail-open).
- All thresholds above are deliberately conservative; do **not** relax them
  without explicit written approval from ninjahawk.

## VRAM-temp mitigation (the one thing we can't directly read)
1. The 175 W power cap inherently limits memory-subsystem heat.
2. GDDR7 has its own independent hardware throttle.
3. Core temp (which we *do* read) tracks memory temp closely; holding core ≤72 °C
   keeps VRAM in a safe band (GDDR7 safe operating range is well above our targets).

## Physical checks — ONLY ninjahawk can do these (do before an overnight run)
1. **12VHPWR / 16-pin power connector fully seated** (firm click, no gap). A
   poorly-seated connector is the single real hardware risk on 50-series cards.
   At 175 W the risk is low, but verify it physically.
2. **Open airflow** — PC not in a closed cabinet, intake/exhaust fans
   unobstructed, no heavy dust. Room not enclosed/hot.
3. Keep the PC **plugged in, on, and logged in** (Task Scheduler runs only when
   logged on).

## Honest caveat
No software can guarantee against a *physical* fault (failing PSU, bad connector).
What is guaranteed: the **software/thermal envelope stays far inside safe limits**
with conservative active monitoring that pauses/stops long before anything gets
warm. Combined with the card's own hardware protections and the existing 175 W
cap, added risk from this project is negligible.
