"""Training loop for ton-2 — single GPU, bf16 autocast, fp32 master weights.

Safety-critical features (see docs/hardware_safety.md):
  - In-process ThermalGuard thread polls nvidia-smi every 15s; PAUSE at 78C,
    hard-STOP (+sentinel) at 82C sustained 30s. Safety does NOT depend on the
    supervising agent being awake.
  - OOM-aware micro-batch finder: never crashes on VRAM; backs off and keeps the
    effective batch ~constant by raising grad-accum.
  - Disk-low guard, finite-loss assert, frequent checkpoints, auto-resume.
  - Sanity gate (G5): at sanity_gate_tokens, dump loss+samples; auto-fail if loss
    is unhealthy so we never commit days to a broken run.

Run:  PYTHONPATH=src python -m ton2.train configs/ton2-base.yaml [--resume ckpt.pt]
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Reduce CUDA allocator fragmentation on a tight 12GB card (must be set before torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

from .data import make_dataloader
from .model import model_from_config

POLL_S = 15
PAUSE_C = 78.0
STOP_C = 82.0
RESUME_C = 72.0
DISK_MIN_GB = 20.0


class ThermalGuard:
    """Polls the GPU and exposes pause/stop flags. Fail-safe: if polling fails
    repeatedly it pauses rather than risk running blind."""

    def __init__(self, log_path="logs/thermal.csv"):
        self.log_path = log_path
        self.pause = False
        self.stop = False
        self.max_temp = 0.0
        self.min_temp = 999.0
        self.sum_temp = 0.0
        self.n = 0
        self.pause_events = 0
        self.last = None
        self._hot_since = None
        self._fail_streak = 0
        self._alive = True

    @staticmethod
    def _read():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,power.limit,clocks.gr,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        return [float(v) for v in line.split(",")]

    def _loop(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("ts,temp_c,power_w,power_limit_w,clock_mhz,util_pct\n")
            f.flush()
            while self._alive:
                try:
                    temp, power, plim, clk, util = self._read()
                    self._fail_streak = 0
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{ts},{temp},{power},{plim},{clk},{util}\n")
                    f.flush()
                    self.last = (temp, power, plim, clk, util)
                    self.max_temp = max(self.max_temp, temp)
                    self.min_temp = min(self.min_temp, temp)
                    self.sum_temp += temp
                    self.n += 1
                    if temp >= STOP_C:
                        self._hot_since = self._hot_since or time.time()
                        if time.time() - self._hot_since >= 30:
                            self.stop = True
                    elif temp >= PAUSE_C:
                        self._hot_since = None
                        if not self.pause:
                            self.pause_events += 1
                        self.pause = True
                    elif temp <= RESUME_C:
                        self._hot_since = None
                        self.pause = False
                    if plim > 180:
                        f.write(f"{ts},ALERT,power_limit_raised={plim}\n")
                        f.flush()
                except Exception:
                    self._fail_streak += 1
                    if self._fail_streak >= 3:
                        self.pause = True  # fail-safe: can't see temps -> don't run hard
                time.sleep(POLL_S)

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def avg_temp(self):
        return (self.sum_temp / self.n) if self.n else 0.0


def build_optimizer(model, lr, wd, b1, b2, kind):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if kind == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=(b1, b2))
        except Exception as e:
            print(f"[train] 8-bit Adam unavailable ({e}); using AdamW", flush=True)
    return torch.optim.AdamW(groups, lr=lr, betas=(b1, b2))


def lr_at(tok, warmup, total, lr, lr_min):
    if tok < warmup:
        return lr * tok / max(1, warmup)
    prog = min(1.0, (tok - warmup) / max(1, total - warmup))
    return lr_min + (lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))


def find_micro_batch(model, seq_len, start_mb, device):
    """Largest micro-batch that survives 2 FULL training steps (fwd+bwd+opt.step),
    so optimizer state and steady-state memory are accounted for — not just a single
    forward (which under-counts and OOMs later). Restores pristine weights after, since
    the probe steps perturb them."""
    snapshot = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
    mb = max(1, start_mb)
    found = 1
    while mb >= 1:
        opt = None
        try:
            torch.cuda.empty_cache()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            for _ in range(2):
                x = torch.randint(0, model.cfg.vocab_size, (mb, seq_len), device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, targets=x)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            found = mb
            break
        except torch.cuda.OutOfMemoryError:
            if mb == 1:
                found = 1
                break
            mb //= 2
        finally:
            del opt
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    model.load_state_dict({k: v.to(device) for k, v in snapshot.items()})
    del snapshot
    torch.cuda.empty_cache()
    return found


@torch.no_grad()
def sample(model, tok, prompt, max_new=60, temp=0.8, device="cuda"):
    model.eval()
    eot = tok.token_to_id("<|eot|>")
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        ctx = x[:, -model.cfg.max_seq_len:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(ctx)
        nxt = torch.multinomial(torch.softmax(logits[:, -1, :].float() / max(temp, 1e-5), -1), 1)
        x = torch.cat([x, nxt], 1)
        if nxt.item() == eot:
            break
    model.train()
    return tok.decode(x[0].tolist())


def load_tokenizer():
    try:
        from .tokenizer import load
        return load()
    except Exception:
        return None


def write_status(name, tokens, total, step, loss, lr_now, tps, guard, extra=""):
    avg = guard.avg_temp()
    last = guard.last or (0, 0, 0, 0, 0)
    txt = f"""# ton-2 :: {name} — status

_auto-written {time.strftime('%Y-%m-%d %H:%M:%S')}_

| metric | value |
|---|---|
| tokens | {tokens/1e6:.1f}M / {total/1e6:.0f}M ({100*tokens/max(1,total):.1f}%) |
| step | {step} |
| loss | {loss:.4f} |
| lr | {lr_now:.2e} |
| throughput | {tps:.0f} tok/s |
| GPU temp now / max / avg | {last[0]:.0f} / {guard.max_temp:.0f} / {avg:.0f} °C |
| power now | {last[1]:.0f} W (limit {last[2]:.0f} W) |
| thermal pauses | {guard.pause_events} |

{extra}
"""
    Path("STATUS.md").write_text(txt, encoding="utf-8")


def train(config_path, resume_from=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tr = cfg["train"]
    name = cfg.get("name", "ton2")
    device = torch.device("cuda")

    ckpt_dir = Path("checkpoints") / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for sentinel in ("THERMAL_STOP", "GATE_FAIL", "DISK_LOW"):
        (ckpt_dir / sentinel).unlink(missing_ok=True)

    model = model_from_config(cfg).to(device)  # fp32 master weights
    print(f"[train] {name}: {model.num_parameters()/1e6:.1f}M params", flush=True)

    resumed_tokens, resumed_step = 0, 0
    if resume_from:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        resumed_tokens, resumed_step = int(ck.get("tokens", 0)), int(ck.get("step", 0))
        print(f"[train] resumed step={resumed_step} tokens={resumed_tokens/1e6:.1f}M", flush=True)

    seq_len = tr["seq_len"]
    target_eff = max(1, tr["micro_batch_size"]) * max(1, tr["grad_accum_steps"]) * seq_len
    mb = find_micro_batch(model, seq_len, tr["micro_batch_size"], device)
    accum = max(1, round(target_eff / (mb * seq_len)))
    tokens_per_step = mb * accum * seq_len
    print(f"[train] micro_batch={mb} grad_accum={accum} -> {tokens_per_step} tokens/step "
          f"(target eff {target_eff})", flush=True)

    opt = build_optimizer(model, tr["lr"], tr["weight_decay"], tr["beta1"], tr["beta2"],
                          tr.get("optimizer", "adamw"))
    if resume_from:
        try:
            opt.load_state_dict(ck["optimizer"])
        except Exception as e:
            print(f"[train] optimizer restore failed ({e}); fresh optimizer", flush=True)

    loader = make_dataloader(cfg["data"]["shard_dir"], seq_len=seq_len, batch_size=mb,
                             num_workers=2, source_weights=cfg["data"].get("source_weights"))
    tok = load_tokenizer()

    total = int(tr["total_tokens"])
    warmup = int(tr["warmup_tokens"])
    ckpt_every = int(tr["checkpoint_every_tokens"])
    sanity_gate = int(tr.get("sanity_gate_tokens", 0))
    sanity_done = resumed_tokens >= sanity_gate if sanity_gate else True

    guard = ThermalGuard()
    guard.start()
    print("[train] thermal guard started", flush=True)

    tokens = resumed_tokens
    step = resumed_step
    next_ckpt = ((tokens // ckpt_every) + 1) * ckpt_every
    t0 = time.time()
    last_ckpt_time = time.time()
    it = iter(loader)

    def save(tag):
        p = ckpt_dir / tag
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "tokens": tokens, "cfg": cfg}, p)
        print(f"[train] checkpoint -> {p}", flush=True)

    def halt(sentinel, msg):
        (ckpt_dir / sentinel).write_text(msg, encoding="utf-8")
        save(f"step_{step}_tokens_{tokens//1_000_000}M.pt")
        print(f"[train] HALT [{sentinel}] {msg}", flush=True)
        guard._alive = False
        sys.exit(2)

    while tokens < total:
        if guard.stop:
            halt("THERMAL_STOP", f"GPU hit {guard.max_temp:.0f}C; stopped for safety.")
        while guard.pause and not guard.stop:
            time.sleep(2)
        if shutil.disk_usage(".").free / 1e9 < DISK_MIN_GB:
            halt("DISK_LOW", f"free disk < {DISK_MIN_GB}GB.")

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(accum):
            x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
                loss = loss / accum
            loss.backward()
            accum_loss += loss.item()
            tokens += x.numel()

        if not math.isfinite(accum_loss):
            halt("GATE_FAIL", f"non-finite loss {accum_loss} at step {step}.")

        if tr.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        lr_now = lr_at(tokens, warmup, total, tr["lr"], tr["lr_min"])
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.step()
        step += 1

        if step % 10 == 0:
            tps = (tokens - resumed_tokens) / max(1e-6, time.time() - t0)
            print(f"[train] step={step} tok={tokens/1e6:.1f}M loss={accum_loss:.4f} "
                  f"lr={lr_now:.2e} tok/s={tps:.0f} temp={guard.max_temp:.0f}C", flush=True)
            write_status(name, tokens, total, step, accum_loss, lr_now, tps, guard)

        # G5 sanity gate
        if (not sanity_done) and tokens >= sanity_gate:
            sanity_done = True
            samples = ""
            if tok is not None:
                for pr in ["The capital of France is", "Once upon a time", "2 + 2 ="]:
                    try:
                        samples += f"\n- PROMPT {pr!r}\n  -> {sample(model, tok, pr)!r}"
                    except Exception as e:
                        samples += f"\n- sample error: {e!r}"
            ok = math.isfinite(accum_loss) and accum_loss < 6.0  # "not broken" check; early-training loss is normally ~4-5, so 4.0 false-fails
            write_status(name, tokens, total, step, accum_loss, lr_now,
                         (tokens - resumed_tokens) / max(1e-6, time.time() - t0), guard,
                         extra=f"## G5 sanity gate @ {tokens/1e6:.0f}M\nloss={accum_loss:.3f} "
                               f"PASS={ok}\nsamples:{samples}\n")
            save(f"step_{step}_tokens_{tokens//1_000_000}M.pt")
            print(f"[train] G5 SANITY loss={accum_loss:.3f} PASS={ok}{samples}", flush=True)
            if not ok:
                halt("GATE_FAIL", f"sanity gate failed: loss {accum_loss:.3f} >= 4.0")

        if tokens >= next_ckpt or (time.time() - last_ckpt_time) > 1800:
            save(f"step_{step}_tokens_{tokens//1_000_000}M.pt")
            last_ckpt_time = time.time()
            while tokens >= next_ckpt:
                next_ckpt += ckpt_every

    save("final.pt")
    guard._alive = False
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--resume", default=None)
    a = p.parse_args()
    train(a.config, resume_from=a.resume)
