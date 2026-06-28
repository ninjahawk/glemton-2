"""G0 environment gate: verify PyTorch actually computes on the Blackwell GPU.

The classic RTX 50-series/Windows failure is a "no kernel image is available"
error on the first CUDA op even though `cuda.is_available()` is True. This script
forces real fp32 + bf16 compute and an autograd backward, and checks numerics.

Exit 0 = PASS, 1 = FAIL.
Run: .venv\\Scripts\\python.exe scripts\\check_env.py
"""
from __future__ import annotations

import sys


def main() -> int:
    import torch

    print("torch", torch.__version__)
    if not torch.cuda.is_available():
        print("GATE_FAIL: CUDA not available")
        return 1
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))

    torch.manual_seed(0)
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    ref = a @ b
    got = (a.cuda() @ b.cuda()).cpu()
    fp32 = torch.allclose(ref, got, rtol=1e-3, atol=1e-3)
    print("fp32_allclose", fp32)

    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    bf16 = bool(torch.isfinite(x @ x).all())
    print("bf16_finite", bf16)

    w = torch.randn(256, 256, device="cuda", requires_grad=True)
    (w @ w).sum().backward()
    bwd = bool(torch.isfinite(w.grad).all())
    print("backward_finite", bwd)

    if fp32 and bf16 and bwd:
        print("GATE_PASS")
        return 0
    print("GATE_FAIL: numerics")
    return 1


if __name__ == "__main__":
    sys.exit(main())
