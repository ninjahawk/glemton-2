"""ton-2 model — dense decoder-only transformer, Llama-style.

RMSNorm (pre-norm), SwiGLU FFN, grouped-query attention (GQA), RoPE, tied
input/output embeddings. Plain causal attention via ``scaled_dot_product_attention``
(short context, full attention — no sliding window; that was unused in Glemton-1).

Ported and cleaned from Glemton-1's verified model.py. Adds GPT-2-style scaled
init on the residual output projections (o_proj, ffn.down) for training stability
at depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 24000
    d_model: int = 768
    n_layers: int = 24
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(x.dtype) * self.weight


def precompute_rope_cache(head_dim, max_seq_len, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, cfg.d_model)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Ton2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight
        self._rope_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-path output projections.
        scale = (2 * cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope(self, T: int, device, dtype):
        if (
            self._rope_cache is None
            or self._rope_cache[0].size(0) < T
            or self._rope_cache[0].device != device
        ):
            self._rope_cache = precompute_rope_cache(
                self.cfg.head_dim, max(T, self.cfg.max_seq_len), self.cfg.rope_theta, device, dtype
            )
        cos, sin = self._rope_cache
        return cos[:T], sin[:T]

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        x = self.tok_embed(input_ids)
        cos, sin = self._get_rope(T, x.device, x.dtype)
        for block in self.layers:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    def num_parameters(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            n += p.numel()
        return n


def model_from_config(cfg_dict: dict) -> Ton2:
    m = cfg_dict["model"]
    cfg = ModelConfig(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        n_kv_heads=m["n_kv_heads"],
        ffn_dim=m["ffn_dim"],
        max_seq_len=m["max_seq_len"],
        rope_theta=float(m["rope_theta"]),
        norm_eps=float(m["norm_eps"]),
        tie_embeddings=bool(m.get("tie_embeddings", True)),
    )
    return Ton2(cfg)
