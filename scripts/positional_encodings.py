"""Positional-encoding variants for ConvTran, built on the pinned upstream code.

The upstream submodule is not modified. Variants are selected through the
upstream configuration keys ``Fix_pos_encode`` and ``Rel_pos_encode``:

- ``tape_erpe``: tAPE + eRPE, the released ConvTran configuration.
- ``sinusoidal``: upstream ``AbsolutePositionalEncoding`` ("Sin"), eRPE off,
  so ConvTran uses the upstream dense ``Attention``.
- ``rope``: no absolute encoding and no eRPE; ConvTran's dense ``Attention``
  is replaced by :class:`RoPEAttention`, which rotates Q and K per position.
"""

import sys
from pathlib import Path

import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream" / "ConvTran"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from Models import model as upstream_model  # noqa: E402
from Models.Attention import Attention  # noqa: E402

VARIANTS = {
    "tape_erpe": {"Fix_pos_encode": "tAPE", "Rel_pos_encode": "eRPE"},
    "sinusoidal": {"Fix_pos_encode": "Sin", "Rel_pos_encode": "None"},
    "rope": {"Fix_pos_encode": "None", "Rel_pos_encode": "RoPE"},
}
ROPE_BASE = 10000.0


def rope_angles(seq_len, head_dim, device=None, dtype=torch.float32):
    """Return cos and sin tables of shape (seq_len, head_dim // 2).

    Pair i of each head rotates by ``position * ROPE_BASE ** (-2i / head_dim)``.
    """
    inverse_frequency = ROPE_BASE ** (
        -torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate adjacent coordinate pairs of x (batch, heads, seq_len, head_dim)."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class RoPEAttention(Attention):
    """Upstream dense attention with rotary position embedding on Q and K.

    Projections, the ``emb_size ** -0.5`` scale, softmax, V, and the output
    LayerNorm are those of upstream ``Attention``; only Q and K are rotated,
    after the per-head split and before the dot product.
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__(emb_size, num_heads, dropout)
        if emb_size % num_heads:
            raise ValueError("emb_size must be divisible by num_heads")
        self.head_dim = emb_size // num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # q, k, v shape = (batch_size, num_heads, seq_len, head_dim)

        cos, sin = rope_angles(seq_len, self.head_dim, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(out)


def model_factory(config, upstream_factory=upstream_model.model_factory):
    """Build ConvTran with upstream code; RoPE swaps in RoPEAttention.

    For ``Rel_pos_encode == "RoPE"`` the model is built as the upstream
    "None" configuration (dense ``Attention``) with that class temporarily
    replaced, so every other layer is constructed exactly as upstream does.
    """
    if config["Rel_pos_encode"] != "RoPE":
        return upstream_factory(config)
    original = upstream_model.Attention
    upstream_model.Attention = RoPEAttention
    try:
        return upstream_factory({**config, "Rel_pos_encode": "None"})
    finally:
        upstream_model.Attention = original
