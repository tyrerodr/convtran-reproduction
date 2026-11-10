"""Synthetic checks for the positional-encoding variants (CPU, no training).

Run from the repository root: python tests/test_positional_encodings.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import positional_encodings as pe  # noqa: E402
from Models.AbsolutePositionalEncoding import (  # noqa: E402
    AbsolutePositionalEncoding, tAPE,
)
from Models.Attention import Attention, Attention_Rel_Scl  # noqa: E402

EMB, HEADS, CHANNELS, LENGTH, CLASSES = 16, 8, 6, 100, 4


def config(variant):
    return {
        "Net_Type": ["C-T"], "emb_size": EMB, "num_heads": HEADS, "dim_ff": 256,
        "dropout": 0.01, "Data_shape": (2, CHANNELS, LENGTH),
        "num_labels": CLASSES, **pe.VARIANTS[variant],
    }


def test_variant_selection():
    expected = {
        "tape_erpe": (tAPE, Attention_Rel_Scl),
        "sinusoidal": (AbsolutePositionalEncoding, Attention),
        "rope": (None, pe.RoPEAttention),
    }
    x = torch.randn(2, CHANNELS, LENGTH)
    for variant, (position_class, attention_class) in expected.items():
        torch.manual_seed(0)
        model = pe.model_factory(config(variant)).eval()
        if position_class is None:
            assert not hasattr(model, "Fix_Position"), variant
            assert model.Fix_pos_encode == "None", variant
        else:
            assert type(model.Fix_Position) is position_class, variant
        assert type(model.attention_layer) is attention_class, variant
        assert model(x).shape == (2, CLASSES), variant
    # The temporary class swap for RoPE must not leak into later builds.
    assert type(pe.model_factory(config("sinusoidal")).attention_layer) is Attention


def test_rope_dimensions():
    attention = pe.RoPEAttention(EMB, HEADS, dropout=0.01)
    assert attention.head_dim == 2
    cos, sin = pe.rope_angles(LENGTH, attention.head_dim)
    assert cos.shape == sin.shape == (LENGTH, 1)  # one pair per head
    # With one pair, the frequency is base ** 0 = 1: angle equals position.
    assert torch.allclose(torch.atan2(sin[3], cos[3]), torch.tensor([3.0]))


def test_rope_position_zero_and_rotation():
    torch.manual_seed(0)
    q = torch.randn(2, HEADS, LENGTH, EMB // HEADS)
    k = torch.randn(2, HEADS, LENGTH, EMB // HEADS)
    cos, sin = pe.rope_angles(LENGTH, EMB // HEADS)
    for x in (q, k):
        rotated = pe.apply_rope(x, cos, sin)
        assert torch.equal(rotated[:, :, 0], x[:, :, 0])  # position 0 kept
        position = 5
        pair, new = x[:, :, position], rotated[:, :, position]
        assert pair.norm(dim=-1).min() > 0
        assert not torch.allclose(new, pair)
        assert torch.allclose(new.norm(dim=-1), pair.norm(dim=-1), atol=1e-6)
        c, s = math.cos(position), math.sin(position)
        expected = torch.stack(
            (pair[..., 0] * c - pair[..., 1] * s, pair[..., 0] * s + pair[..., 1] * c),
            dim=-1,
        )
        assert torch.allclose(new, expected, atol=1e-6)


def test_rope_matches_dense_attention_without_rotation():
    torch.manual_seed(0)
    dense = Attention(EMB, HEADS, dropout=0.01).eval()
    rope = pe.RoPEAttention(EMB, HEADS, dropout=0.01).eval()
    rope.load_state_dict(dense.state_dict())
    x = torch.randn(2, LENGTH, EMB)
    original = pe.rope_angles
    pe.rope_angles = lambda n, d, device=None, dtype=torch.float32: (
        torch.ones(n, d // 2, dtype=dtype), torch.zeros(n, d // 2, dtype=dtype)
    )
    try:
        assert torch.allclose(rope(x), dense(x), atol=1e-6)
    finally:
        pe.rope_angles = original
    assert not torch.allclose(rope(x), dense(x), atol=1e-6)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
