"""Synthetic checks for the temporal front-end variants (CPU, no training).

Run from the repository root: python tests/test_multiscale_frontend.py
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import multiscale_frontend as mf  # noqa: E402
from Models.AbsolutePositionalEncoding import tAPE  # noqa: E402
from Models.Attention import Attention_Rel_Scl  # noqa: E402

EMB, HEADS, CHANNELS, CLASSES, BATCH = 16, 8, 6, 4, 2
LENGTHS = (1, 7, 8, 100, 101)


def config(length):
    return {
        "Net_Type": ["C-T"], "emb_size": EMB, "num_heads": HEADS, "dim_ff": 256,
        "dropout": 0.01, "Data_shape": (BATCH, CHANNELS, length),
        "num_labels": CLASSES, "Fix_pos_encode": "tAPE", "Rel_pos_encode": "eRPE",
    }


def build(variant, length=100, seed=0):
    torch.manual_seed(seed)
    return mf.model_factory(config(length), variant).eval()


def test_layers():
    original, multiscale = build("original"), build("multiscale")
    first = original.embed_layer[0]
    assert type(first) is nn.Conv2d
    assert (first.in_channels, first.out_channels) == (1, 4 * EMB)
    assert tuple(first.kernel_size) == (1, 8) and first.bias is not None
    branches = multiscale.embed_layer[0].branches
    assert [(b.out_channels, tuple(b.kernel_size)) for b in branches] == [
        (16, (1, 3)), (32, (1, 7)), (16, (1, 15)),
    ]
    for branch in branches:
        assert branch.in_channels == 1 and tuple(branch.stride) == (1, 1)
        assert branch.bias is not None
    for model in (original, multiscale):
        assert type(model.embed_layer[1]) is nn.BatchNorm2d
        assert model.embed_layer[1].num_features == 4 * EMB
        assert type(model.embed_layer[2]) is nn.GELU
        assert type(model.Fix_Position) is tAPE
        assert type(model.attention_layer) is Attention_Rel_Scl


def test_shapes():
    for length in LENGTHS:
        x = torch.randn(BATCH, CHANNELS, length)
        stage = {}
        for variant in mf.FRONTENDS:
            model = build(variant, length)
            with torch.no_grad():
                first = model.embed_layer[0](x.unsqueeze(1))
                stage[variant] = model.embed_layer(x.unsqueeze(1))
                second = model.embed_layer2(stage[variant])
                assert first.shape == (BATCH, 4 * EMB, CHANNELS, length), variant
                assert second.shape == (BATCH, EMB, 1, length), variant
                assert model(x).shape == (BATCH, CLASSES), variant
        assert stage["original"].shape == stage["multiscale"].shape


def test_parameter_counts():
    original, multiscale = build("original"), build("multiscale")
    first, replacement = original.embed_layer[0], multiscale.embed_layer[0]
    assert first.weight.numel() == 512
    assert sum(b.weight.numel() for b in replacement.branches) == 512
    assert first.bias.numel() == 64
    assert sum(b.bias.numel() for b in replacement.branches) == 64
    assert mf.trainable_parameters(first) == mf.trainable_parameters(replacement) == 576
    assert mf.trainable_parameters(original) == mf.trainable_parameters(multiscale)


def test_rest_of_model_and_random_state_unchanged():
    torch.manual_seed(0)
    mf.model_factory(config(100), "original")
    after_original = torch.rand(3)
    torch.manual_seed(0)
    mf.model_factory(config(100), "multiscale")
    after_multiscale = torch.rand(3)
    assert torch.equal(after_original, after_multiscale)

    original = build("original").state_dict()
    multiscale = build("multiscale").state_dict()
    shared = [key for key in original if not key.startswith("embed_layer.0.")]
    assert shared == [k for k in multiscale if not k.startswith("embed_layer.0.")]
    for key in shared:
        assert torch.equal(original[key], multiscale[key]), key


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
