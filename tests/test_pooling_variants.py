"""Synthetic checks for the final pooling variants (CPU, no training).

Run from the repository root: python tests/test_pooling_variants.py
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pooling_variants as pv  # noqa: E402
from Models.AbsolutePositionalEncoding import tAPE  # noqa: E402
from Models.Attention import Attention_Rel_Scl  # noqa: E402

EMB, HEADS, CHANNELS, LENGTH, CLASSES, BATCH = 16, 8, 6, 100, 4, 3


def config(length=LENGTH):
    return {
        "Net_Type": ["C-T"], "emb_size": EMB, "num_heads": HEADS, "dim_ff": 256,
        "dropout": 0.01, "Data_shape": (BATCH, CHANNELS, length),
        "num_labels": CLASSES, "Fix_pos_encode": "tAPE", "Rel_pos_encode": "eRPE",
    }


def build(variant, seed=0):
    torch.manual_seed(seed)
    return pv.model_factory(config(), variant).eval()


def test_layers():
    average, avg_max = build("average"), build("avg_max")
    assert type(average.gap) is nn.AdaptiveAvgPool1d
    assert type(avg_max.gap) is pv.AvgMaxPool
    assert avg_max.gap.beta.shape == () and avg_max.gap.beta.item() == 0.0
    for model in (average, avg_max):
        assert tuple(model.embed_layer[0].kernel_size) == (1, 8)
        assert type(model.Fix_Position) is tAPE
        assert type(model.attention_layer) is Attention_Rel_Scl
        assert model.out.in_features == EMB
    params = dict(avg_max.named_parameters())
    assert pv.BETA_KEY in params and params[pv.BETA_KEY].requires_grad
    assert pv.trainable_parameters(avg_max) == pv.trainable_parameters(average) + 1


def test_beta_zero_matches_average():
    torch.manual_seed(1)
    x = torch.randn(BATCH, EMB, LENGTH)
    assert torch.equal(pv.AvgMaxPool()(x), nn.AdaptiveAvgPool1d(1)(x))

    average, avg_max = build("average"), build("avg_max")
    shared = {k: v for k, v in avg_max.state_dict().items() if k != pv.BETA_KEY}
    assert shared.keys() == average.state_dict().keys()
    for key, value in average.state_dict().items():
        assert torch.equal(value, shared[key]), key
    series = torch.randn(BATCH, CHANNELS, LENGTH)
    with torch.no_grad():
        assert torch.equal(pv.pooled_features(average, series),
                           pv.pooled_features(avg_max, series))
        assert torch.equal(average(series), avg_max(series))


def test_nonzero_beta_adds_max():
    torch.manual_seed(2)
    x = torch.randn(BATCH, EMB, LENGTH)
    pool = pv.AvgMaxPool()
    with torch.no_grad():
        pool.beta.fill_(0.5)
    expected = x.mean(-1, keepdim=True) + torch.tanh(torch.tensor(0.5)) * x.amax(-1, keepdim=True)
    assert torch.allclose(pool(x), expected, atol=1e-6)


def test_beta_receives_gradient():
    model = build("avg_max").train()
    series = torch.randn(BATCH, CHANNELS, LENGTH)
    labels = torch.tensor([0, 1, 2])
    nn.functional.cross_entropy(model(series), labels).backward()
    gradient = model.gap.beta.grad
    assert gradient is not None and torch.isfinite(gradient) and gradient.abs() > 0


def test_classifier_input_shape():
    series = torch.randn(BATCH, CHANNELS, LENGTH)
    shapes = {}
    for variant in pv.POOLINGS:
        model = build(variant)
        with torch.no_grad():
            shapes[variant] = tuple(pv.pooled_features(model, series).shape)
            assert model(series).shape == (BATCH, CLASSES)
    assert shapes["average"] == shapes["avg_max"] == (BATCH, EMB)


def test_random_state_unchanged():
    torch.manual_seed(0)
    pv.model_factory(config(), "average")
    after_average = torch.rand(3)
    torch.manual_seed(0)
    pv.model_factory(config(), "avg_max")
    assert torch.equal(after_average, torch.rand(3))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
