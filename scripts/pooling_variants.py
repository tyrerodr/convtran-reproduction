"""Final temporal pooling variants for ConvTran, built on the pinned upstream code.

The upstream submodule is not modified. Both variants use the original
front-end with tAPE + eRPE and are built by the upstream ``model_factory``.
Upstream ``ConvTran.forward`` permutes the Transformer output to
(batch, emb_size, length), applies ``self.gap = nn.AdaptiveAvgPool1d(1)``,
flattens and applies the linear classifier ``self.out``.

- ``average``: the upstream model as released (global average pooling).
- ``avg_max``: ``self.gap`` is replaced by :class:`AvgMaxPool`, which returns
  ``avg + tanh(beta) * max`` over the same temporal axis, with a single
  trainable scalar ``beta`` initialized to zero. The output shape, the
  flatten and the classifier are unchanged.
"""

import sys
from pathlib import Path

import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream" / "ConvTran"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from Models import model as upstream_model  # noqa: E402

POSITIONAL_ENCODING = {"Fix_pos_encode": "tAPE", "Rel_pos_encode": "eRPE"}
POOLINGS = {
    "average": "avg",
    "avg_max": "avg-tanhbeta-max-beta0",
}
BETA_KEY = "gap.beta"


def pooling_tag(variant):
    """Short identifier of a pooling configuration, used in run paths."""
    return POOLINGS[variant]


class AvgMaxPool(nn.Module):
    """``avg + tanh(beta) * max`` over the last axis, output length 1.

    The average uses ``nn.AdaptiveAvgPool1d(1)`` as upstream does, so with
    ``beta = 0`` the output equals the upstream pooling exactly.
    """

    def __init__(self):
        super().__init__()
        self.average = nn.AdaptiveAvgPool1d(1)
        self.maximum = nn.AdaptiveMaxPool1d(1)
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return self.average(x) + torch.tanh(self.beta) * self.maximum(x)


def model_factory(config, variant, upstream_factory=upstream_model.model_factory):
    """Build ConvTran with upstream code and the selected final pooling.

    ``AvgMaxPool`` draws no random numbers, so for the same seed every layer
    and the later shuffling and dropout match the ``average`` variant.
    """
    model = upstream_factory({**config, **POSITIONAL_ENCODING})
    if variant == "average":
        return model
    if variant != "avg_max":
        raise ValueError(f"Unknown pooling variant: {variant}")
    if type(model.gap) is not nn.AdaptiveAvgPool1d:
        raise TypeError("Expected the upstream ConvTran global average pooling.")
    model.gap = AvgMaxPool()
    return model


def pooled_features(model, x):
    """Features given to the classifier ``model.out``, shape (batch, emb_size)."""
    captured = {}
    handle = model.out.register_forward_pre_hook(
        lambda module, inputs: captured.setdefault("features", inputs[0])
    )
    try:
        model(x)
    finally:
        handle.remove()
    return captured["features"]


def trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
