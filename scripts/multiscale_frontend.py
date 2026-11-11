"""Temporal front-end variants for ConvTran, built on the pinned upstream code.

The upstream submodule is not modified. Both variants use tAPE + eRPE and are
built by the upstream ``model_factory``:

- ``original``: the upstream model as released. Its first temporal
  convolution is ``Conv2d(1, 4 * emb_size, kernel_size=[1, 8],
  padding='same')`` with bias (64 filters for ``emb_size=16``).
- ``multiscale``: only that convolution is replaced by
  :class:`MultiScaleTemporalConv`, three parallel ``Conv2d`` branches with
  kernels (1, 3), (1, 7) and (1, 15), stride 1, ``padding='same'`` and bias,
  whose outputs are concatenated along the filter axis. The upstream
  ``BatchNorm2d`` and ``GELU`` that follow it, the channel-mixing second
  convolution and the rest of the model are kept.
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
# (filters, temporal kernel) per branch, for emb_size = 16 (64 filters in all).
FRONTENDS = {
    "original": ((64, 8),),
    "multiscale": ((16, 3), (32, 7), (16, 15)),
}


def frontend_tag(variant):
    """Short identifier of a front-end configuration, used in run paths."""
    return "-".join(f"k{kernel}x{filters}" for filters, kernel in FRONTENDS[variant])


class MultiScaleTemporalConv(nn.Module):
    """Parallel temporal convolutions concatenated along the filter axis.

    Input (batch, 1, channels, length); output (batch, sum(filters), channels,
    length). Each branch convolves along time only, as the upstream layer does.
    """

    def __init__(self, branches, bias=True):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(1, filters, kernel_size=[1, kernel], stride=1,
                      padding="same", bias=bias)
            for filters, kernel in branches
        )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


def model_factory(config, variant, upstream_factory=upstream_model.model_factory):
    """Build ConvTran with upstream code and the selected temporal front-end.

    For ``multiscale`` the upstream model is built first, so every other layer
    is initialized exactly as in ``original`` for the same seed. The branches
    are then initialized inside ``torch.random.fork_rng``, so the random state
    used later for data shuffling and dropout is also the same as in
    ``original``.
    """
    model = upstream_factory({**config, **POSITIONAL_ENCODING})
    if variant == "original":
        return model
    if variant != "multiscale":
        raise ValueError(f"Unknown front-end variant: {variant}")
    first = model.embed_layer[0]
    branches = FRONTENDS[variant]
    if sum(filters for filters, _ in branches) != first.out_channels:
        raise ValueError(
            f"Multiscale branches need {first.out_channels} filters in total "
            f"(emb_size = {config['emb_size']})."
        )
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        replacement = MultiScaleTemporalConv(branches, bias=first.bias is not None)
    model.embed_layer[0] = replacement
    return model


def trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
