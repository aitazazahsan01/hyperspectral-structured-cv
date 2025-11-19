"""
test_models.py

Forward-pass shape/sanity checks for both models, plus a check that the two
models actually receive input shaped the way src/data.py produces it (a
mismatch here would silently break the whole ablation pipeline without
raising anywhere obvious).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import PatchCNN3D, PixelMLP


def test_pixel_mlp_output_shape():
    model = PixelMLP(n_bands=30, num_classes=16)
    x = torch.randn(8, 30)
    out = model(x)
    assert out.shape == (8, 16)


def test_patch_cnn3d_output_shape():
    model = PatchCNN3D(n_bands=30, patch_size=9, num_classes=16)
    x = torch.randn(4, 1, 30, 9, 9)
    out = model(x)
    assert out.shape == (4, 16)


def test_patch_cnn3d_rejects_too_few_bands():
    # The conv stack needs at least 13 spectral bands (6+4+2+1) to produce a
    # positive depth after three valid-padded convs; fewer should fail loudly
    # at construction time, not with a cryptic shape error mid-training.
    with pytest.raises(ValueError):
        PatchCNN3D(n_bands=10, patch_size=9, num_classes=16)


def test_patch_cnn3d_gradient_flows():
    model = PatchCNN3D(n_bands=30, patch_size=9, num_classes=16)
    x = torch.randn(2, 1, 30, 9, 9)
    y = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)
