"""
models.py

Two models sharing the same PCA-reduced spectral input representation (see
src/data.py apply_pca) so the only difference between them is whether they
see spatial neighbors -- that isolates the actual research question this
repo measures: does structure help, and does it help more as labeled data
shrinks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelMLP(nn.Module):
    """Baseline: sees only the PCA-reduced spectral vector of a single
    pixel, no spatial neighbors."""

    def __init__(self, n_bands: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bands, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_bands)
        return self.net(x)


class PatchCNN3D(nn.Module):
    """Structured: sees a (patch_size, patch_size, n_bands) spatial-spectral
    neighborhood around the pixel, processed with 3D convolutions over
    (spectral-depth, height, width) -- the "structure" being tested is
    literally the spatial neighborhood this model gets that PixelMLP does
    not."""

    def __init__(self, n_bands: int, patch_size: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
        )
        # Each conv uses a "valid" (unpadded) kernel along the spectral
        # depth axis, so depth shrinks by (kernel_depth - 1) per layer.
        depth_out = n_bands - (7 - 1) - (5 - 1) - (3 - 1)
        if depth_out < 1:
            raise ValueError(
                f"n_bands={n_bands} is too small for this conv stack "
                f"(needs at least {(7 - 1) + (5 - 1) + (3 - 1) + 1})"
            )
        flat_size = 32 * depth_out * patch_size * patch_size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_bands, patch_size, patch_size)
        return self.head(self.conv(x))
