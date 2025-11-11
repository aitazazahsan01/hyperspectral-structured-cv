"""
test_data.py

Sanity checks for src/data.py's preprocessing and splitting logic, using
small synthetic arrays so these run instantly and don't depend on the
Indian Pines download. test_load_real_data is the one exception -- a real
integration check against the actual dataset, skipped (not failed) if the
network isn't reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    apply_pca,
    extract_patches,
    labeled_coords,
    load_raw,
    random_stratified_split,
    spatial_block_split,
    standardize_bands,
    stratified_budget_subset,
)


def test_standardize_bands_zero_mean_unit_std():
    rng = np.random.RandomState(0)
    X = rng.normal(loc=50, scale=10, size=(10, 10, 5)).astype(np.float32)
    out = standardize_bands(X)
    flat = out.reshape(-1, 5)
    assert np.allclose(flat.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(flat.std(axis=0), 1, atol=1e-5)


def test_apply_pca_reduces_band_dimension():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 20, 50)).astype(np.float32)
    out = apply_pca(X, n_components=5)
    assert out.shape == (20, 20, 5)


def test_extract_patches_center_matches_source_pixel():
    X = np.arange(5 * 5 * 2).reshape(5, 5, 2).astype(np.float32)
    coords = np.array([[2, 2], [0, 0], [4, 4]])
    patches = extract_patches(X, coords, patch_size=3)
    assert patches.shape == (3, 3, 3, 2)
    # Center of each 3x3 patch (index [1,1]) must equal the source pixel
    for i, (r, c) in enumerate(coords):
        assert np.array_equal(patches[i, 1, 1, :], X[r, c, :])


def test_extract_patches_reflect_pads_at_edges_without_crashing():
    X = np.zeros((4, 4, 3), dtype=np.float32)
    coords = np.array([[0, 0], [3, 3]])
    patches = extract_patches(X, coords, patch_size=5)
    assert patches.shape == (2, 5, 5, 3)


def test_labeled_coords_excludes_background_and_shifts_labels():
    y = np.array([[0, 1], [2, 0]])
    coords, labels = labeled_coords(y)
    assert len(coords) == 2  # only the two non-zero pixels
    assert set(labels.tolist()) == {0, 1}  # shifted from {1, 2} to {0, 1}


def test_random_stratified_split_has_no_overlap_and_covers_all_classes():
    labels = np.repeat(np.arange(4), 20)  # 4 classes, 20 samples each
    train_idx, test_idx = random_stratified_split(labels, test_frac=0.3, seed=0)
    assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
    assert set(labels[train_idx].tolist()) == {0, 1, 2, 3}
    assert set(labels[test_idx].tolist()) == {0, 1, 2, 3}


def test_spatial_block_split_has_no_overlap():
    rng = np.random.RandomState(0)
    coords = np.array([[r, c] for r in range(30) for c in range(30)])
    labels = rng.randint(0, 3, size=len(coords))
    train_idx, test_idx = spatial_block_split(coords, labels, test_frac=0.3, seed=0, block_size=5)
    assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
    assert len(train_idx) + len(test_idx) == len(coords)


def test_spatial_block_split_test_pixels_are_spatially_clustered():
    """A basic check that the spatial split actually behaves differently
    from a random split: test pixels should form contiguous blocks, so the
    average distance from a test pixel to its nearest *other* test pixel
    should be small relative to image size."""
    rng = np.random.RandomState(1)
    coords = np.array([[r, c] for r in range(40) for c in range(40)])
    labels = rng.randint(0, 3, size=len(coords))
    _, test_idx = spatial_block_split(coords, labels, test_frac=0.2, seed=0, block_size=8)
    test_coords = coords[test_idx]
    # If tiles are respected, every test pixel should share its tile with
    # at least one other test pixel (tiles are 8x8=64 px, test_frac picks
    # whole tiles) -- i.e. no isolated singleton test pixels.
    tile_id = (test_coords[:, 0] // 8) * 1000 + (test_coords[:, 1] // 8)
    _, counts = np.unique(tile_id, return_counts=True)
    assert counts.min() > 1


def test_stratified_budget_subset_respects_floor_for_rare_classes():
    labels = np.concatenate([np.zeros(5, dtype=int), np.ones(200, dtype=int)])
    pool_idx = np.arange(len(labels))
    subset = stratified_budget_subset(labels, pool_idx, budget_frac=0.01, seed=0, min_per_class=3)
    subset_labels = labels[subset]
    assert (subset_labels == 0).sum() == 3  # floor applied to the 5-sample rare class
    assert (subset_labels == 1).sum() >= 3  # floor also applies, proportional would be ~2


def test_stratified_budget_subset_scales_with_budget_for_common_classes():
    labels = np.zeros(1000, dtype=int)
    pool_idx = np.arange(len(labels))
    small = stratified_budget_subset(labels, pool_idx, budget_frac=0.05, seed=0, min_per_class=3)
    large = stratified_budget_subset(labels, pool_idx, budget_frac=0.5, seed=0, min_per_class=3)
    assert len(small) < len(large)


def test_load_real_data_matches_known_indian_pines_shape():
    try:
        X, y = load_raw()
    except Exception as e:
        pytest.skip(f"Could not download/load real Indian Pines data: {e}")
    assert X.shape == (145, 145, 200)
    assert y.shape == (145, 145)
    assert set(np.unique(y).tolist()) == set(range(17))  # 0 (background) + 16 classes
