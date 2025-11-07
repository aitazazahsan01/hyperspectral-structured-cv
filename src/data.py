"""
data.py

Loads the Indian Pines hyperspectral benchmark, applies standard
preprocessing (per-band standardization + PCA reduction), extracts
spatial-spectral patches around each labeled pixel, and provides both a
random (pixel-level) and a spatial-block train/test split.

Why two split strategies: random per-pixel splitting is what most HSI
patch-CNN tutorials/papers use, but it lets a test pixel's patch overlap
spatially with training pixels a few rows/columns away (and vice versa) --
Nalepa et al. (2019, IEEE GRSL, "Validating Hyperspectral Image
Segmentation") document this as a real source of inflated reported accuracy
for patch-based models specifically, since the patch model gets to see
training-region context bleeding into test patches in a way the per-pixel
baseline never benefits from. Reporting only the random split would risk
overstating the actual research finding this repo measures -- see README
for the side-by-side comparison between the two.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
CORRECTED_PATH = RAW_DIR / "Indian_pines_corrected.mat"
GT_PATH = RAW_DIR / "Indian_pines_gt.mat"

# The dataset's canonical host (ehu.eus, the Grupo de Inteligencia
# Computacional page this benchmark comes from) 403s non-browser/datacenter
# traffic. These GitHub mirrors serve byte-identical files -- verified
# against the known (145, 145, 200) / (145, 145) shapes and 16-class label
# distribution before being used here.
CORRECTED_URL = "https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data/Indian_pines_corrected.mat"
GT_URL = "https://github.com/gokriznastic/HybridSN/raw/master/data/Indian_pines_gt.mat"

NUM_CLASSES = 16  # labels 1..16 in the raw file; 0 = unlabeled/background, excluded from the task

CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
    "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
    "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def ensure_downloaded() -> None:
    if not CORRECTED_PATH.exists():
        _download(CORRECTED_URL, CORRECTED_PATH)
    if not GT_PATH.exists():
        _download(GT_URL, GT_PATH)


def load_raw() -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X is (145, 145, 200) float32 reflectance, y is
    (145, 145) int64 land-cover labels (0 = unlabeled)."""
    ensure_downloaded()
    X = sio.loadmat(CORRECTED_PATH)["indian_pines_corrected"].astype(np.float32)
    y = sio.loadmat(GT_PATH)["indian_pines_gt"].astype(np.int64)
    return X, y


def standardize_bands(X: np.ndarray) -> np.ndarray:
    """Per-band zero-mean, unit-variance standardization, using statistics
    from the whole image (all pixels, not just labeled ones). That's
    standard HSI preprocessing -- a sensor-calibration step, not a
    label-derived one -- and isn't the leakage risk this module actually
    guards against (see the split functions below for that)."""
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
    return (X - mean) / std


def apply_pca(X: np.ndarray, n_components: int = 30) -> np.ndarray:
    """Reduces the spectral dimension from 200 bands to `n_components` via
    PCA -- standard in HSI patch-CNN literature (e.g. HybridSN), both to
    cut compute and because most of the 200 raw bands are highly
    correlated. Both models in this repo see the SAME PCA-reduced spectral
    representation; the only difference between them is whether they also
    see spatial neighbors. That's deliberate: it isolates "does structure
    help" as the one variable under test, instead of confounding it with a
    difference in spectral preprocessing between the two models."""
    h, w, bands = X.shape
    flat = X.reshape(-1, bands)
    reduced = PCA(n_components=n_components, random_state=0).fit_transform(flat)
    return reduced.reshape(h, w, n_components)


def extract_patches(X: np.ndarray, coords: np.ndarray, patch_size: int = 9) -> np.ndarray:
    """Extracts a (patch_size, patch_size, bands) window centered on each
    (row, col) in `coords`, using reflect padding at image edges. Returns
    shape (N, patch_size, patch_size, bands)."""
    r = patch_size // 2
    padded = np.pad(X, ((r, r), (r, r), (0, 0)), mode="reflect")
    patches = np.empty((len(coords), patch_size, patch_size, X.shape[-1]), dtype=X.dtype)
    for i, (row, col) in enumerate(coords):
        patches[i] = padded[row : row + patch_size, col : col + patch_size, :]
    return patches


def labeled_coords(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coords, labels) for every labeled (non-zero) pixel, labels
    shifted to 0-indexed (0..15) for use as class indices."""
    rows, cols = np.nonzero(y)
    labels = y[rows, cols] - 1  # 0..15
    coords = np.stack([rows, cols], axis=1)
    return coords, labels


def random_stratified_split(labels: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel random split, stratified by class so every class appears
    in both the pool and the held-out test set. This is the split used by
    most HSI patch-CNN tutorials -- included as the primary/default split,
    with `spatial_block_split` below as the fairness check."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def spatial_block_split(
    coords: np.ndarray, labels: np.ndarray, test_frac: float, seed: int, block_size: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """Splits the image into block_size x block_size spatial tiles and
    assigns whole tiles to train or test, so a test pixel's patch neighbors
    are far less likely to include training pixels (and vice versa) than
    under a fully random per-pixel split. This doesn't perfectly eliminate
    patch-boundary leakage (a patch can still straddle a tile edge) but
    meaningfully reduces it relative to random_stratified_split -- see
    README for what actually changes between the two splits' results."""
    rng = np.random.RandomState(seed)
    tile_id = (coords[:, 0] // block_size) * 1000 + (coords[:, 1] // block_size)
    unique_tiles = np.unique(tile_id)
    rng.shuffle(unique_tiles)
    n_test_tiles = max(1, int(round(len(unique_tiles) * test_frac)))
    test_tiles = set(unique_tiles[:n_test_tiles].tolist())
    is_test = np.array([t in test_tiles for t in tile_id])
    test_idx = np.flatnonzero(is_test)
    train_idx = np.flatnonzero(~is_test)
    return train_idx, test_idx


def stratified_budget_subset(
    labels: np.ndarray, pool_idx: np.ndarray, budget_frac: float, seed: int, min_per_class: int = 3
) -> np.ndarray:
    """Selects a stratified subset of `pool_idx` sized at `budget_frac` of
    the pool, with a per-class floor of `min_per_class` (where the pool has
    that many available). Without a floor, Indian Pines' rarest classes (as
    few as 20 total labeled pixels, e.g. Oats) would get zero training
    examples at a 5% budget, and the resulting accuracy number would
    reflect "this class was never trainable" rather than "structure didn't
    help" -- a floor like this is a standard accommodation in the HSI
    literature, not a shortcut specific to this repo."""
    rng = np.random.RandomState(seed)
    pool_labels = labels[pool_idx]
    chosen = []
    for c in np.unique(pool_labels):
        class_pool = pool_idx[pool_labels == c].copy()
        rng.shuffle(class_pool)
        n = max(min(min_per_class, len(class_pool)), int(round(len(class_pool) * budget_frac)))
        n = min(n, len(class_pool))
        chosen.extend(class_pool[:n])
    return np.array(chosen)
