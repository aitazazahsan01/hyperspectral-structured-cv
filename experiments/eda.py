"""
eda.py

Quick exploratory look at Indian Pines before any modeling: class balance
(it's heavily imbalanced -- from 20 pixels for Oats to 2455 for
Soybean-mintill) and a few classes' mean spectral signatures (a sanity
check that the bands actually separate classes, which is the entire
premise of spectral classification in the first place).

Run: python experiments/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASS_NAMES, labeled_coords, load_raw  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    X, y = load_raw()
    coords, labels = labeled_coords(y)

    FIGURES_DIR.mkdir(exist_ok=True)

    counts = np.bincount(labels, minlength=16)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(16), counts)
    ax.set_xticks(range(16))
    ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Labeled pixel count")
    ax.set_title(f"Indian Pines class balance ({len(labels)} labeled pixels total)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for c in [1, 4, 10, 13]:  # Corn-notill, Grass-pasture, Soybean-mintill, Woods
        mask = labels == c
        px = X[coords[mask, 0], coords[mask, 1], :]
        ax.plot(px.mean(axis=0), label=CLASS_NAMES[c])
    ax.set_xlabel("Spectral band index (raw, pre-PCA)")
    ax.set_ylabel("Mean reflectance")
    ax.set_title("Mean spectral signature by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spectral_signatures.png", dpi=150)
    plt.close(fig)

    print(f"Saved class_balance.png and spectral_signatures.png to {FIGURES_DIR}")
    print(f"Class counts: {dict(zip(CLASS_NAMES, counts.tolist()))}")


if __name__ == "__main__":
    main()
