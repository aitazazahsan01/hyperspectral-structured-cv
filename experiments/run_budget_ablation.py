"""
run_budget_ablation.py

The core experiment this repo exists to run: train the per-pixel baseline
(PixelMLP) and the patch-based structured model (PatchCNN3D) at several
labeled-data budgets, and see whether the structured model's advantage over
the baseline grows as labeled data shrinks -- which is the specific claim
posting 51807 makes about structured representations in hyperspectral
imaging, "especially where labeled data is limited."

Run it once per split strategy to see how much the reported gap depends on
that choice (see README for why -- random per-pixel splitting is known to
let patch-based models see spatially leaked context that the pixel baseline
never benefits from):

    python experiments/run_budget_ablation.py --split random
    python experiments/run_budget_ablation.py --split spatial
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    apply_pca,
    extract_patches,
    labeled_coords,
    load_raw,
    random_stratified_split,
    spatial_block_split,
    standardize_bands,
    stratified_budget_subset,
)
from src.models import PatchCNN3D, PixelMLP  # noqa: E402
from src.train import evaluate, make_loader, train_model  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def run(
    split_kind: str,
    budgets: list[float],
    seeds: list[int],
    patch_size: int,
    n_pca: int,
    epochs: int,
    test_frac: float,
) -> dict:
    print(f"Loading and preprocessing Indian Pines (split={split_kind})...")
    X, y = load_raw()
    X = standardize_bands(X)
    X_pca = apply_pca(X, n_components=n_pca)

    coords, labels = labeled_coords(y)
    print(f"{len(labels)} labeled pixels across {len(np.unique(labels))} classes.")

    print(f"Extracting {patch_size}x{patch_size} patches for all labeled pixels...")
    all_patches = extract_patches(X_pca, coords, patch_size=patch_size)  # (N, ps, ps, n_pca)
    all_pixels = X_pca[coords[:, 0], coords[:, 1], :]  # (N, n_pca)

    if split_kind == "random":
        pool_idx, test_idx = random_stratified_split(labels, test_frac=test_frac, seed=0)
    elif split_kind == "spatial":
        pool_idx, test_idx = spatial_block_split(coords, labels, test_frac=test_frac, seed=0)
    else:
        raise ValueError(f"Unknown split kind: {split_kind!r}")
    print(f"Pool: {len(pool_idx)} pixels, held-out test: {len(test_idx)} pixels.")

    # Fixed test set, reused across every budget/seed combination.
    test_patches = np.transpose(all_patches[test_idx], (0, 3, 1, 2))[:, None, :, :, :]
    test_pixels = all_pixels[test_idx]
    test_labels = labels[test_idx]
    pixel_test_loader = make_loader(test_pixels, test_labels, batch_size=128, shuffle=False)
    patch_test_loader = make_loader(test_patches, test_labels, batch_size=64, shuffle=False)

    results: dict = {"split": split_kind, "budgets": budgets, "pixel_mlp": {}, "patch_cnn3d": {}}

    for budget in budgets:
        pixel_accs, patch_accs = [], []
        for seed in seeds:
            train_idx = stratified_budget_subset(labels, pool_idx, budget, seed=seed)
            n_train = len(train_idx)

            torch.manual_seed(seed)
            train_loader = make_loader(all_pixels[train_idx], labels[train_idx], batch_size=32, shuffle=True)
            model = PixelMLP(n_bands=n_pca, num_classes=16)
            model = train_model(model, train_loader, epochs=epochs)
            pixel_accs.append(evaluate(model, pixel_test_loader))

            torch.manual_seed(seed)
            train_patches = np.transpose(all_patches[train_idx], (0, 3, 1, 2))[:, None, :, :, :]
            train_loader = make_loader(train_patches, labels[train_idx], batch_size=32, shuffle=True)
            model = PatchCNN3D(n_bands=n_pca, patch_size=patch_size, num_classes=16)
            model = train_model(model, train_loader, epochs=epochs)
            patch_accs.append(evaluate(model, patch_test_loader))

            print(
                f"  budget={budget:.0%} seed={seed} n_train={n_train} "
                f"pixel_acc={pixel_accs[-1]:.3f} patch_acc={patch_accs[-1]:.3f}"
            )

        results["pixel_mlp"][str(budget)] = {
            "mean": float(np.mean(pixel_accs)), "std": float(np.std(pixel_accs)), "runs": pixel_accs,
        }
        results["patch_cnn3d"][str(budget)] = {
            "mean": float(np.mean(patch_accs)), "std": float(np.std(patch_accs)), "runs": patch_accs,
        }

    return results


def plot_results(results: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = results["budgets"]
    pixel_means = [results["pixel_mlp"][str(b)]["mean"] for b in budgets]
    pixel_stds = [results["pixel_mlp"][str(b)]["std"] for b in budgets]
    patch_means = [results["patch_cnn3d"][str(b)]["mean"] for b in budgets]
    patch_stds = [results["patch_cnn3d"][str(b)]["std"] for b in budgets]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar([b * 100 for b in budgets], pixel_means, yerr=pixel_stds, marker="o", label="PixelMLP (baseline)")
    ax.errorbar([b * 100 for b in budgets], patch_means, yerr=patch_stds, marker="s", label="PatchCNN3D (structured)")
    ax.set_xlabel("Labeled data budget (% of training pool)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy vs. label budget -- {results['split']} split")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "spatial"], default="random")
    parser.add_argument("--budgets", type=str, default="0.05,0.2,0.5")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--pca", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--test-frac", type=float, default=0.3)
    args = parser.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    results = run(args.split, budgets, seeds, args.patch_size, args.pca, args.epochs, args.test_frac)

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / f"budget_ablation_{args.split}.json"
    out_json.write_text(json.dumps(results, indent=2))
    plot_results(results, FIGURES_DIR / f"budget_ablation_{args.split}.png")
    print(f"\nSaved {out_json} and figures/budget_ablation_{args.split}.png")


if __name__ == "__main__":
    main()
