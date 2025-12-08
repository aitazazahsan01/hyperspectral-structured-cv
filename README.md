# Hyperspectral Structured-Representation CV Benchmark

**Research domain:** Structured vs. unstructured representations for hyperspectral image classification under limited labeled data

**Status:** Fully built and run end-to-end on real data. Every number in this README comes from an actual completed training run, not a projection — see [Results](#results).

## What this is

A small, honest benchmark testing the core claim behind posting **51807** (*Learning from Structured Representations in Hyperspectral Imaging*, Lakehead, Saad Ahmed): that treating hyperspectral pixels as a plain per-pixel signal loses information a structured (spatial-neighborhood) representation can capture — **especially where labeled data is limited.**

Two models are trained on the **Indian Pines** benchmark (the exact dataset the posting's sister posting names) and compared at three labeled-data budgets:

- **`PixelMLP`** — sees only a single pixel's spectral signature. The "plain, unstructured" baseline.
- **`PatchCNN3D`** — sees a 9×9 spatial neighborhood around the same pixel, as a 3D conv over (spectral, height, width). The "structured" model.

Both models see the **exact same spectral preprocessing** (per-band standardization + PCA to 30 components) — the only difference between them is whether they also get spatial context. That's deliberate: it isolates "does structure help" as the one variable under test.

## The finding, and the honest twist in it

**Headline result:** the structured model beats the baseline at every labeled-data budget, and its advantage *does* grow as labeled data shrinks — exactly what the posting's framing predicts.

**But** — while building this, I ran into a well-documented methodological trap in HSI benchmarking (Nalepa et al., 2019, *"Validating Hyperspectral Image Segmentation,"* IEEE GRSL) and decided to test for it rather than just cite it: **the standard way almost every tutorial splits Indian Pines (random per-pixel splitting) lets test-pixel patches overlap spatially with training pixels a few rows away.** That leaks information into the patch-based model specifically, since it's the one actually consuming that neighborhood. The pixel-only baseline can't benefit from it the same way.

So this repo runs the same ablation under **two split strategies** and reports both, rather than picking the one with the more impressive number:

1. **Random split** (what most tutorials do) — pixels assigned to train/test independently at random, stratified by class.
2. **Spatial-block split** (the fairness check) — the image is divided into 15×15 tiles, and whole tiles are assigned to train or test, so neighboring pixels mostly stay on the same side of the split.

## Results

All numbers are mean test accuracy over 2 random seeds, 15 training epochs, on a fixed 30%-of-labeled-pixels held-out test set (held out identically across all budgets within a split).

### Random split (standard practice)

| Label budget | PixelMLP (baseline) | PatchCNN3D (structured) | Structured advantage |
|---|---|---|---|
| 5%  | 51.6% | 86.9% | **+35.3 pts** |
| 20% | 66.7% | 96.3% | **+29.6 pts** |
| 50% | 75.0% | 99.3% | **+24.2 pts** |

### Spatial-block split (fairness check)

| Label budget | PixelMLP (baseline) | PatchCNN3D (structured) | Structured advantage |
|---|---|---|---|
| 5%  | 56.0% | 67.5% | **+11.5 pts** |
| 20% | 59.8% | 70.3% | **+10.4 pts** |
| 50% | 65.7% | 73.7% | **+8.0 pts** |

![Accuracy vs. label budget, random split](figures/budget_ablation_random.png)
![Accuracy vs. label budget, spatial split](figures/budget_ablation_spatial.png)

### What this actually shows

1. **The core claim survives the fairness check.** Under *both* splits, the structured model wins at every budget, and its margin over the baseline shrinks monotonically as labeled data grows (5% → 20% → 50%). That trend — not just the single-budget comparison — is the real evidence for "structure helps more when labels are scarce."
2. **But the random split overstates the effect by roughly 3x.** The structured model's advantage is +24 to +35 points under the random split, and only +8 to +12 points under the spatial split. The random split isn't wrong so much as it's answering a slightly different, easier question ("how well does the model do when it gets to see nearby training pixels indirectly") than the one the posting actually asks ("how well does structure generalize to genuinely new regions").
3. **Absolute accuracy drops much more for the structured model when leakage is removed** (patch model: 96.3% → 70.3% at 20% budget, a 26-point drop) **than for the baseline** (66.7% → 59.8%, a 7-point drop). That asymmetry makes sense: the patch model is the one directly consuming spatial neighbors, so it's the one with something to lose when neighbors are no longer guaranteed to share a train/test split.

Reported honestly, both directions: the posting's hypothesis holds, and the effect size in most tutorials citing this dataset is probably inflated. Full raw numbers (both seeds, not just the mean) are in `results/budget_ablation_random.json` and `results/budget_ablation_spatial.json`.

## Data

**Indian Pines** — 145×145 pixels, 200 spectral bands (after removing water-absorption bands), 16 land-cover classes, 10,249 labeled pixels total. Heavily class-imbalanced (20 pixels for "Oats" vs. 2,455 for "Soybean-mintill" — see `figures/class_balance.png`).

The dataset's canonical host (ehu.eus) blocks non-browser/datacenter traffic (403), so `src/data.py` pulls byte-identical mirrors from a public GitHub repo instead, and verifies the exact expected shape and class distribution on load (`tests/test_data.py::test_load_real_data_matches_known_indian_pines_shape`).

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Data | Indian Pines (`.mat`, scipy.io) | Standard, small, well-documented HSI benchmark; the exact dataset named in the target posting's sister posting |
| Preprocessing | Per-band standardization + PCA (30 components) | Standard HSI practice (e.g. HybridSN); keeps both models' spectral input identical so only spatial context differs |
| Baseline model | PyTorch MLP over a single pixel's spectral vector | The "unstructured" reference point |
| Structured model | PyTorch 3D-CNN over spatial-spectral patches | Captures spatial neighborhood + spectral depth jointly |
| Splitting | Random-stratified (standard) + spatial-block (fairness check) | Tests whether the reported effect survives removing spatial leakage — see [Results](#results) |
| Evaluation | Accuracy vs. label-budget ablation, 2 seeds/budget | Directly tests the posting's "especially where labeled data is limited" claim, not just a single accuracy number |
| Testing | `pytest`, 15 tests | Verifies preprocessing, splitting, and model shapes independent of any specific training run |

## Repo structure

```
src/
  data.py       download, standardize, PCA, patch extraction, both split strategies, budget subsetting
  models.py     PixelMLP (baseline), PatchCNN3D (structured)
  train.py      shared train/eval loop, used identically by both models
experiments/
  eda.py                    class balance + spectral signature figures
  run_budget_ablation.py    the main experiment: trains both models across budgets/seeds, saves results + plot
tests/
  test_data.py    preprocessing/splitting correctness (synthetic data, fast) + one real-data shape check
  test_models.py  forward-pass shape and gradient-flow checks for both models
results/          saved ablation JSON (both splits, all budgets, all seeds)
figures/          class balance, spectral signatures, both accuracy-vs-budget plots
```

## How to run

```bash
pip install -r requirements.txt

# Sanity-check preprocessing, splitting, and model logic (fast, no download needed
# except for one integration test that gracefully skips if offline)
python -m pytest tests/ -v

# EDA: class balance + spectral signature figures -> figures/
python experiments/eda.py

# The main experiment (downloads Indian Pines automatically on first run)
python experiments/run_budget_ablation.py --split random  --budgets 0.05,0.2,0.5 --seeds 0,1
python experiments/run_budget_ablation.py --split spatial --budgets 0.05,0.2,0.5 --seeds 0,1
```

Runs entirely on CPU in well under an hour total for both splits (each (budget, seed) combination takes roughly 1–9 minutes depending on training-set size at 15 epochs) — no GPU required, consistent with the posting's "GPU light" scope. A GPU will be used automatically via CUDA if available (`src/train.py` accepts a `device` argument).

## What would extend this (not done here, scope-boxed on purpose)

- **More seeds** (2 were used here to keep the full two-split, three-budget matrix runnable in under an hour on CPU; 5+ would tighten the error bars in the plots).
- **A GNN over superpixels** instead of a fixed patch window, which is the fuller version of "structured representation" the posting gestures at — `PatchCNN3D` is the simpler, faster-to-implement version of that same idea, chosen deliberately to fit a ~1.5 week prep timeline.
- **Buffer zones between spatial-split tiles** — the current spatial split reduces train/test patch overlap but doesn't fully eliminate it at tile boundaries; a margin equal to the patch radius around each tile edge would close that gap completely.
- This workflow (unfamiliar structured data → simple baseline → structured model → labeled-data-budget ablation, checked under two split strategies) transfers directly to the other two related postings: swap in a point-cloud dataset (e.g. ModelNet10) and PointNet for **52212**, or UAV multispectral imagery for **52116**.

## Application pitch line

*"I benchmarked a per-pixel baseline against a patch-based structured model on Indian Pines across three labeled-data budgets, and found the structured model's advantage does grow as labels shrink — but I also checked whether the standard random-pixel train/test split (used by most tutorials on this dataset) was leaking spatial information into that result, by re-running the whole ablation under a spatial-block split instead. It was: the random split overstates the structured model's advantage by roughly 3x, though the core trend survives under the fairer split too. Repo here: [link]."*
