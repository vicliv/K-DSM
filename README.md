# K-DSM: Kurtosis-based Denoising Score Matching for Anomaly Detection

This repository contains the implementation of **K-DSM** and accompanying baselines for anomaly detection on tabular and image datasets, evaluated under semi-supervised and unsupervised settings.

## Requirements

Python 3.10 or later. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

### Installation

```bash
# Install uv
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install all dependencies from pyproject.toml
uv sync
```

All scripts must be run from the **repository root**.

> **Note on PyTorch**: The `torch` and `torchvision` versions in `pyproject.toml` specify minimum versions. If your CUDA version requires a specific PyTorch build, install it manually before running `uv sync`, or pin the exact wheel in `pyproject.toml`.

## Datasets

### Tabular (ADBench)

The [ADBench](https://github.com/Minqi824/ADBench) benchmark datasets must be downloaded before running tabular experiments:

```bash
uv run python utils/download.py
```

This downloads all ADBench datasets via the `jihulab` mirror into the `adbench` package's data directory.

## Running Tabular Experiments

### Semi-supervised Setting

Training uses only normal (in-distribution) samples; evaluation is on a held-out mixed test split.

```bash
uv run python scripts/run.py --setting semi --seed 42
```

### Unsupervised Setting

No labeled splits are used; the full dataset is treated as unlabeled.

```bash
uv run python scripts/run.py --setting unsup --seed 42
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--setting` | `semi` | Experimental setting: `semi` or `unsup` |
| `--seed` | `42` | Random seed |
| `--override` | `False` | Re-run experiments even if results already exist |

Results are saved as CSV files under `results/<setting>/` (e.g., `results/semi/42_AUCROC.csv`).

### Ablation Studies

```bash
# Semi-supervised: ablates sigma, tau, and kurtosis-scaling variants
uv run python scripts/run_ablations.py --setting semi --seed 42

# Unsupervised: ablates EMA filter percentile
uv run python scripts/run_ablations.py --setting unsup --seed 42
```

Results are saved under `ablations/<setting>/`.

### Image Datasets

Download the datasets manually before running image experiments:

- **MVTec AD**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **VISA**: https://github.com/amazon-science/spot-diff

Embeddings must be extracted with DINOv3 prior to evaluation (see [Extracting Image Embeddings](#extracting-image-embeddings) below).

## Extracting Image Embeddings

Both extraction scripts use the `facebook/dinov3-vitl16-pretrain-lvd1689m` model by default and save embeddings as `.npy` files alongside the dataset.

### MVTec AD

```bash
uv run python utils/extract_mvtech_dinov3_embeddings.py \
    --mvtech-root /path/to/mvtec \
    --output-prefix mvtech_dinov3 \
    --batch-size 32 \
    --use-fp16
```

### VISA

```bash
uv run python utils/extract_visa_dinov3_embeddings.py \
    --visa-root /path/to/visa \
    --output-prefix visa_dinov3 \
    --batch-size 32 \
    --use-fp16
```

When a `meta.json` file is present at the VISA root (as distributed by the dataset authors), it is used for explicit train/test splits and anomaly labels.

**Shared extraction arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model-name` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Hugging Face model ID |
| `--batch-size` | `32` | Inference batch size |
| `--num-workers` | `4` | DataLoader worker count |
| `--device` | auto (cuda if available) | `cpu`, `cuda`, `cuda:0`, … |
| `--output-prefix` | `mvtech_dinov3` / `visa_dinov3` | Prefix for output `.npy` filenames |
| `--use-fp16` | `False` | Run in float16 on CUDA |
| `--trust-remote-code` | `False` | Pass `trust_remote_code=True` to HF loaders |

Each script saves four files: `<prefix>_embeddings.npy`, `<prefix>_paths.npy`, `<prefix>_splits.npy`, `<prefix>_labels.npy`.

## Running Image Experiments

After extracting embeddings, run the image benchmark:

```bash
uv run python scripts/run_images_dinov3.py \
    --dataset all \
    --mvtech-root /path/to/mvtec \
    --visa-root /path/to/visa \
    --results-dir ./results/images/ \
    --seed 42
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `all` | `visa`, `mvtech`, or `all` |
| `--mvtech-root` | — | Path to MVTec root (required when evaluating MVTec) |
| `--visa-root` | — | Path to VISA root (required when evaluating VISA) |
| `--mvtech-prefix` | `mvtech_dinov3` | Embedding filename prefix (must match extraction) |
| `--visa-prefix` | `visa_dinov3` | Embedding filename prefix (must match extraction) |
| `--results-dir` | `./results/images/` | Output directory for CSV results |
| `--seed` | `42` | Random seed |
| `--no-standardize` | — | Disable per-class feature standardization |
| `--override` | `False` | Re-run even if results already exist |

Results are saved as CSV files under `--results-dir` (e.g., `results/images/42_AUCROC.csv`), with one row per `<dataset>/<class>` and one column per method.

## Aggregating Results

```bash
# Tabular ablation results
uv run python utils/ablation.py semi
uv run python utils/ablation.py unsup
```

This reads the CSV files from `ablations/` or `results/` and produces summary tables (LaTeX) and benchmark plots (mean ± SEM across seeds).

## Methods

| Method | Description |
|---|---|
| **K-DSM** | Kurtosis-based Denoising Score Matching (proposed) |
| **DSM** | Standard Denoising Score Matching |
| **K-DSM (EMA Teacher)** | K-DSM with EMA teacher filtering for unsupervised setting |
| **DSM (EMA Teacher)** | DSM with EMA teacher filtering for unsupervised setting |
| KNN | k-Nearest Neighbors |
| LOF | Local Outlier Factor |
| ICL | Instance-wise Contrastive Learning |
| DTE | Diffusion Time Estimation |
| DDAE | Diffusion Denoising Autoencoder |
| MSM | Multi-Scale Matching |
| SLAD | Subspace Learning for Anomaly Detection |
| MCD | Minimum Covariance Determinant (unsupervised only) |
