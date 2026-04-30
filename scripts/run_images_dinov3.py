from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from baselines.knn import KNN
from baselines.lof import LOF
from baselines.ICL import ICL
from baselines.DTE import DTECategorical, DTEInverseGamma
from baselines.MSM.MSM import MSM
from baselines.DDAE import DDAE
from baselines.SLAD.slad import SLAD
from methods.DSM import KDSM, DSM


# ---------------------------------------------------------------------------
# Model registry — edit this dict to add / remove models.
# Keys encode the sigma as the last "_<float>" suffix (same convention as run.py).
# ---------------------------------------------------------------------------
def build_model_dict() -> dict:
    model_dict = {}
    model_dict['KNN'] = KNN
    model_dict['LOF'] = LOF
    model_dict['ICL'] = ICL
    model_dict['DTE'] = DTECategorical
    model_dict['DSM'] = DSM
    model_dict['K-DSM'] = KDSM
    model_dict['MSM'] = MSM
    model_dict['DDAE'] = DDAE
    model_dict['SLAD'] = SLAD
    return model_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate anomaly detection models on VISA/MVTec DINOv3 embeddings."
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["visa", "mvtech", "all"],
        default="all",
        help="Which image dataset(s) to evaluate on (default: all).",
    )
    p.add_argument(
        "--visa-root",
        type=str,
        default=None,
        help="VISA root directory.",
    )
    p.add_argument(
        "--visa-prefix",
        type=str,
        default="visa_dinov3",
        help="Embedding filename prefix for VISA (from extract script).",
    )
    p.add_argument(
        "--mvtech-root",
        type=str,
        default=None,
        help="MVTec root directory.",
    )
    p.add_argument(
        "--mvtech-prefix",
        type=str,
        default="mvtech_dinov3",
        help="Embedding filename prefix for MVTec (from extract script).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--results-dir",
        type=str,
        default="./results/images/",
        help="Directory to save CSV result files.",
    )
    p.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Disable standardization of embeddings per class.",
    )
    p.set_defaults(standardize=True)
    p.add_argument(
        "--override",
        action="store_true",
        help="Re-run experiments even if results already exist in the CSVs.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_root(user_root: str | None, dataset_name: str) -> Path:
    if not user_root:
        raise ValueError(f"Provide --{dataset_name}-root to specify the dataset directory.")
    root = Path(user_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def load_embeddings(root: Path, prefix: str) -> dict[str, np.ndarray]:
    files = {
        "embeddings": root / f"{prefix}_embeddings.npy",
        "paths": root / f"{prefix}_paths.npy",
        "splits": root / f"{prefix}_splits.npy",
        "labels": root / f"{prefix}_labels.npy",
    }
    missing = [str(v) for v in files.values() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing embedding files (run the extract script first):\n  "
            + "\n  ".join(missing)
        )
    return {
        "embeddings": np.load(files["embeddings"], mmap_mode="r"),
        "paths": np.load(files["paths"], allow_pickle=False),
        "splits": np.load(files["splits"], allow_pickle=False),
        "labels": np.load(files["labels"], allow_pickle=False).astype(np.int64),
    }


def load_visa_meta_index(root: Path) -> dict[str, tuple[str, int, str]] | None:
    """Load VISA meta.json, which gives explicit split/label/class-name per image."""
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return None
    import json
    with meta_path.open("r") as f:
        meta = json.load(f)
    out: dict[str, tuple[str, int, str]] = {}
    for split in ("train", "test"):
        for _cls, items in meta.get(split, {}).items():
            for it in items:
                rel = str(it["img_path"]).lstrip("/")
                label = int(it.get("anomaly", 0))
                cls_name = (
                    str(it.get("cls_name") or _cls or "").strip()
                    or rel.split("/", 1)[0]
                )
                out[rel] = (split, label, cls_name)
    return out


def get_class_names(paths: np.ndarray, meta_index) -> np.ndarray:
    """Infer per-sample class name: prefer meta.json, fall back to first path segment."""
    cls_names = []
    for p in paths:
        if meta_index is not None and str(p) in meta_index:
            cls_names.append(meta_index[str(p)][2])
        else:
            cls_names.append(str(p).split("/", 1)[0] if "/" in str(p) else "")
    return np.asarray(cls_names, dtype=np.str_)


def fix_splits_from_meta(
    paths: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
    meta_index,
) -> tuple[np.ndarray, np.ndarray]:
    """For VISA: fall back to meta.json when the .npy splits look empty."""
    if meta_index is None:
        return splits, labels
    train_ok = (splits == "train").any() and ((splits == "train") & (labels == 0)).any()
    test_ok = (splits == "test").any()
    if train_ok and test_ok:
        return splits, labels
    split_list, label_list = [], []
    for p in paths:
        sp, lab, _ = meta_index.get(str(p), ("unknown", 0, ""))
        split_list.append(sp)
        label_list.append(lab)
    return np.asarray(split_list, dtype=np.str_), np.asarray(label_list, dtype=np.int64)


def low_density_anomalies(scores: np.ndarray, num_anomalies: int) -> np.ndarray:
    """Select the top-num_anomalies highest scores as predicted anomalies."""
    if num_anomalies <= 0:
        return np.zeros(len(scores), dtype=np.int64)
    k = min(num_anomalies, len(scores))
    preds = np.zeros(len(scores), dtype=np.int64)
    preds[np.argpartition(scores, -k)[-k:]] = 1
    return preds


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
def run_dataset(
    *,
    dataset_name: str,
    root: Path,
    prefix: str,
    model_dict: dict,
    seed: int,
    standardize: bool,
    override: bool,
    df_aucroc: pd.DataFrame,
    df_aucpr: pd.DataFrame,
    df_f1: pd.DataFrame,
    df_train: pd.DataFrame,
    df_inference: pd.DataFrame,
    csv_paths: dict[str, Path],
) -> None:
    data = load_embeddings(root, prefix)
    embeddings = data["embeddings"]
    paths = data["paths"]
    splits = data["splits"]
    labels = data["labels"]

    meta_index = load_visa_meta_index(root) if dataset_name == "visa" else None
    splits, labels = fix_splits_from_meta(paths, splits, labels, meta_index)
    cls_names = get_class_names(paths, meta_index)

    unique_classes = sorted({c for c in cls_names.tolist() if c})
    if not unique_classes:
        raise RuntimeError(f"[{dataset_name}] No class names found in embedding paths.")

    for cls in unique_classes:
        cls_mask = cls_names == cls
        cls_train_idx = np.where(cls_mask & (splits == "train") & (labels == 0))[0]
        cls_test_idx = np.where(cls_mask & (splits == "test"))[0]

        if len(cls_train_idx) == 0 or len(cls_test_idx) == 0:
            print(
                f"[{dataset_name}/{cls}] skipped "
                f"(train_good={len(cls_train_idx)}, test={len(cls_test_idx)})"
            )
            continue

        X_train = np.asarray(embeddings[cls_train_idx], dtype=np.float32)
        X_test = np.asarray(embeddings[cls_test_idx], dtype=np.float32)
        y_test = labels[cls_test_idx]

        if standardize:
            mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        row = f"{dataset_name}/{cls}"

        for name, clf_cls in model_dict.items():
            if not override:
                if (
                    row in df_aucroc.index
                    and name in df_aucroc.columns
                    and not pd.isna(df_aucroc.loc[row, name])
                ):
                    print(f"  [{row}] {name}: skipping (already computed)")
                    continue

            clf = clf_cls(seed=seed, model_name=name)

            t0 = time.time()
            clf.fit(X_train, np.zeros(len(X_train), dtype=np.int64))
            time_fit = time.time() - t0

            t0 = time.time()
            scores = clf.predict_score(X_test)
            time_inference = time.time() - t0

            scores = np.where(np.isnan(scores), 0.0, scores)
            aucroc = skm.roc_auc_score(y_test, scores)
            aucpr = skm.average_precision_score(y_test, scores)
            preds = low_density_anomalies(scores, int((y_test == 1).sum()))
            f1 = skm.f1_score(y_test, preds)

            print(
                f"  [{row}] {name}: "
                f"train={len(X_train)} test={len(X_test)} "
                f"anom={int((y_test == 1).sum())} "
                f"AUCROC={aucroc:.4f} AUCPR={aucpr:.4f} F1={f1:.4f} "
                f"(fit={time_fit:.1f}s infer={time_inference:.2f}s)"
            )

            df_aucroc.loc[row, name] = aucroc
            df_aucpr.loc[row, name] = aucpr
            df_f1.loc[row, name] = f1
            df_train.loc[row, name] = time_fit
            df_inference.loc[row, name] = time_inference

            # Save after every model result (matches run.py behaviour)
            df_aucroc.to_csv(csv_paths["aucroc"])
            df_aucpr.to_csv(csv_paths["aucpr"])
            df_f1.to_csv(csv_paths["f1"])
            df_train.to_csv(csv_paths["train"])
            df_inference.to_csv(csv_paths["inference"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    seed = args.seed

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = {
        "aucroc":    results_dir / f"{seed}_AUCROC.csv",
        "aucpr":     results_dir / f"{seed}_AUCPR.csv",
        "f1":        results_dir / f"{seed}_AUCF1.csv",
        "train":     results_dir / f"{seed}_TrainTime.csv",
        "inference": results_dir / f"{seed}_InferenceTime.csv",
    }

    df_aucroc    = _load_csv(csv_paths["aucroc"])
    df_aucpr     = _load_csv(csv_paths["aucpr"])
    df_f1        = _load_csv(csv_paths["f1"])
    df_train     = _load_csv(csv_paths["train"])
    df_inference = _load_csv(csv_paths["inference"])

    model_dict = build_model_dict()

    datasets_to_run: list[tuple[str, str | None, str]] = []
    if args.dataset in ("visa", "all"):
        datasets_to_run.append(("visa", args.visa_root, args.visa_prefix))
    if args.dataset in ("mvtech", "all"):
        datasets_to_run.append(("mvtech", args.mvtech_root, args.mvtech_prefix))

    for dataset_name, user_root, prefix in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*60}")
        root = resolve_root(user_root, dataset_name)
        run_dataset(
            dataset_name=dataset_name,
            root=root,
            prefix=prefix,
            model_dict=model_dict,
            seed=seed,
            standardize=args.standardize,
            override=args.override,
            df_aucroc=df_aucroc,
            df_aucpr=df_aucpr,
            df_f1=df_f1,
            df_train=df_train,
            df_inference=df_inference,
            csv_paths=csv_paths,
        )

    # Final summary
    print(f"\n{'='*60}")
    print("Summary (mean per dataset)")
    print(f"{'='*60}")
    dataset_labels = {
        "visa":   [i for i in df_aucroc.index if str(i).startswith("visa/")],
        "mvtech": [i for i in df_aucroc.index if str(i).startswith("mvtech/")],
    }
    for df, metric in [(df_aucroc, "AUCROC"), (df_aucpr, "AUCPR"), (df_f1, "F1")]:
        if not df.empty:
            print(f"\n{metric}:")
            for ds_name, rows in dataset_labels.items():
                subset = df.loc[[r for r in rows if r in df.index]]
                if subset.empty:
                    continue
                means = subset.mean(numeric_only=True)
                print(f"  {ds_name.upper()}:")
                for col, val in means.items():
                    print(f"    {col}: {val:.4f}")


if __name__ == "__main__":
    main()
