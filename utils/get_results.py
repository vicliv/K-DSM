"""Aggregate experiment results and export tables/plots."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAR_HEIGHT = 0.6
BAR_Y_STEP = 0.9
BAR_FIG_HEIGHT_PER_ROW = 0.55

# -------------------- Config --------------------
# Map raw model column names to display names (tables/plots).
MODEL_NAME_MAP: Dict[str, str] = {
}

# Set to a list to include only specific models. Use None to keep all.
MODEL_INCLUDE: List[str] | None = None

# Set to a list to exclude specific datasets (after aggregation). Use None to keep all.
DATASET_EXCLUDE: List[str] | None = None

# Dataset groups to average across variations.
DATASET_GROUPS: Dict[str, Iterable[str]] = {
    "agnews": ("agnews_",),
    "CIFAR10": ("CIFAR10_",),
    "MVTec": ("MVTec-AD_", "MVTec_", "mvtec"),
    "MNIST-C": ("MNIST-C_", "mnist-c"),
    "FashionMNIST": ("FashionMNIST_", "fashionmnist_"),
    "20news": ("20news_",),
    "SVHN": ("SVHN_",),
}

METRICS = {
    "f1": "AUCF1",
    "pr": "AUCPR",
    "roc": "AUCROC",
}

# -------------------- Helpers --------------------

def _match_group(name: str, prefixes: Iterable[str]) -> bool:
    lower = name.lower()
    for p in prefixes:
        if lower.startswith(p.lower()):
            return True
    return False


def aggregate_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Average predefined dataset variations into single rows."""
    df = df.copy()
    used = set()
    new_rows = {}
    for group, prefixes in DATASET_GROUPS.items():
        rows = [idx for idx in df.index if _match_group(idx, prefixes)]
        if rows:
            used.update(rows)
            new_rows[group] = df.loc[rows].mean(axis=0)
    if used:
        df = df.drop(index=list(used))
    if new_rows:
        df = pd.concat([df, pd.DataFrame.from_dict(new_rows, orient="index")])
    if DATASET_EXCLUDE is not None:
        df = df.drop(index=[d for d in DATASET_EXCLUDE if d in df.index], errors="ignore")
    return df


def sort_datasets(names: Iterable[str]) -> List[str]:
    def key(n: str) -> Tuple[int, int, str]:
        m = re.match(r"^(\d+)_", n)
        if m:
            return (0, int(m.group(1)), n)
        return (1, math.inf, n.lower())

    return sorted(names, key=key)


def display_name(name: str) -> str:
    m = re.match(r"^(\d+)_", name)
    if m:
        return name[m.end():]
    return name


def bold_best_row(values: pd.Series, precision: int = 3) -> List[str]:
    best = values.max()
    formatted = []
    for v in values:
        if pd.isna(v):
            formatted.append("-")
            continue
        s = f"{v:.{precision}f}"
        if v == best:
            s = f"\\textbf{{{s}}}"
        formatted.append(s)
    return formatted


def sem_from_group(std: pd.DataFrame, count: pd.DataFrame) -> pd.DataFrame:
    sem = std / np.sqrt(count)
    return sem


def read_seed_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if MODEL_INCLUDE is not None:
        df = df[[c for c in df.columns if c in MODEL_INCLUDE]]
    if MODEL_NAME_MAP:
        df = df.rename(columns=MODEL_NAME_MAP)
    if MODEL_INCLUDE is not None:
        ordered_cols = [MODEL_NAME_MAP.get(c, c) for c in MODEL_INCLUDE if MODEL_NAME_MAP.get(c, c) in df.columns]
        df = df[ordered_cols]
    return df


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -------------------- Main logic --------------------

def process_image_results(base_dir: Path, out_dir: Path) -> None:
    """Summarise results/images CSVs for VISA and MVTec separately. No plots."""
    tables_dir = out_dir / "tables"
    ensure_dir(tables_dir)

    # Collect all seed files for each metric
    for metric_key, metric_name in METRICS.items():
        pattern = f"*_{metric_name}.csv"
        seed_files = sorted(base_dir.glob(pattern))
        if not seed_files:
            print(f"No files for {metric_name} in {base_dir}")
            continue

        # Build per-seed DataFrames
        seed_dfs: dict[str, pd.DataFrame] = {}
        for path in seed_files:
            seed = path.name.split("_")[0]
            df = read_seed_df(path)
            seed_dfs[seed] = df

        if not seed_dfs:
            continue

        combined = pd.concat(seed_dfs, names=["seed", "row"])
        mean_all = combined.groupby("row").mean()
        std_all = combined.groupby("row").std(ddof=1)
        count_all = combined.groupby("row").count().replace(0, np.nan)
        sem_all = std_all / np.sqrt(count_all)

        # Collect per-dataset overall stats for the combined overview table
        overview_mean: dict[str, pd.Series] = {}
        overview_sem: dict[str, pd.Series] = {}

        for ds in ("visa", "mvtech"):
            prefix = f"{ds}/"
            rows = [r for r in mean_all.index if str(r).startswith(prefix)]
            if not rows:
                continue

            ds_mean = mean_all.loc[rows].copy()
            ds_sem = sem_all.loc[rows].copy()
            # Strip the "visa/" / "mvtech/" prefix for display
            ds_mean.index = [r[len(prefix):] for r in rows]
            ds_sem.index = ds_mean.index

            # Compute per-seed means across classes for bench-style mean/SEM
            seed_ds_means = []
            for seed, sdf in seed_dfs.items():
                seed_rows = [r for r in sdf.index if str(r).startswith(prefix)]
                if seed_rows:
                    seed_ds_means.append(sdf.loc[seed_rows].mean(axis=0))
            seed_means_df = pd.DataFrame(seed_ds_means) if seed_ds_means else None
            if seed_means_df is not None:
                bench_mean = seed_means_df.mean(axis=0)
                bench_sem = seed_means_df.std(axis=0, ddof=1) / np.sqrt(
                    seed_means_df.count(axis=0).replace(0, np.nan)
                )
            else:
                bench_mean = ds_mean.mean(numeric_only=True)
                bench_sem = pd.Series(float("nan"), index=bench_mean.index)

            overview_mean[ds.upper()] = bench_mean
            overview_sem[ds.upper()] = bench_sem

            # Print summary
            print(f"\n{'='*60}")
            print(f"  {ds.upper()} — {metric_key.upper()}")
            print(f"{'='*60}")
            print(ds_mean.round(4).to_string())
            print(f"\n  Overall mean ({ds.upper()}):")
            for col, val in bench_mean.items():
                print(f"    {col}: {val:.4f} ± {bench_sem[col]:.4f}")

            # Save per-class table CSV
            ds_mean.to_csv(tables_dir / f"{ds}_{metric_key}_per_class.csv")

            # Save overall stats CSV
            bench_df = pd.DataFrame({"mean": bench_mean, "sem": bench_sem})
            bench_df.to_csv(tables_dir / f"{ds}_{metric_key}_overall.csv")

            # LaTeX table (per-class, bold best per row, with SEM)
            formatted_rows = []
            for idx in ds_mean.index:
                row_mean = ds_mean.loc[idx]
                row_sem = ds_sem.loc[idx]
                best = row_mean.max()
                formatted = []
                for model in ds_mean.columns:
                    m = row_mean[model]
                    s = row_sem[model]
                    if pd.isna(m):
                        formatted.append("-")
                    else:
                        m_str = f"{m:.3f}"
                        s_str = f"{s:.3f}" if not pd.isna(s) else "-"
                        cell = f"{m_str} ({s_str})"
                        if m == best:
                            cell = f"\\textbf{{{cell}}}"
                        formatted.append(cell)
                formatted_rows.append(formatted)

            table_fmt = pd.DataFrame(formatted_rows, index=ds_mean.index, columns=ds_mean.columns)
            latex = table_fmt.to_latex(escape=False, column_format="l" + "c" * len(ds_mean.columns))

            # Build Mean row and inject before \bottomrule
            best_overall = bench_mean.max()
            mean_cells = []
            for model in ds_mean.columns:
                m = bench_mean.get(model, float("nan"))
                s = bench_sem.get(model, float("nan"))
                if pd.isna(m):
                    mean_cells.append("-")
                else:
                    m_str = f"{m:.3f}"
                    s_str = f"{s:.3f}" if not pd.isna(s) else "-"
                    cell = f"{m_str} ({s_str})"
                    if m == best_overall:
                        cell = f"\\textbf{{{cell}}}"
                    mean_cells.append(cell)
            mean_row = "Mean & " + " & ".join(mean_cells) + " \\\\"
            latex = latex.replace("\\bottomrule", "\\midrule\n" + mean_row + "\n\\bottomrule")

            (tables_dir / f"{ds}_{metric_key}.tex").write_text(latex)

        # Combined overview table: one row per dataset (VISA, MVTec)
        if overview_mean:
            all_models = list(next(iter(overview_mean.values())).index)
            ov_formatted_rows = []
            for ds_label, bm in overview_mean.items():
                bs = overview_sem[ds_label]
                best = bm.max()
                row_cells = []
                for model in all_models:
                    m = bm.get(model, float("nan"))
                    s = bs.get(model, float("nan"))
                    if pd.isna(m):
                        row_cells.append("-")
                    else:
                        m_str = f"{m:.3f}"
                        s_str = f"{s:.3f}" if not pd.isna(s) else "-"
                        cell = f"{m_str} ({s_str})"
                        if m == best:
                            cell = f"\\textbf{{{cell}}}"
                        row_cells.append(cell)
                ov_formatted_rows.append(row_cells)
            ov_table = pd.DataFrame(ov_formatted_rows, index=list(overview_mean.keys()), columns=all_models)
            ov_latex = ov_table.to_latex(escape=False, column_format="l" + "c" * len(all_models))
            (tables_dir / f"{metric_key}_overview.tex").write_text(ov_latex)

    # Process timing metrics for images
    for time_key in ("TrainTime", "InferenceTime"):
        seed_files = sorted(base_dir.glob(f"*_{time_key}.csv"))
        if not seed_files:
            print(f"No files for {time_key} in {base_dir}")
            continue

        seed_dfs_t: dict[str, pd.DataFrame] = {}
        for path in seed_files:
            seed = path.name.split("_")[0]
            df = read_seed_df(path)
            seed_dfs_t[seed] = df

        if not seed_dfs_t:
            continue

        combined_t = pd.concat(seed_dfs_t, names=["seed", "row"])
        mean_all_t = combined_t.groupby("row").mean()
        std_all_t = combined_t.groupby("row").std(ddof=1)
        count_all_t = combined_t.groupby("row").count().replace(0, np.nan)
        sem_all_t = std_all_t / np.sqrt(count_all_t)

        # Compute overall benchmark stats per dataset
        for ds in ("visa", "mvtech"):
            prefix = f"{ds}/"
            rows = [r for r in mean_all_t.index if str(r).startswith(prefix)]
            if not rows:
                continue

            ds_mean_t = mean_all_t.loc[rows].copy()
            ds_sem_t = sem_all_t.loc[rows].copy()

            # Compute per-seed means across classes for bench-style mean/SEM
            seed_ds_means_t = []
            for seed, sdf in seed_dfs_t.items():
                seed_rows = [r for r in sdf.index if str(r).startswith(prefix)]
                if seed_rows:
                    seed_ds_means_t.append(sdf.loc[seed_rows].mean(axis=0))
            seed_means_df_t = pd.DataFrame(seed_ds_means_t) if seed_ds_means_t else None
            if seed_means_df_t is not None:
                bench_mean_t = seed_means_df_t.mean(axis=0)
                bench_sem_t = seed_means_df_t.std(axis=0, ddof=1) / np.sqrt(
                    seed_means_df_t.count(axis=0).replace(0, np.nan)
                )
            else:
                bench_mean_t = ds_mean_t.mean(numeric_only=True)
                bench_sem_t = pd.Series(float("nan"), index=bench_mean_t.index)

            # Save benchmark CSV
            timing_summary = pd.DataFrame({"mean": bench_mean_t, "sem": bench_sem_t})
            timing_summary.to_csv(tables_dir / f"{ds}_{time_key.lower()}_benchmark.csv")

    print(f"\nImage results written to {tables_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results and export tables/plots.")
    parser.add_argument("mode", choices=["semi", "unsup", "images"], help="results folder to use")
    parser.add_argument("--out", default=None, help="output directory (default: results/<mode>)")
    args = parser.parse_args()

    _mode_label = {"semi": "semi-supervised", "unsup": "unsupervised"}.get(args.mode, args.mode)
    _metric_label = {"f1": "AUC-F1", "pr": "AUC-PR", "roc": "AUC-ROC"}

    if args.mode == "images":
        base_dir = Path("results") / "images"
        if not base_dir.exists():
            raise SystemExit(f"Results directory not found: {base_dir}")
        out_dir = Path(args.out) if args.out else base_dir
        process_image_results(base_dir, out_dir)
        print(f"Done. Outputs in {out_dir}")
        return

    base_dir = Path("results") / args.mode
    if not base_dir.exists():
        raise SystemExit(f"Results directory not found: {base_dir}")

    out_dir = Path(args.out) if args.out else base_dir
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    for metric_key, metric_name in METRICS.items():
        pattern = f"*_{metric_name}.csv"
        seed_files = sorted(base_dir.glob(pattern))
        if not seed_files:
            print(f"No files for {metric_name} in {base_dir}")
            continue

        seed_dfs = {}
        for path in seed_files:
            seed = path.name.split("_")[0]
            df = read_seed_df(path)
            df = aggregate_datasets(df)
            seed_dfs[seed] = df

        # Align datasets and models across seeds
        combined = pd.concat(seed_dfs, names=["seed", "dataset"])

        mean = combined.groupby("dataset").mean()
        std = combined.groupby("dataset").std(ddof=1)
        count = combined.groupby("dataset").count().replace(0, np.nan)
        sem = sem_from_group(std, count)

        ordered = sort_datasets(mean.index)
        mean = mean.loc[ordered]
        sem = sem.loc[ordered]

        # Save CSVs
        mean.to_csv(out_dir / f"{metric_key}_mean.csv")
        sem.to_csv(out_dir / f"{metric_key}_sem.csv")

        # Benchmark averages per model (mean over datasets per seed, then mean/sem over seeds)
        seed_means = []
        seed_labels = []
        for seed, df in seed_dfs.items():
            df = df.reindex(ordered)
            seed_means.append(df.mean(axis=0))
            seed_labels.append(seed)
        seed_means_df = pd.DataFrame(seed_means, index=seed_labels)

        bench_mean = seed_means_df.mean(axis=0)
        bench_std = seed_means_df.std(axis=0, ddof=1)
        bench_count = seed_means_df.count(axis=0).replace(0, np.nan)
        bench_sem = bench_std / np.sqrt(bench_count)

        summary = pd.DataFrame({"mean": bench_mean, "sem": bench_sem})
        summary.to_csv(out_dir / f"{metric_key}_benchmark.csv")

        # LaTeX table
        display_index = [display_name(n) for n in mean.index]
        table_mean = mean.copy()
        table_sem = sem.copy()
        table_mean.index = display_index
        table_sem.index = display_index

        formatted_rows = []
        for idx in table_mean.index:
            row_mean = table_mean.loc[idx]
            row_sem = table_sem.loc[idx]
            best = row_mean.max()
            formatted = []
            for model in table_mean.columns:
                m = row_mean[model]
                s = row_sem[model]
                if pd.isna(m):
                    formatted.append("-")
                    continue
                m_str = f"{m:.3f}"
                s_str = f"{s:.3f}" if not pd.isna(s) else "-"
                cell = f"{m_str} ({s_str})"
                if m == best:
                    cell = f"\\textbf{{{cell}}}"
                formatted.append(cell)
            formatted_rows.append(formatted)

        table_formatted = pd.DataFrame(formatted_rows, index=table_mean.index, columns=table_mean.columns)
        latex = table_formatted.to_latex(escape=False, column_format="l" + "c" * len(table_mean.columns))

        # Build Overall row and inject before \bottomrule
        best_overall = bench_mean.max()
        overall_cells = []
        for model in table_mean.columns:
            m = bench_mean.get(model, float("nan"))
            s = bench_sem.get(model, float("nan"))
            if pd.isna(m):
                overall_cells.append("-")
            else:
                m_str = f"{m:.3f}"
                s_str = f"{s:.3f}" if not pd.isna(s) else "-"
                cell = f"{m_str} ({s_str})"
                if m == best_overall:
                    cell = f"\\textbf{{{cell}}}"
                overall_cells.append(cell)
        overall_row = "Overall & " + " & ".join(overall_cells) + " \\\\"
        latex = latex.replace("\\bottomrule", "\\midrule\n" + overall_row + "\n\\bottomrule")

        caption = (
            f"Average {_metric_label[metric_key]} and standard deviations over five seeds"
            f" for the {_mode_label} setting on ADBench."
        )
        latex = (
            "\\begin{table}H]\n"
            "    \\centering\n"
            f"    \\caption{{{caption}}}\n"
            "    \\resizebox{\\textwidth}{!}{\n"
            + latex +
            "}\n"
            "\\end{table}\n"
        )
        (tables_dir / f"{metric_key}.tex").write_text(latex)

        print(f"\nBenchmark summary ({metric_key.upper()}):")
        for model in summary.index:
            m = summary.loc[model, "mean"]
            s = summary.loc[model, "sem"]
            print(f"  {model}: {m:.4f} ± {s:.4f}")

        # Plot
        _color_groups = [
            ({"K-DSM-EMA", "DSM-EMA", "DDAE-EMA", "DTE-EMA", "MSM-EMA"}, plt.cm.magma(0.92)),
            ({"K-DSM", "DSM", "MSM"},            plt.cm.magma(0.80)),
            ({"DTE", "DDAE", "SLAD", "ICL"},     plt.cm.magma(0.50)),
            ({"LOF", "KNN"},                      plt.cm.magma(0.22)),
        ]
        _model_color = {}
        for _models, _color in _color_groups:
            for _m in _models:
                _model_color[_m] = _color
        bar_colors = [_model_color.get(m, plt.cm.magma(0.5)) for m in bench_mean.index]

        #plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})

        fig, ax = plt.subplots(
            figsize=(6, max(3.5, BAR_FIG_HEIGHT_PER_ROW * len(bench_mean) * BAR_Y_STEP))
        )
        y = np.arange(len(bench_mean)) * BAR_Y_STEP
        ax.barh(y, bench_mean.values, xerr=bench_sem.values,
                color=bar_colors, edgecolor="white", linewidth=0.6,
                height=BAR_HEIGHT, capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "#555555"})
        ax.set_yticks(y)
        ax.set_yticklabels(bench_mean.index, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel(_metric_label[metric_key], fontsize=11)
        if metric_key in {"pr", "f1"}:
            ax.set_xlim(left=0.49)
        else:
            ax.set_xlim(left=0.75)
        ax.grid(axis="x", linestyle="--", alpha=0.35, color="grey")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric_key}_benchmark_avg.png", dpi=200, bbox_inches="tight")
        fig.savefig(plots_dir / f"{metric_key}_benchmark_avg.pdf", bbox_inches="tight")
        plt.close(fig)

        # plt.rcParams.update({"font.family": plt.rcParamsDefault["font.family"],
        #                       "font.serif":  plt.rcParamsDefault["font.serif"]})

    # ---- Timing plots ----
    _color_groups = [
        ({"K-DSM-EMA", "DSM-EMA", "DDAE-EMA", "DTE-EMA", "MSM-EMA"}, plt.cm.magma(0.92)),
        ({"K-DSM", "DSM", "MSM"},        plt.cm.magma(0.80)),
        ({"DTE", "DDAE", "SLAD", "ICL"}, plt.cm.magma(0.50)),
        ({"LOF", "KNN"},                  plt.cm.magma(0.22)),
    ]

    for time_key, time_label in [("TrainTime", "Training Time (s)"), ("InferenceTime", "Inference Time (s)")]:
        seed_files = sorted(base_dir.glob(f"*_{time_key}.csv"))
        if not seed_files:
            print(f"No files for {time_key} in {base_dir}")
            continue

        seed_dfs_t = {}
        for path in seed_files:
            seed = path.name.split("_")[0]
            df = read_seed_df(path)
            df = aggregate_datasets(df)
            seed_dfs_t[seed] = df

        combined_t = pd.concat(seed_dfs_t, names=["seed", "dataset"])
        mean_t = combined_t.groupby("dataset").mean()
        ordered_t = sort_datasets(mean_t.index)
        mean_t = mean_t.loc[ordered_t]

        seed_means_t = pd.DataFrame(
            [seed_dfs_t[s].reindex(ordered_t).mean(axis=0) for s in seed_dfs_t],
            index=list(seed_dfs_t.keys()),
        )
        bench_mean_t = seed_means_t.mean(axis=0)
        bench_sem_t  = seed_means_t.std(axis=0, ddof=1) / np.sqrt(seed_means_t.count(axis=0).replace(0, np.nan))

        _model_color = {m: c for models, c in _color_groups for m in models}
        bar_colors = [_model_color.get(m, plt.cm.magma(0.5)) for m in bench_mean_t.index]

        # Save timing benchmark CSV
        timing_summary = pd.DataFrame({"mean": bench_mean_t, "sem": bench_sem_t})
        timing_summary.to_csv(out_dir / f"{time_key.lower()}_benchmark.csv")

        fig, ax = plt.subplots(
            figsize=(6, max(3.5, BAR_FIG_HEIGHT_PER_ROW * len(bench_mean_t) * BAR_Y_STEP))
        )
        y = np.arange(len(bench_mean_t)) * BAR_Y_STEP
        ax.barh(y, bench_mean_t.values, xerr=bench_sem_t.values,
                color=bar_colors, edgecolor="white", linewidth=0.6,
                height=BAR_HEIGHT, capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "#555555"})
        ax.set_yticks(y)
        ax.set_yticklabels(bench_mean_t.index, fontsize=11)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(time_label, fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.35, color="grey")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{time_key.lower()}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {time_key} plot.")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
