#!/usr/bin/env python3
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

# -------------------- Config --------------------
SIGMA_LEVELS = ["0.1", "0.3", "0.5", "0.7", "1.0"]
EMA_LEVELS = ["75", "80", "85", "90", "95"]

TAU_KDSMC_MODELS = [
    "K-DSMc_0.875_0.00001",
    "K-DSMc_0.560_0.00005",
    "K-DSMc_0.506_0.0001",
    "K-DSMc_0.326_0.001",
    "K-DSMc_0.203_0.005",
    "K-DSMc_0.151_0.01",
    "K-DSMc_0.035_0.05",
    "K-DSMc_0.012_0.07",
    #"K-DSMc_-0.012_0.1",
]
TAU_KDSM_GGD_MODELS = [
    #"K-DSMggd_0.000001",
    #"K-DSMggd_0.000005",
    "K-DSMggd_0.00001",
    "K-DSMggd_0.00005",
    "K-DSMggd_0.0001",
    "K-DSMggd_0.001",
    "K-DSMggd_0.005",
    "K-DSMggd_0.01",
    "K-DSMggd_0.05",
]

# Map raw model column names to display names (tables/plots).
MODEL_NAME_MAP: Dict[str, str] = {
    **{f"K-DSM_{s}": f"K-DSM_{s}" for s in SIGMA_LEVELS},
    **{f"DSM_{s}":   f"DSM_{s}"   for s in SIGMA_LEVELS},
    **{f"K-DSM-EMA_{s}": f"K-DSM-EMA_{s}" for s in EMA_LEVELS},
    **{f"DSM-EMA_{s}":   f"DSM-EMA_{s}"   for s in EMA_LEVELS},
}

# Set to a list to include only specific models. Use None to keep all.
MODEL_INCLUDE: List[str] | None = [
    *[f"K-DSM_{s}" for s in SIGMA_LEVELS],
    *[f"DSM_{s}"   for s in SIGMA_LEVELS],
    *[f"K-DSM-EMA_{s}" for s in EMA_LEVELS],
    *[f"DSM-EMA_{s}"   for s in EMA_LEVELS],
    *TAU_KDSMC_MODELS,
    *TAU_KDSM_GGD_MODELS,
]

MODEL_ORDER: List[str] = MODEL_INCLUDE

# Set to a list to exclude specific datasets (after aggregation). Use None to keep all.
DATASET_EXCLUDE: List[str] | None = None#["agnews", "CIFAR10", "MNIST-C", "FashionMNIST", "20news", "SVHN", "MVTec", "amazon", "imdb", "yelp"]

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

METRIC_TITLES = {
    "f1": "F1",
    "pr": "AUC-PR",
    "roc": "AUC-ROC",
}

# Seeds excluded from benchmark/plot aggregation.
PLOT_EXCLUDE_SEEDS = {"52"}

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


def seed_from_filename(path: Path) -> str:
    return path.name.split("_")[0]


def read_seed_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if MODEL_INCLUDE is not None:
        df = df[[c for c in df.columns if c in MODEL_INCLUDE]]
    if MODEL_NAME_MAP:
        df = df.rename(columns=MODEL_NAME_MAP)
    return df


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_base_dir(mode: str) -> Path:
    base = Path("ablations") / mode
    if base.exists():
        return base
    raise SystemExit(f"Results directory not found: {base}")


def get_main_ablation_family(columns: Iterable[str]) -> Tuple[List[str], str, str, str]:
    """Infer whether main ablation columns are sigma-based or EMA-based."""
    colset = set(columns)
    has_ema = any(f"K-DSM-EMA_{s}" in colset or f"DSM-EMA_{s}" in colset for s in EMA_LEVELS)
    if has_ema:
        return EMA_LEVELS, "K-DSM-EMA_{}", "DSM-EMA_{}", "Filter Percentile ($\\gamma$)"
    return SIGMA_LEVELS, "K-DSM_{}", "DSM_{}", r"Base sigma $\\sigma_0$"


def compute_summary(base_dir: Path) -> dict[str, pd.DataFrame]:
    """Load seed CSVs from base_dir and return benchmark summary per metric.

    Returns a dict mapping metric_key -> DataFrame with columns ['mean', 'sem'].
    """
    results: dict[str, pd.DataFrame] = {}
    for metric_key, metric_name in METRICS.items():
        seed_files = sorted(base_dir.glob(f"*_{metric_name}.csv"))
        if not seed_files:
            continue

        seed_dfs: dict[str, pd.DataFrame] = {}
        for path in seed_files:
            seed = seed_from_filename(path)
            if seed in PLOT_EXCLUDE_SEEDS:
                continue
            df = read_seed_df(path)
            if df.empty:
                continue
            df = aggregate_datasets(df)
            seed_dfs[seed] = df

        if not seed_dfs:
            continue

        combined = pd.concat(seed_dfs, names=["seed", "dataset"])
        mean_ds = combined.groupby("dataset").mean()
        ordered = sort_datasets(mean_ds.index)

        seed_means = []
        for seed, df in seed_dfs.items():
            seed_means.append(df.reindex(ordered).mean(axis=0))
        seed_means_df = pd.DataFrame(seed_means)

        bench_mean = seed_means_df.mean(axis=0)
        bench_std  = seed_means_df.std(axis=0, ddof=1)
        bench_count = seed_means_df.count(axis=0).replace(0, np.nan)
        bench_sem  = bench_std / np.sqrt(bench_count)

        results[metric_key] = pd.DataFrame({"mean": bench_mean, "sem": bench_sem})
    return results


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

            overall = ds_mean.mean(numeric_only=True)

            # Print summary
            print(f"\n{'='*60}")
            print(f"  {ds.upper()} — {metric_key.upper()}")
            print(f"{'='*60}")
            print(ds_mean.round(4).to_string())
            print(f"\n  Overall mean ({ds.upper()}):")
            for col, val in overall.items():
                print(f"    {col}: {val:.4f}")

            # Save per-class table CSV
            ds_mean.to_csv(tables_dir / f"{ds}_{metric_key}_per_class.csv")

            # Save overall stats CSV
            overall_df = overall.to_frame(name="mean")
            overall_df.to_csv(tables_dir / f"{ds}_{metric_key}_overall.csv")

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
            (tables_dir / f"{ds}_{metric_key}.tex").write_text(latex)

    print(f"\nImage results written to {tables_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results and export tables/plots.")
    parser.add_argument("mode", choices=["semi", "unsup", "images"], help="results folder to use")
    parser.add_argument("--out", default=None, help="output directory (default: results/<mode>)")
    args = parser.parse_args()

    if args.mode == "images":
        base_dir = Path("results") / "images"
        if not base_dir.exists():
            raise SystemExit(f"Results directory not found: {base_dir}")
        out_dir = Path(args.out) if args.out else base_dir
        process_image_results(base_dir, out_dir)
        print(f"Done. Outputs in {out_dir}")
        return

    base_dir = resolve_base_dir(args.mode)

    out_dir = Path(args.out) if args.out else base_dir
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    ensure_dir(tables_dir)
    ensure_dir(plots_dir)

    # Load summaries from the reparametrized-loss directory (ablations/<mode>) if it exists.
    reparam_dir = Path("ablations") / args.mode
    reparam_summaries = compute_summary(reparam_dir) if reparam_dir.exists() else {}

    for metric_key, metric_name in METRICS.items():
        pattern = f"*_{metric_name}.csv"
        seed_files = sorted(base_dir.glob(pattern))
        if not seed_files:
            print(f"No files for {metric_name} in {base_dir}")
            continue
        
        #seed_files = seed_files[:3]

        seed_dfs = {}
        for path in seed_files:
            seed = seed_from_filename(path)
            df = read_seed_df(path)
            if df.empty:
                continue
            df = aggregate_datasets(df)
            seed_dfs[seed] = df

        if not seed_dfs:
            print(f"No matching ablation models found for {metric_name} in {base_dir}")
            continue

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
                    cell = f"\\\\textbf{{{cell}}}"
                formatted.append(cell)
            formatted_rows.append(formatted)

        table_formatted = pd.DataFrame(formatted_rows, index=table_mean.index, columns=table_mean.columns)
        latex = table_formatted.to_latex(escape=False, column_format="l" + "c" * len(table_mean.columns))
        (tables_dir / f"{metric_key}.tex").write_text(latex)

        # Benchmark averages per model (mean over datasets per seed, then mean/sem over seeds)
        seed_means = []
        seed_labels = []
        for seed, df in seed_dfs.items():
            if seed in PLOT_EXCLUDE_SEEDS:
                continue
            df = df.reindex(ordered)
            seed_means.append(df.mean(axis=0))
            seed_labels.append(seed)

        if not seed_means:
            print(
                f"No seeds available for benchmark/plot summary in {metric_name} "
                f"after excluding: {sorted(PLOT_EXCLUDE_SEEDS)}"
            )
            continue
        seed_means_df = pd.DataFrame(seed_means, index=seed_labels)

        bench_mean = seed_means_df.mean(axis=0)
        bench_std = seed_means_df.std(axis=0, ddof=1)
        bench_count = seed_means_df.count(axis=0).replace(0, np.nan)
        bench_sem = bench_std / np.sqrt(bench_count)

        summary = pd.DataFrame({"mean": bench_mean, "sem": bench_sem})
        summary = summary.reindex([m for m in MODEL_ORDER if m in summary.index])
        summary.to_csv(out_dir / f"{metric_key}_benchmark.csv")

        print(f"\nBenchmark summary ({metric_key.upper()}):")
        for model in summary.index:
            m = summary.loc[model, "mean"]
            s = summary.loc[model, "sem"]
            print(f"  {model}: {m:.4f} ± {s:.4f}")

        # Plot — two-line ablation: K-DSM vs DSM across sigma/EMA values
        levels, kdsm_tpl, dsm_tpl, x_label = get_main_ablation_family(summary.index)
        kdsm_cols = [kdsm_tpl.format(s) for s in levels if kdsm_tpl.format(s) in summary.index]
        dsm_cols  = [dsm_tpl.format(s)  for s in levels if dsm_tpl.format(s)  in summary.index]

        color_kdsm = plt.cm.magma(0.8)
        color_dsm  = plt.cm.magma(0.50)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(levels))

        def _plot_line(cols, color, label, marker):
            if not cols:
                return
            means = summary.loc[cols, "mean"].values
            sems  = summary.loc[cols, "sem"].values
            ax.errorbar(
                x[:len(cols)], means, yerr=sems,
                fmt=f"{marker}-",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
                linewidth=2.0,
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=label,
                alpha=0.92,
            )

        _plot_line(kdsm_cols, color_kdsm, "K-DSM", "o")
        _plot_line(dsm_cols,  color_dsm,  "DSM",   "s")

        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=10)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(METRIC_TITLES[metric_key], fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35, color="grey")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric_key}_ablation.pdf", dpi=200, bbox_inches="tight")
        fig.savefig(plots_dir / f"{metric_key}_ablation.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Combined plot — 4 curves: K-DSM/DSM × normal/reparametrized loss
        if metric_key in reparam_summaries:
            summary_reparam = reparam_summaries[metric_key]
            levels_r, kdsm_tpl_r, dsm_tpl_r, x_label_r = get_main_ablation_family(summary_reparam.index)
            kdsm_cols_r = [kdsm_tpl_r.format(s) for s in levels_r if kdsm_tpl_r.format(s) in summary_reparam.index]
            dsm_cols_r  = [dsm_tpl_r.format(s)  for s in levels_r if dsm_tpl_r.format(s)  in summary_reparam.index]

            if levels_r != levels:
                print(
                    f"Skipping combined sigma/EMA plot for {metric_name}: "
                    "normal and reparametrized runs use different ablation levels."
                )
            else:
                color_kdsm_r = plt.cm.magma(0.7)
                color_dsm_r  = plt.cm.magma(0.40)
                color_kdsm_n = plt.cm.viridis(0.7)
                color_dsm_n  = plt.cm.viridis(0.40)

                fig_c, ax_c = plt.subplots(figsize=(5, 3.5))
                x = np.arange(len(levels))

                def _plot_line_on(ax, summary_src, cols, color, label, marker, linestyle="-"):
                    if not cols:
                        return
                    means = summary_src.loc[cols, "mean"].values
                    sems  = summary_src.loc[cols, "sem"].values
                    ax.errorbar(
                        x[:len(cols)], means, yerr=sems,
                        fmt=f"{marker}{linestyle}",
                        color=color, ecolor=color,
                        elinewidth=1.2, capsize=4, capthick=1.2,
                        linewidth=2.0, markersize=6,
                        markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=0.8, label=label, alpha=0.92,
                    )

                _plot_line_on(ax_c, summary_reparam, kdsm_cols_r, color_kdsm_r, "K-DSM (reparam.)", "o")
                _plot_line_on(ax_c, summary_reparam, dsm_cols_r,  color_dsm_r,  "DSM (reparam.)",   "s")
                _plot_line_on(ax_c, summary,         kdsm_cols,   color_kdsm_n, "K-DSM",            "o", "--")
                _plot_line_on(ax_c, summary,         dsm_cols,    color_dsm_n,  "DSM",              "s", "--")

                ax_c.set_xticks(x)
                ax_c.set_xticklabels(levels, fontsize=10)
                ax_c.set_xlabel(x_label_r, fontsize=11)
                ax_c.set_ylabel(METRIC_TITLES[metric_key], fontsize=11)
                ax_c.tick_params(axis="both", labelsize=10)
                ax_c.grid(axis="y", linestyle="--", alpha=0.35, color="grey")
                ax_c.set_axisbelow(True)
                ax_c.spines[["top", "right"]].set_visible(False)
                ax_c.legend(frameon=False, fontsize=10, loc="best")
                fig_c.tight_layout()
                fig_c.savefig(plots_dir / f"{metric_key}_ablation_sigma_combined.pdf", dpi=200, bbox_inches="tight")
                fig_c.savefig(plots_dir / f"{metric_key}_ablation_sigma_combined.png", dpi=200, bbox_inches="tight")
                plt.close(fig_c)

        # Plot — tau ablation: K-DSMc vs K-DSMggd across tau values
        kdsmc_cols  = [m for m in TAU_KDSMC_MODELS   if m in summary.index]
        kdsm_ggd_cols = [m for m in TAU_KDSM_GGD_MODELS if m in summary.index]

        if kdsmc_cols or kdsm_ggd_cols:
            color_kdsmc   = plt.cm.magma(0.85)
            color_kdsm_ggd = plt.cm.magma(0.2)

            fig2, ax2 = plt.subplots(figsize=(5, 3.5))

            def _tau_from_model(name: str) -> float:
                return float(name.split("_")[-1])

            def _c_from_kdsmc(name: str) -> str:
                return name.split("_")[-2]

            if kdsmc_cols:
                kdsmc_sorted = sorted(kdsmc_cols, key=_tau_from_model)
                tau_x = [_tau_from_model(m) for m in kdsmc_sorted]
                means = summary.loc[kdsmc_sorted, "mean"].values
                sems  = summary.loc[kdsmc_sorted, "sem"].values
                ax2.errorbar(
                    tau_x, means, yerr=sems,
                    fmt="o-",
                    color=color_kdsmc,
                    ecolor=color_kdsmc,
                    elinewidth=1.2,
                    capsize=4,
                    capthick=1.2,
                    linewidth=2.0,
                    markersize=6,
                    markerfacecolor=color_kdsmc,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label="Cornish-Fisher",
                    alpha=0.92,
                )
                for i, (m, tx, val) in enumerate(zip(kdsmc_sorted, tau_x, means)):
                    if i == 0:
                        xyoffset = (6, 12)
                    elif i == 1:
                        xyoffset = (0, 16)
                    elif i == 2:
                        xyoffset = (4, 10)
                    elif i == 4:
                        xyoffset = (0, 16)
                    elif i == 5:
                        xyoffset = (4, 10)
                    c_val = "c=" + _c_from_kdsmc(m)
                    ax2.annotate(
                        c_val,
                        xy=(tx, val),
                        xytext=xyoffset,
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="black",
                    )

            if kdsm_ggd_cols:
                ggd_sorted = sorted(kdsm_ggd_cols, key=_tau_from_model)
                tau_x_ggd = [_tau_from_model(m) for m in ggd_sorted]
                means_ggd = summary.loc[ggd_sorted, "mean"].values
                sems_ggd  = summary.loc[ggd_sorted, "sem"].values
                ax2.errorbar(
                    tau_x_ggd, means_ggd, yerr=sems_ggd,
                    fmt="s-",
                    color=color_kdsm_ggd,
                    ecolor=color_kdsm_ggd,
                    elinewidth=1.2,
                    capsize=4,
                    capthick=1.2,
                    linewidth=2.0,
                    markersize=6,
                    markerfacecolor=color_kdsm_ggd,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label="General Gaussian",
                    alpha=0.92,
                )

            ax2.set_xscale("log")
            ax2.set_xlabel(r"$\tau$", fontsize=11)
            ax2.set_ylabel(METRIC_TITLES[metric_key], fontsize=11)
            ax2.tick_params(axis="both", labelsize=10)
            ax2.grid(axis="y", linestyle="--", alpha=0.35, color="grey")
            ax2.set_axisbelow(True)
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.legend(frameon=False, fontsize=10, loc="best")
            fig2.tight_layout()
            fig2.savefig(plots_dir / f"{metric_key}_ablation_tau.pdf", dpi=200, bbox_inches="tight")
            fig2.savefig(plots_dir / f"{metric_key}_ablation_tau.png", dpi=200, bbox_inches="tight")
            plt.close(fig2)

            # Combined tau plot — 4 curves: Cornish-Fisher/GGD × normal/reparametrized loss
            if metric_key in reparam_summaries:
                summary_reparam = reparam_summaries[metric_key]
                kdsmc_cols_r    = [m for m in TAU_KDSMC_MODELS   if m in summary_reparam.index]
                kdsm_ggd_cols_r = [m for m in TAU_KDSM_GGD_MODELS if m in summary_reparam.index]

                if kdsmc_cols_r or kdsm_ggd_cols_r:
                    color_kdsmc_r   = plt.cm.magma(0.8)
                    color_kdsm_ggd_r = plt.cm.magma(0.2)
                    color_kdsmc_n   = plt.cm.magma(0.85)
                    color_kdsm_ggd_n = plt.cm.magma(0.25)

                    fig3, ax3 = plt.subplots(figsize=(5, 3.5))

                    def _plot_tau_line(ax, summary_src, cols, color, label, marker, linestyle="-"):
                        if not cols:
                            return
                        sorted_cols = sorted(cols, key=_tau_from_model)
                        tau_x = [_tau_from_model(m) for m in sorted_cols]
                        means = summary_src.loc[sorted_cols, "mean"].values
                        sems  = summary_src.loc[sorted_cols, "sem"].values
                        ax.errorbar(
                            tau_x, means, yerr=sems,
                            fmt=f"{marker}{linestyle}",
                            color=color, ecolor=color,
                            elinewidth=1.2, capsize=4, capthick=1.2,
                            linewidth=2.0, markersize=6,
                            markerfacecolor=color, markeredgecolor="white",
                            markeredgewidth=0.8, label=label, alpha=0.92,
                        )

                    _plot_tau_line(ax3, summary_reparam, kdsmc_cols_r,    color_kdsmc_r,    "Cornish-Fisher (reparam.)", "o")
                    _plot_tau_line(ax3, summary_reparam, kdsm_ggd_cols_r, color_kdsm_ggd_r, "Gen. Gaussian (reparam.)", "s")
                    _plot_tau_line(ax3, summary,         kdsmc_cols,      color_kdsmc_n,    "Cornish-Fisher",            "o", "--")
                    _plot_tau_line(ax3, summary,         kdsm_ggd_cols,   color_kdsm_ggd_n, "Gen. Gaussian",             "s", "--")

                    ax3.set_xscale("log")
                    ax3.set_xlabel(r"$\tau$", fontsize=11)
                    ax3.set_ylabel(METRIC_TITLES[metric_key], fontsize=11)
                    ax3.tick_params(axis="both", labelsize=10)
                    ax3.grid(axis="y", linestyle="--", alpha=0.35, color="grey")
                    ax3.set_axisbelow(True)
                    ax3.spines[["top", "right"]].set_visible(False)
                    ax3.legend(frameon=False, fontsize=10, loc="best")
                    fig3.tight_layout()
                    fig3.savefig(plots_dir / f"{metric_key}_ablation_tau_combined.pdf", dpi=200, bbox_inches="tight")
                    fig3.savefig(plots_dir / f"{metric_key}_ablation_tau_combined.png", dpi=200, bbox_inches="tight")
                    plt.close(fig3)

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
