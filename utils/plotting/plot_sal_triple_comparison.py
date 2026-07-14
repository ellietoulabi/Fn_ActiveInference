"""
Compare three SAL Overcooked conditions (e.g. FC, IND, MAPPO) on matched episode seeds.

Runs are paired by episode_seed. Team reward = max(reward_a0, reward_a1) per step.

Usage:
    python utils/plotting/plot_sal_triple_comparison.py \\
        --fc logs/sal_fc --ind logs/sal_ind --mappo logs/sal_mappo \\
        -o results/Overcooked/compare_fc_ind_mappo

    python utils/plotting/plot_sal_triple_comparison.py \\
        --fc logs/sal_fc --ind logs/sal_ind --ic logs/sal_ic --mappo logs/sal_mappo \\
        -o results/Overcooked/compare_fc_ind_ic_mappo --smooth-window 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_sal_pair_comparison import (  # noqa: E402
    aggregate_curve,
    ci95,
    cumulative_soup_curves,
    runs_dataframe,
    savefig,
    smooth_curve,
)
from plot_sal_semantic_action_level import W_CURVE, load_seed_csvs  # noqa: E402

DPI = 150
COLORS = {
    "FC": "#2E86AB",
    "IND": "#E67E22",
    "IC": "#8E44AD",
    "MAPPO": "#27AE60",
}
DEFAULT_LABELS = ("FC", "IND", "MAPPO")
DEFAULT_LABELS_FOUR = ("FC", "IND", "IC", "MAPPO")


def _filter_dfs_by_seeds(dfs: list[pd.DataFrame], seeds: set[int]) -> list[pd.DataFrame]:
    out = [df for df in dfs if int(df["episode_seed"].iloc[0]) in seeds]
    return sorted(out, key=lambda d: int(d["episode_seed"].iloc[0]))


def wide_table(runs_by_label: dict[str, pd.DataFrame], common_seeds: list[int]) -> pd.DataFrame:
    rows = []
    for ep in common_seeds:
        row = {"episode_seed": ep}
        for label, runs in runs_by_label.items():
            r = runs.set_index("episode_seed").loc[ep]
            row[f"soups_{label}"] = r["soups_delivered"]
            row[f"return_{label}"] = r["episode_return"]
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(wide: pd.DataFrame, labels: tuple[str, ...]) -> None:
    print("\n" + "=" * 70)
    print(f"MULTI-WAY COMPARISON (n={len(wide)} matched episode_seeds)")
    print("=" * 70)
    print(f"Seeds: {list(wide['episode_seed'])}")
    for label in labels:
        col = f"soups_{label}"
        print(f"  Mean soups {label}: {wide[col].mean():.3f} ± {wide[col].std(ddof=1):.3f}")
    best = wide[[f"soups_{l}" for l in labels]].idxmax(axis=1)
    for label in labels:
        n = int((best == f"soups_{label}").sum())
        print(f"  Best on {n}/{len(wide)} seeds: {label}")


def plot_cumulative_soups(
    steps: np.ndarray,
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    path: Path,
    labels: tuple[str, ...],
    smoothing_window: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for mean, err, label in curves:
        color = COLORS.get(label, None)
        ax.plot(steps, mean, lw=2.5, color=color, label=label)
        if np.any(np.isfinite(err)):
            ax.fill_between(steps, mean - err, mean + err, color=color, alpha=0.2)
    ax.set_xlabel("Timestep within episode")
    ax.set_ylabel("Cumulative soups delivered (mean across matched runs)")
    title = f"{' vs '.join(labels)}: cumulative deliveries"
    if smoothing_window > 1:
        title += f" (smoothed, window={smoothing_window})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, path)


def plot_mean_soups_bar(runs_by_label: dict[str, pd.DataFrame], labels: tuple[str, ...], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    means = [runs_by_label[l]["soups_delivered"].mean() for l in labels]
    errs = [ci95(runs_by_label[l]["soups_delivered"]) for l in labels]
    colors = [COLORS.get(l, "#888888") for l in labels]
    ax.bar(x, means, yerr=errs, capsize=6, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Soups delivered per episode")
    n = len(next(iter(runs_by_label.values())))
    ax.set_title(f"Mean ± 95% CI (n={n} matched runs)")
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, path)


def plot_soups_by_seed(wide: pd.DataFrame, labels: tuple[str, ...], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ep = wide["episode_seed"].astype(str).values
    n = len(labels)
    width = 0.25
    x = np.arange(len(ep))
    for i, label in enumerate(labels):
        offset = (i - (n - 1) / 2) * width
        vals = wide[f"soups_{label}"].values
        ax.bar(x + offset, vals, width, label=label, color=COLORS.get(label), edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ep, rotation=45)
    ax.set_xlabel("Episode seed")
    ax.set_ylabel("Soups delivered")
    ax.set_title("Per-seed soups (matched runs)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, path)


def plot_paired_lines(wide: pd.DataFrame, labels: tuple[str, ...], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    x_pos = {labels[i]: i for i in range(len(labels))}
    for _, row in wide.iterrows():
        ys = [row[f"soups_{l}"] for l in labels]
        ax.plot(
            [x_pos[l] for l in labels],
            ys,
            "o-",
            color="#888888",
            alpha=0.55,
            lw=1.2,
            markersize=5,
        )
        ax.text(-0.15, ys[0], str(int(row["episode_seed"])), ha="right", va="center", fontsize=7)
    for label in labels:
        ax.scatter(
            np.full(len(wide), x_pos[label]),
            wide[f"soups_{label}"],
            s=70,
            color=COLORS.get(label),
            label=label,
            zorder=3,
        )
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Soups delivered")
    ax.set_title("Paired by episode_seed")
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, path)


def run_comparison(
    log_dirs: dict[str, Path],
    output_dir: Path,
    labels: tuple[str, ...],
    smoothing_window: int = W_CURVE,
) -> None:
    output_dir = Path(output_dir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    paradigm_key = {"FC": "fc", "IND": "ind", "IC": "ic", "MAPPO": "mappo"}

    dfs_by_label: dict[str, list] = {}
    runs_by_label: dict[str, pd.DataFrame] = {}
    for label in labels:
        par = paradigm_key.get(label, label.lower())
        dfs, _, _ = load_seed_csvs(log_dirs[label], par)
        dfs_by_label[label] = dfs
        runs_by_label[label] = runs_dataframe(dfs, label)

    common = sorted(
        set.intersection(*[set(r["episode_seed"]) for r in runs_by_label.values()])
    )
    if not common:
        raise ValueError(f"No episode_seed shared by all conditions: {labels}")

    common_set = set(common)
    runs_by_label = {
        label: runs[runs["episode_seed"].isin(common_set)].sort_values("episode_seed")
        for label, runs in runs_by_label.items()
    }
    dfs_by_label = {
        label: _filter_dfs_by_seeds(dfs_by_label[label], common_set) for label in labels
    }

    wide = wide_table(runs_by_label, common)
    print_summary(wide, labels)

    wide.to_csv(output_dir / "tables" / "paired_all_conditions.csv", index=False)
    for label, runs in runs_by_label.items():
        runs.to_csv(output_dir / "tables" / f"runs_{label.lower()}.csv", index=False)

    steps_ref, _, _ = cumulative_soup_curves(dfs_by_label[labels[0]])
    curves = []
    for label in labels:
        steps, mat, _ = cumulative_soup_curves(dfs_by_label[label])
        if not np.array_equal(steps, steps_ref):
            n = min(len(steps), len(steps_ref))
            steps = steps_ref[:n]
            mat = mat[:, :n]
        mean, err = aggregate_curve(mat)
        mean, err = smooth_curve(mean, err, smoothing_window)
        curves.append((mean, err, label))

    plots = output_dir / "plots"
    print(f"\nSaving plots → {plots}")
    if smoothing_window > 1:
        print(f"  Cumulative curve smoothing: rolling window = {smoothing_window}")
    plot_steps = steps_ref[: len(curves[0][0])]
    plot_cumulative_soups(
        plot_steps,
        curves,
        plots / "compare_cumulative_soups.png",
        labels,
        smoothing_window=smoothing_window,
    )
    plot_mean_soups_bar(runs_by_label, labels, plots / "compare_mean_soups.png")
    plot_soups_by_seed(wide, labels, plots / "compare_soups_by_seed.png")
    plot_paired_lines(wide, labels, plots / "compare_paired_soups.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-way SAL comparison (FC, IND, IC, MAPPO)"
    )
    parser.add_argument("--fc", type=Path, default=Path("logs/sal_fc"))
    parser.add_argument("--ind", type=Path, default=Path("logs/sal_ind"))
    parser.add_argument("--ic", type=Path, default=None, help="IC logs; if set, 4-way comparison")
    parser.add_argument("--mappo", type=Path, default=Path("logs/sal_mappo"))
    parser.add_argument("--label-fc", default="FC")
    parser.add_argument("--label-ind", default="IND")
    parser.add_argument("--label-ic", default="IC")
    parser.add_argument("--label-mappo", default="MAPPO")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=W_CURVE,
        help=f"Rolling window for compare_cumulative_soups (default: {W_CURVE}; use 1 for raw)",
    )
    args = parser.parse_args()

    if args.ic is not None:
        labels = (args.label_fc, args.label_ind, args.label_ic, args.label_mappo)
        log_dirs = {
            args.label_fc: args.fc,
            args.label_ind: args.ind,
            args.label_ic: args.ic,
            args.label_mappo: args.mappo,
        }
    else:
        labels = (args.label_fc, args.label_ind, args.label_mappo)
        log_dirs = {
            args.label_fc: args.fc,
            args.label_ind: args.ind,
            args.label_mappo: args.mappo,
        }

    run_comparison(log_dirs, args.output_dir, labels=labels, smoothing_window=args.smooth_window)
    print(f"\nDone. Tables in {args.output_dir / 'tables'}, plots in {args.output_dir / 'plots'}")


if __name__ == "__main__":
    main()
