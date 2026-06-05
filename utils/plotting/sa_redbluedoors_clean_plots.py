"""
Red–Blue Doors (9 agents): clean plots from nine_agents_comparison_*.csv logs.

Uses the same log files as plot_sa_redbluebuttons_nine.py and the same clean-plot
styles as overcooked_clean_plots.py (seed dots, metric grid, faint seed lines,
checkpoint CIs, small multiples, rolling success, paired baseline differences).

Usage:
    python utils/plotting/sa_redbluedoors_clean_plots.py logs/sa_redbluedoors \\
        -o results/RedBlueDoors/plots_clean

    python utils/plotting/sa_redbluedoors_clean_plots.py logs/sa_redbluedoors \\
        -o results/RedBlueDoors/plots_clean --baseline QLearning --checkpoint-every 25
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from overcooked_clean_plots import (  # noqa: E402
    ci95,
    ensure_dir,
    plot_final_seed_dotplot,
    savefig,
)
from plot_sa_redbluebuttons_nine import (  # noqa: E402
    EPISODES_PER_CONFIG_DEFAULT,
    collect_log_files_from_paths,
    load_episode_data,
)

W_ROLLING = 25
CHECKPOINT_EVERY_DEFAULT = EPISODES_PER_CONFIG_DEFAULT

METRIC_LABELS: dict[str, tuple[str, str]] = {
    "episode_return_mean": ("Mean episode return", "Mean episode return"),
    "success_rate": ("Success rate", "Fraction of episodes successful"),
    "mean_episode_length": ("Mean episode length", "Mean steps per episode"),
    "total_return": ("Total return", "Sum of episode returns"),
}

MAIN_SEED_METRICS = [
    "episode_return_mean",
    "success_rate",
    "mean_episode_length",
    "total_return",
]

GRID_METRICS = [
    "episode_return_mean",
    "success_rate",
    "mean_episode_length",
]


def build_tables_from_episode_df(episode_df: pd.DataFrame) -> dict:
    """
    Build tables from per-episode data. Uses overcooked column names where
    curve helpers expect them (paradigm, brain_seed, step, cumulative_soups).
    """
    ep = episode_df.copy()
    if "algorithm" not in ep.columns and "agent" in ep.columns:
        ep = ep.rename(columns={"agent": "algorithm"})

    seed_metrics = (
        ep.groupby(["seed", "algorithm"], as_index=False)
        .agg(
            episode_return_mean=("episode_return", "mean"),
            success_rate=("success", "mean"),
            mean_episode_length=("episode_length", "mean"),
            total_return=("episode_return", "sum"),
        )
        .rename(columns={"algorithm": "paradigm", "seed": "brain_seed"})
    )

    curve_rows = []
    for (paradigm, brain_seed), g in ep.groupby(["algorithm", "seed"]):
        g = g.sort_values("episode")
        for episode, episode_return, cumulative_return in zip(
            g["episode"], g["episode_return"], g["episode_return"].cumsum()
        ):
            curve_rows.append(
                {
                    "paradigm": paradigm,
                    "brain_seed": brain_seed,
                    "step": int(episode),
                    "cumulative_soups": float(cumulative_return),
                    "episode_return": float(episode_return),
                }
            )
    seed_curves = pd.DataFrame(curve_rows)

    agg_rows = []
    for (paradigm, step), g in seed_curves.groupby(["paradigm", "step"]):
        vals = g["cumulative_soups"]
        agg_rows.append(
            {
                "paradigm": paradigm,
                "step": step,
                "mean_cumulative_soups": vals.mean(),
                "ci95_cumulative_soups": ci95(vals),
            }
        )
    aggregate_curves = pd.DataFrame(agg_rows)

    ep_curve_rows = []
    for (paradigm, brain_seed), g in ep.groupby(["algorithm", "seed"]):
        g = g.sort_values("episode")
        for _, row in g.iterrows():
            ep_curve_rows.append(
                {
                    "paradigm": paradigm,
                    "brain_seed": brain_seed,
                    "step": int(row["episode"]),
                    "episode_return": float(row["episode_return"]),
                }
            )
    seed_episode_return_curves = pd.DataFrame(ep_curve_rows)

    agg_ep_rows = []
    for (paradigm, step), g in seed_episode_return_curves.groupby(["paradigm", "step"]):
        vals = g["episode_return"]
        agg_ep_rows.append(
            {
                "paradigm": paradigm,
                "step": step,
                "mean_episode_return": vals.mean(),
                "ci95_episode_return": ci95(vals),
            }
        )
    aggregate_episode_return_curves = pd.DataFrame(agg_ep_rows)

    window_rows = []
    for (paradigm, brain_seed), g in ep.groupby(["algorithm", "seed"]):
        g = g.sort_values("episode").reset_index(drop=True)
        for start in range(len(g)):
            end = min(start + W_ROLLING, len(g))
            window = g.iloc[start:end]
            window_rows.append(
                {
                    "paradigm": paradigm,
                    "brain_seed": brain_seed,
                    "window_start": int(window["episode"].iloc[0]),
                    "success_pct": float(window["success"].mean()) * 100.0,
                }
            )
    seed_window = pd.DataFrame(window_rows)

    agg_win_rows = []
    for (paradigm, window_start), g in seed_window.groupby(["paradigm", "window_start"]):
        vals = g["success_pct"]
        agg_win_rows.append(
            {
                "paradigm": paradigm,
                "window_start": window_start,
                "mean_success_pct": vals.mean(),
            }
        )
    aggregate_window = pd.DataFrame(agg_win_rows)

    return {
        "seed_metrics": seed_metrics,
        "seed_curves": seed_curves,
        "aggregate_curves": aggregate_curves,
        "seed_episode_return_curves": seed_episode_return_curves,
        "aggregate_episode_return_curves": aggregate_episode_return_curves,
        "seed_window": seed_window,
        "aggregate_window": aggregate_window,
    }


def plot_metric_grid_rb(
    seed_metrics: pd.DataFrame,
    metrics: list[str],
    output_path: Path | str,
    title: str = "Seed-level metrics with mean ± 95% CI",
) -> None:
    metrics = [m for m in metrics if m in seed_metrics.columns]
    if not metrics:
        warnings.warn("No valid metrics for metric grid.")
        return

    paradigms = list(seed_metrics["paradigm"].dropna().unique())
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.3 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        x = np.arange(len(paradigms))
        means, cis = [], []
        for i, paradigm in enumerate(paradigms):
            values = (
                seed_metrics.loc[seed_metrics["paradigm"] == paradigm, metric]
                .dropna()
                .reset_index(drop=True)
            )
            if len(values) == 0:
                means.append(np.nan)
                cis.append(np.nan)
                continue
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), i) + jitter, values, alpha=0.75, s=35, zorder=2)
            means.append(values.mean())
            cis.append(ci95(values))
        ax.errorbar(x, means, yerr=cis, fmt="o", capsize=4, linewidth=2, markersize=6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(paradigms, rotation=25, ha="right")
        plot_title, ylabel = METRIC_LABELS.get(metric, (metric, metric))
        ax.set_title(plot_title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(title, y=1.01, fontsize=14)
    savefig(output_path)


def plot_mean_with_seed_lines(
    seed_curves: pd.DataFrame,
    agg_curves: pd.DataFrame,
    seed_y: str,
    agg_y: str,
    output_path: Path | str,
    xlabel: str,
    ylabel: str,
    title: str,
    x_col: str = "step",
    seed_x_col: str | None = None,
) -> None:
    seed_x = seed_x_col or x_col
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for paradigm, agg_pg in agg_curves.groupby("paradigm"):
        agg_pg = agg_pg.sort_values(x_col)
        mean_line, = ax.plot(
            agg_pg[x_col], agg_pg[agg_y], linewidth=3.0, label=f"{paradigm} mean", zorder=3
        )
        color = mean_line.get_color()
        seed_pg = seed_curves[seed_curves["paradigm"] == paradigm]
        for _, seed_g in seed_pg.groupby("brain_seed"):
            seed_g = seed_g.sort_values(seed_x)
            ax.plot(seed_g[seed_x], seed_g[seed_y], color=color, alpha=0.18, linewidth=1.0, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(output_path)


def plot_checkpoint_ci(
    agg_curves: pd.DataFrame,
    mean_col: str,
    ci_col: str,
    output_path: Path | str,
    xlabel: str,
    ylabel: str,
    title: str,
    checkpoint_every: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for paradigm, pg in agg_curves.groupby("paradigm"):
        pg = pg.sort_values("step")
        ax.plot(pg["step"], pg[mean_col], linewidth=3.0, label=f"{paradigm} mean")
        checkpoints = pg[pg["step"] % checkpoint_every == 0]
        ax.errorbar(
            checkpoints["step"],
            checkpoints[mean_col],
            yerr=checkpoints[ci_col],
            fmt="none",
            capsize=3,
            linewidth=1.5,
            alpha=0.85,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(output_path)


def plot_small_multiples_rb(
    seed_curves: pd.DataFrame,
    agg_curves: pd.DataFrame,
    seed_y: str,
    agg_y: str,
    output_path: Path | str,
    suptitle: str,
    ylabel: str,
) -> None:
    paradigms = list(agg_curves["paradigm"].dropna().unique())
    if not paradigms:
        return
    ncols = 3
    nrows = math.ceil(len(paradigms) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), sharex=True, sharey=True, squeeze=False
    )
    axes_flat = axes.flatten()
    for ax, paradigm in zip(axes_flat, paradigms):
        seed_pg = seed_curves[seed_curves["paradigm"] == paradigm]
        agg_pg = agg_curves[agg_curves["paradigm"] == paradigm].sort_values("step")
        mean_line, = ax.plot(
            agg_pg["step"], agg_pg[agg_y], linewidth=3.0, label="Mean", zorder=3
        )
        color = mean_line.get_color()
        for _, seed_g in seed_pg.groupby("brain_seed"):
            seed_g = seed_g.sort_values("step")
            ax.plot(seed_g["step"], seed_g[seed_y], color=color, alpha=0.25, linewidth=1.0, zorder=1)
        ax.set_title(paradigm)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(paradigms) :]:
        ax.axis("off")
    fig.suptitle(suptitle, y=1.01, fontsize=14)
    savefig(output_path)


def plot_difference_from_baseline_rb(
    seed_metrics: pd.DataFrame,
    metric: str,
    baseline_name: str,
    output_path: Path | str,
) -> pd.DataFrame:
    from overcooked_clean_plots import compute_difference_from_baseline

    diff_df = compute_difference_from_baseline(seed_metrics, metric, baseline_name)
    if diff_df.empty:
        warnings.warn(f"No difference plot for {metric}.")
        return diff_df

    paradigms = list(diff_df["paradigm"].dropna().unique())
    x = np.arange(len(paradigms))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    means, cis = [], []
    for i, paradigm in enumerate(paradigms):
        values = diff_df.loc[diff_df["paradigm"] == paradigm, "difference"].dropna().reset_index(drop=True)
        jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), i) + jitter, values, alpha=0.8, s=48, zorder=2)
        means.append(values.mean())
        cis.append(ci95(values))
    ax.errorbar(
        x, means, yerr=cis, fmt="o", capsize=5, linewidth=2.3, markersize=8,
        label="Mean paired difference ± 95% CI", zorder=3,
    )
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, rotation=25, ha="right")
    plot_title, _ = METRIC_LABELS.get(metric, (metric, metric))
    ax.set_ylabel(f"Δ {plot_title} vs {baseline_name}")
    ax.set_title(f"Paired difference from {baseline_name}: {plot_title}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    savefig(output_path)
    return diff_df


def run_clean_plotting(
    log_paths: list[Path],
    output_dir: Path | str,
    baseline_name: str | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY_DEFAULT,
) -> None:
    output_dir = ensure_dir(output_dir)
    log_files = collect_log_files_from_paths(log_paths)

    print(f"Using {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  {f.name}")
    print()

    episode_df, agent_names, episodes = load_episode_data(log_files)
    print(f"Episodes {episodes.min()}–{episodes.max()}, algorithms: {list(agent_names)}")

    tables = build_tables_from_episode_df(episode_df)
    seed_metrics = tables["seed_metrics"]

    print("Creating seed-level dot plots...")
    for metric in MAIN_SEED_METRICS:
        if metric not in seed_metrics.columns:
            continue
        title, ylabel = METRIC_LABELS[metric]
        plot_final_seed_dotplot(
            seed_metrics=seed_metrics,
            metric=metric,
            output_path=output_dir / f"{metric}_seed_dots_mean_ci.png",
            ylabel=ylabel,
            title=title,
        )

    plot_metric_grid_rb(
        seed_metrics=seed_metrics,
        metrics=GRID_METRICS,
        output_path=output_dir / "main_metrics_grid_seed_dots_mean_ci.png",
        title="Red–Blue Doors: main metrics (mean ± 95% CI)",
    )

    print("Creating learning curves...")
    plot_mean_with_seed_lines(
        tables["seed_curves"],
        tables["aggregate_curves"],
        seed_y="cumulative_soups",
        agg_y="mean_cumulative_soups",
        output_path=output_dir / "cumulative_return_clean_seed_lines.png",
        xlabel="Episode",
        ylabel="Cumulative episode return",
        title="Cumulative return over training",
    )
    plot_checkpoint_ci(
        tables["aggregate_curves"],
        mean_col="mean_cumulative_soups",
        ci_col="ci95_cumulative_soups",
        output_path=output_dir / "cumulative_return_checkpoint_ci.png",
        xlabel="Episode",
        ylabel="Cumulative episode return",
        title=f"Cumulative return with 95% CI every {checkpoint_every} episodes",
        checkpoint_every=checkpoint_every,
    )
    plot_small_multiples_rb(
        tables["seed_curves"],
        tables["aggregate_curves"],
        seed_y="cumulative_soups",
        agg_y="mean_cumulative_soups",
        output_path=output_dir / "cumulative_return_small_multiples.png",
        suptitle="Cumulative return by algorithm",
        ylabel="Cumulative return",
    )

    plot_mean_with_seed_lines(
        tables["seed_episode_return_curves"],
        tables["aggregate_episode_return_curves"],
        seed_y="episode_return",
        agg_y="mean_episode_return",
        output_path=output_dir / "episode_return_clean_seed_lines.png",
        xlabel="Episode",
        ylabel="Episode return",
        title="Episode return over training",
    )
    plot_checkpoint_ci(
        tables["aggregate_episode_return_curves"],
        mean_col="mean_episode_return",
        ci_col="ci95_episode_return",
        output_path=output_dir / "episode_return_checkpoint_ci.png",
        xlabel="Episode",
        ylabel="Episode return",
        title=f"Episode return with 95% CI every {checkpoint_every} episodes",
        checkpoint_every=checkpoint_every,
    )

    print("Creating rolling success-rate plot...")
    plot_mean_with_seed_lines(
        tables["seed_window"],
        tables["aggregate_window"],
        seed_y="success_pct",
        agg_y="mean_success_pct",
        output_path=output_dir / "rolling_success_rate_clean_seed_lines.png",
        xlabel="Episode (window start)",
        ylabel=f"Success rate × 100 ({W_ROLLING}-episode window)",
        title=f"Rolling success rate ({W_ROLLING}-episode window)",
        x_col="window_start",
        seed_x_col="window_start",
    )

    if baseline_name is not None:
        print(f"Creating paired differences vs {baseline_name}...")
        diff_tables = []
        for metric in MAIN_SEED_METRICS:
            if metric not in seed_metrics.columns:
                continue
            try:
                diff_df = plot_difference_from_baseline_rb(
                    seed_metrics, metric, baseline_name,
                    output_dir / f"{metric}_difference_from_{baseline_name}.png",
                )
                if not diff_df.empty:
                    diff_tables.append(diff_df)
            except ValueError as e:
                warnings.warn(str(e))
        if diff_tables:
            pd.concat(diff_tables, ignore_index=True).to_csv(
                output_dir / f"paired_differences_from_{baseline_name}.csv",
                index=False,
            )

    print(f"\nDone. Saved clean plots to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Red–Blue Doors clean plots from nine_agents_comparison_*.csv logs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Directory with nine_agents_comparison_*.csv or explicit CSV paths",
    )
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline algorithm for paired differences (e.g. QLearning)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY_DEFAULT,
        help="CI bars every N episodes (default 25)",
    )
    args = parser.parse_args()
    run_clean_plotting(args.paths, args.out, args.baseline, args.checkpoint_every)


if __name__ == "__main__":
    main()
