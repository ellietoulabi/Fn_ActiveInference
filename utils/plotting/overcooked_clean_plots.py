"""
Overcooked: clean evaluation plots from pre-aggregated metric tables.

Reads CSV tables produced by overcooked_log_metrics.py (seed_metrics.csv and
optional curve/window tables), then writes dot plots, grids, cumulative soup
curves, delivery-rate curves, action rates, and optional paired-difference plots.

For Red–Blue Doors nine-agent logs (nine_agents_comparison_*.csv), use
sa_redbluedoors_clean_plots.py instead.

Usage:
    python utils/plotting/overcooked_clean_plots.py \\
        --tables-dir results/Overcooked/my_run/tables \\
        --out results/Overcooked/my_run/plots_clean

    python utils/plotting/overcooked_clean_plots.py \\
        --tables-dir results/Overcooked/my_run/tables \\
        --out results/Overcooked/my_run/plots_clean \\
        --baseline PPO \\
        --checkpoint-every 250
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Basic utilities
# =============================================================================


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ci95(x) -> float:
    """
    95% confidence interval half-width across independent seeds.
    Use after reducing to one value per brain_seed.
    """
    x = pd.Series(x).dropna()
    n = len(x)

    if n <= 1:
        return np.nan

    tcrit_table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
    }

    tcrit = tcrit_table.get(n, 1.96)
    return tcrit * x.std(ddof=1) / np.sqrt(n)


def savefig(path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def short_metric_name(metric: str) -> str:
    mapping = {
        "soups_delivered": "Soups delivered",
        "episode_return": "Episode return",
        "soups_per_100_steps": "Soups / 100 steps",
        "inactive_tail_length": "Inactive tail length",
        "time_to_first_delivery": "Time to first delivery",
        "joint_semantic_switch_rate": "Joint semantic switch rate",
        "joint_policy_switch_rate": "Joint policy switch rate",
        "mean_policy_entropy": "Policy entropy",
        "mean_top_policy_prob": "Top policy probability",
        "mean_stay_rate": "Stay rate",
        "mean_interact_rate": "Interact rate",
        "mean_move_rate": "Move rate",
    }
    return mapping.get(metric, metric.replace("_", " ").title())


def metric_ylabel(metric: str) -> str:
    mapping = {
        "soups_delivered": "Soups delivered per 1500-step episode",
        "episode_return": "Episode return",
        "soups_per_100_steps": "Soups per 100 steps",
        "inactive_tail_length": "Steps after last delivery",
        "time_to_first_delivery": "Steps until first delivery",
        "joint_semantic_switch_rate": "Fraction of steps with semantic switch",
        "joint_policy_switch_rate": "Fraction of steps with policy switch",
        "mean_policy_entropy": "Mean policy entropy",
        "mean_top_policy_prob": "Mean top policy probability",
        "mean_stay_rate": "Fraction of primitive actions",
        "mean_interact_rate": "Fraction of primitive actions",
        "mean_move_rate": "Fraction of primitive actions",
    }
    return mapping.get(metric, short_metric_name(metric))


# =============================================================================
# Loading tables
# =============================================================================


def load_tables(tables_dir: Path | str) -> dict:
    tables_dir = Path(tables_dir)

    seed_metrics_path = tables_dir / "seed_metrics.csv"
    if not seed_metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find {seed_metrics_path}. "
            "Run the metrics script first, then pass its tables/ directory here."
        )

    out: dict = {}
    out["seed_metrics"] = pd.read_csv(seed_metrics_path)

    optional_files = {
        "episode_metrics": "episode_metrics.csv",
        "aggregate_metrics": "aggregate_metrics.csv",
        "seed_soup_curves": "seed_cumulative_soup_curves.csv",
        "aggregate_soup_curves": "aggregate_cumulative_soup_curves.csv",
        "seed_reward_curves": "seed_cumulative_reward_curves.csv",
        "aggregate_reward_curves": "aggregate_cumulative_reward_curves.csv",
        "seed_window": "seed_window_delivery_rates.csv",
        "aggregate_window": "aggregate_window_delivery_rates.csv",
    }

    for key, filename in optional_files.items():
        path = tables_dir / filename
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            out[key] = None
            warnings.warn(f"Optional table not found: {path}")

    return out


# =============================================================================
# Plot 1: final seed dot plot + mean CI
# =============================================================================


def plot_final_seed_dotplot(
    seed_metrics: pd.DataFrame,
    metric: str,
    output_path: Path | str,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    if metric not in seed_metrics.columns:
        warnings.warn(f"Metric not found in seed_metrics: {metric}")
        return

    paradigms = list(seed_metrics["paradigm"].dropna().unique())
    x = np.arange(len(paradigms))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    means = []
    cis = []

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

        jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.array([0.0])

        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            alpha=0.8,
            s=48,
            zorder=2,
        )

        means.append(values.mean())
        cis.append(ci95(values))

    ax.errorbar(
        x,
        means,
        yerr=cis,
        fmt="o",
        capsize=5,
        linewidth=2.3,
        markersize=8,
        label="Mean ± 95% CI",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, rotation=25, ha="right")
    ax.set_ylabel(ylabel or metric_ylabel(metric))
    ax.set_title(title or short_metric_name(metric))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    savefig(output_path)


# =============================================================================
# Plot 2: multiple final metrics as small panels
# =============================================================================


def plot_metric_grid(
    seed_metrics: pd.DataFrame,
    metrics: list[str],
    output_path: Path | str,
    title: str = "Seed-level metrics with mean ± 95% CI",
) -> None:
    metrics = [m for m in metrics if m in seed_metrics.columns]

    if len(metrics) == 0:
        warnings.warn("No valid metrics provided for metric grid.")
        return

    paradigms = list(seed_metrics["paradigm"].dropna().unique())
    n = len(metrics)

    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11, 4.3 * nrows),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        x = np.arange(len(paradigms))
        means = []
        cis = []

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

            ax.scatter(
                np.full(len(values), i) + jitter,
                values,
                alpha=0.75,
                s=35,
                zorder=2,
            )

            means.append(values.mean())
            cis.append(ci95(values))

        ax.errorbar(
            x,
            means,
            yerr=cis,
            fmt="o",
            capsize=4,
            linewidth=2,
            markersize=6,
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(paradigms, rotation=25, ha="right")
        ax.set_title(short_metric_name(metric))
        ax.set_ylabel(metric_ylabel(metric))
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")

    fig.suptitle(title, y=1.01, fontsize=14)
    savefig(output_path)


# =============================================================================
# Plot 3: clean cumulative soups, no CI ribbon
# =============================================================================


def plot_cumulative_soups_clean(
    seed_curves: pd.DataFrame,
    agg_curves: pd.DataFrame,
    output_path: Path | str,
) -> None:
    required_seed = {"paradigm", "brain_seed", "step", "cumulative_soups"}
    required_agg = {"paradigm", "step", "mean_cumulative_soups"}

    if not required_seed.issubset(seed_curves.columns):
        raise ValueError(f"seed_curves missing columns: {required_seed - set(seed_curves.columns)}")

    if not required_agg.issubset(agg_curves.columns):
        raise ValueError(f"agg_curves missing columns: {required_agg - set(agg_curves.columns)}")

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for paradigm, agg_pg in agg_curves.groupby("paradigm"):
        agg_pg = agg_pg.sort_values("step")

        mean_line, = ax.plot(
            agg_pg["step"],
            agg_pg["mean_cumulative_soups"],
            linewidth=3.0,
            label=f"{paradigm} mean",
            zorder=3,
        )

        color = mean_line.get_color()

        seed_pg = seed_curves[seed_curves["paradigm"] == paradigm]

        for _, seed_g in seed_pg.groupby("brain_seed"):
            seed_g = seed_g.sort_values("step")
            ax.plot(
                seed_g["step"],
                seed_g["cumulative_soups"],
                color=color,
                alpha=0.18,
                linewidth=1.0,
                zorder=1,
            )

    ax.set_xlabel("Timestep within episode")
    ax.set_ylabel("Cumulative soups delivered")
    ax.set_title("Cumulative soup deliveries over episode")
    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(output_path)


# =============================================================================
# Plot 4: cumulative soups with CI only at checkpoints
# =============================================================================


def plot_cumulative_soups_checkpoint_ci(
    agg_curves: pd.DataFrame,
    output_path: Path | str,
    checkpoint_every: int = 250,
) -> None:
    required = {
        "paradigm",
        "step",
        "mean_cumulative_soups",
        "ci95_cumulative_soups",
    }

    if not required.issubset(agg_curves.columns):
        raise ValueError(f"agg_curves missing columns: {required - set(agg_curves.columns)}")

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for paradigm, pg in agg_curves.groupby("paradigm"):
        pg = pg.sort_values("step")

        ax.plot(
            pg["step"],
            pg["mean_cumulative_soups"],
            linewidth=3.0,
            label=f"{paradigm} mean",
        )

        checkpoints = pg[pg["step"] % checkpoint_every == 0]

        ax.errorbar(
            checkpoints["step"],
            checkpoints["mean_cumulative_soups"],
            yerr=checkpoints["ci95_cumulative_soups"],
            fmt="none",
            capsize=3,
            linewidth=1.5,
            alpha=0.85,
        )

    ax.set_xlabel("Timestep within episode")
    ax.set_ylabel("Cumulative soups delivered")
    ax.set_title(f"Cumulative soups with 95% CI every {checkpoint_every} steps")
    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(output_path)


# =============================================================================
# Plot 5: cumulative soups small multiples
# =============================================================================


def plot_cumulative_soups_small_multiples(
    seed_curves: pd.DataFrame,
    agg_curves: pd.DataFrame,
    output_path: Path | str,
) -> None:
    paradigms = list(agg_curves["paradigm"].dropna().unique())
    n = len(paradigms)

    if n == 0:
        warnings.warn("No paradigms found in aggregate curves.")
        return

    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10.5, 4.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for ax, paradigm in zip(axes_flat, paradigms):
        seed_pg = seed_curves[seed_curves["paradigm"] == paradigm]
        agg_pg = agg_curves[agg_curves["paradigm"] == paradigm].sort_values("step")

        mean_line, = ax.plot(
            agg_pg["step"],
            agg_pg["mean_cumulative_soups"],
            linewidth=3.0,
            label="Mean",
            zorder=3,
        )

        color = mean_line.get_color()

        for _, seed_g in seed_pg.groupby("brain_seed"):
            seed_g = seed_g.sort_values("step")
            ax.plot(
                seed_g["step"],
                seed_g["cumulative_soups"],
                color=color,
                alpha=0.25,
                linewidth=1.0,
                zorder=1,
            )

        ax.set_title(paradigm)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative soups")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(paradigms) :]:
        ax.axis("off")

    fig.suptitle("Cumulative soup deliveries by paradigm", y=1.01, fontsize=14)
    savefig(output_path)


# =============================================================================
# Plot 6: delivery rate over episode, clean version
# =============================================================================


def plot_window_delivery_rate_clean(
    seed_window: pd.DataFrame,
    agg_window: pd.DataFrame,
    output_path: Path | str,
) -> None:
    required_seed = {"paradigm", "brain_seed", "window_start", "soups_per_100_steps"}
    required_agg = {"paradigm", "window_start", "mean_soups_per_100_steps"}

    if not required_seed.issubset(seed_window.columns):
        raise ValueError(f"seed_window missing columns: {required_seed - set(seed_window.columns)}")

    if not required_agg.issubset(agg_window.columns):
        raise ValueError(f"agg_window missing columns: {required_agg - set(agg_window.columns)}")

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for paradigm, agg_pg in agg_window.groupby("paradigm"):
        agg_pg = agg_pg.sort_values("window_start")

        mean_line, = ax.plot(
            agg_pg["window_start"],
            agg_pg["mean_soups_per_100_steps"],
            linewidth=3.0,
            label=f"{paradigm} mean",
            zorder=3,
        )

        color = mean_line.get_color()

        seed_pg = seed_window[seed_window["paradigm"] == paradigm]

        for _, seed_g in seed_pg.groupby("brain_seed"):
            seed_g = seed_g.sort_values("window_start")
            ax.plot(
                seed_g["window_start"],
                seed_g["soups_per_100_steps"],
                color=color,
                alpha=0.18,
                linewidth=1.0,
                zorder=1,
            )

    ax.set_xlabel("Timestep window start")
    ax.set_ylabel("Soups per 100 steps")
    ax.set_title("Delivery rate over episode")
    ax.grid(True, alpha=0.3)
    ax.legend()

    savefig(output_path)


# =============================================================================
# Plot 7: difference from baseline, paired by brain_seed
# =============================================================================


def compute_difference_from_baseline(
    seed_metrics: pd.DataFrame,
    metric: str,
    baseline_name: str,
) -> pd.DataFrame:
    if metric not in seed_metrics.columns:
        raise ValueError(f"Metric not found in seed_metrics: {metric}")

    if baseline_name not in set(seed_metrics["paradigm"]):
        raise ValueError(
            f"Baseline paradigm '{baseline_name}' not found. "
            f"Available: {sorted(seed_metrics['paradigm'].dropna().unique())}"
        )

    baseline = (
        seed_metrics[seed_metrics["paradigm"] == baseline_name][["brain_seed", metric]]
        .rename(columns={metric: "baseline_value"})
    )

    rows = []

    for paradigm in seed_metrics["paradigm"].dropna().unique():
        if paradigm == baseline_name:
            continue

        current = (
            seed_metrics[seed_metrics["paradigm"] == paradigm][["brain_seed", metric]]
            .rename(columns={metric: "method_value"})
        )

        merged = pd.merge(current, baseline, on="brain_seed", how="inner")

        if len(merged) == 0:
            warnings.warn(
                f"No shared brain_seed values between {paradigm} and {baseline_name}. "
                "Skipping paired difference."
            )
            continue

        merged["difference"] = merged["method_value"] - merged["baseline_value"]
        merged["paradigm"] = paradigm
        merged["baseline"] = baseline_name
        merged["metric"] = metric

        rows.append(merged)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def plot_difference_from_baseline(
    seed_metrics: pd.DataFrame,
    metric: str,
    baseline_name: str,
    output_path: Path | str,
) -> pd.DataFrame:
    diff_df = compute_difference_from_baseline(
        seed_metrics=seed_metrics,
        metric=metric,
        baseline_name=baseline_name,
    )

    if diff_df.empty:
        warnings.warn(f"No difference plot created for metric={metric}.")
        return diff_df

    paradigms = list(diff_df["paradigm"].dropna().unique())
    x = np.arange(len(paradigms))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    means = []
    cis = []

    for i, paradigm in enumerate(paradigms):
        values = (
            diff_df.loc[diff_df["paradigm"] == paradigm, "difference"]
            .dropna()
            .reset_index(drop=True)
        )

        jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.array([0.0])

        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            alpha=0.8,
            s=48,
            zorder=2,
        )

        means.append(values.mean())
        cis.append(ci95(values))

    ax.errorbar(
        x,
        means,
        yerr=cis,
        fmt="o",
        capsize=5,
        linewidth=2.3,
        markersize=8,
        label="Mean paired difference ± 95% CI",
        zorder=3,
    )

    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, rotation=25, ha="right")
    ax.set_ylabel(f"Difference in {short_metric_name(metric)} vs {baseline_name}")
    ax.set_title(f"Paired difference from {baseline_name}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    savefig(output_path)

    return diff_df


# =============================================================================
# Plot 8: action behavior rates
# =============================================================================


def plot_action_rates_clean(
    seed_metrics: pd.DataFrame,
    output_path: Path | str,
) -> None:
    metrics = ["mean_stay_rate", "mean_interact_rate", "mean_move_rate"]
    labels = ["Stay", "Interact", "Move"]

    missing = [m for m in metrics if m not in seed_metrics.columns]
    if missing:
        warnings.warn(f"Missing action-rate metrics: {missing}")
        return

    paradigms = list(seed_metrics["paradigm"].dropna().unique())
    x = np.arange(len(paradigms))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    for j, metric in enumerate(metrics):
        means = []
        cis = []

        for paradigm in paradigms:
            values = seed_metrics.loc[seed_metrics["paradigm"] == paradigm, metric].dropna()
            means.append(values.mean())
            cis.append(ci95(values))

        positions = x + (j - 1) * width

        ax.bar(
            positions,
            means,
            width=width,
            yerr=cis,
            capsize=4,
            alpha=0.85,
            label=labels[j],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, rotation=25, ha="right")
    ax.set_ylabel("Fraction of primitive actions")
    ax.set_title("Primitive action behavior")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    savefig(output_path)


# =============================================================================
# Main runner
# =============================================================================


def run_clean_plotting(
    tables_dir: Path | str,
    output_dir: Path | str,
    baseline_name: str | None = None,
    checkpoint_every: int = 250,
) -> None:
    tables_dir = Path(tables_dir)
    output_dir = ensure_dir(output_dir)

    print(f"Reading tables from: {tables_dir}")
    tables = load_tables(tables_dir)

    seed_metrics = tables["seed_metrics"]

    print("Creating final seed-level dot plots...")

    main_metrics = [
        "soups_delivered",
        "episode_return",
        "soups_per_100_steps",
        "inactive_tail_length",
        "time_to_first_delivery",
        "joint_semantic_switch_rate",
        "joint_policy_switch_rate",
        "mean_policy_entropy",
        "mean_top_policy_prob",
        "mean_stay_rate",
        "mean_interact_rate",
        "mean_move_rate",
    ]

    for metric in main_metrics:
        if metric in seed_metrics.columns:
            plot_final_seed_dotplot(
                seed_metrics=seed_metrics,
                metric=metric,
                output_path=output_dir / f"{metric}_seed_dots_mean_ci.png",
                ylabel=metric_ylabel(metric),
                title=short_metric_name(metric),
            )

    plot_metric_grid(
        seed_metrics=seed_metrics,
        metrics=[
            "soups_delivered",
            "soups_per_100_steps",
            "inactive_tail_length",
            "joint_semantic_switch_rate",
            "mean_policy_entropy",
            "mean_top_policy_prob",
        ],
        output_path=output_dir / "main_metrics_grid_seed_dots_mean_ci.png",
        title="Main performance and diagnostic metrics",
    )

    print("Creating clean cumulative soup plots...")

    seed_soup_curves = tables["seed_soup_curves"]
    agg_soup_curves = tables["aggregate_soup_curves"]

    if seed_soup_curves is not None and agg_soup_curves is not None:
        plot_cumulative_soups_clean(
            seed_curves=seed_soup_curves,
            agg_curves=agg_soup_curves,
            output_path=output_dir / "cumulative_soups_clean_seed_lines_no_ci_ribbon.png",
        )

        plot_cumulative_soups_checkpoint_ci(
            agg_curves=agg_soup_curves,
            output_path=output_dir / "cumulative_soups_checkpoint_ci.png",
            checkpoint_every=checkpoint_every,
        )

        plot_cumulative_soups_small_multiples(
            seed_curves=seed_soup_curves,
            agg_curves=agg_soup_curves,
            output_path=output_dir / "cumulative_soups_small_multiples.png",
        )

    print("Creating clean delivery-rate plot...")

    seed_window = tables["seed_window"]
    agg_window = tables["aggregate_window"]

    if seed_window is not None and agg_window is not None:
        plot_window_delivery_rate_clean(
            seed_window=seed_window,
            agg_window=agg_window,
            output_path=output_dir / "window_delivery_rate_clean_seed_lines_no_ci_ribbon.png",
        )

    print("Creating action behavior plot...")

    plot_action_rates_clean(
        seed_metrics=seed_metrics,
        output_path=output_dir / "action_rates_clean_mean_ci.png",
    )

    if baseline_name is not None:
        print(f"Creating paired difference plots from baseline: {baseline_name}")

        diff_tables = []

        for metric in [
            "soups_delivered",
            "soups_per_100_steps",
            "episode_return",
            "inactive_tail_length",
            "joint_semantic_switch_rate",
        ]:
            if metric not in seed_metrics.columns:
                continue

            try:
                diff_df = plot_difference_from_baseline(
                    seed_metrics=seed_metrics,
                    metric=metric,
                    baseline_name=baseline_name,
                    output_path=output_dir / f"{metric}_difference_from_{baseline_name}.png",
                )

                if not diff_df.empty:
                    diff_tables.append(diff_df)

            except ValueError as e:
                warnings.warn(str(e))

        if len(diff_tables) > 0:
            all_diff = pd.concat(diff_tables, ignore_index=True)
            all_diff.to_csv(
                output_dir / f"paired_differences_from_{baseline_name}.csv",
                index=False,
            )

    print("\nDone.")
    print(f"Saved clean plots to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overcooked clean plots from overcooked_log_metrics tables/."
    )

    parser.add_argument(
        "--tables-dir",
        type=str,
        required=True,
        help="Path to the tables/ directory produced by overcooked_log_metrics.py.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for clean plots.",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional baseline paradigm name, e.g. PPO. Creates paired difference plots.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Show CI error bars every N timesteps for checkpoint-CI plot.",
    )

    args = parser.parse_args()

    run_clean_plotting(
        tables_dir=args.tables_dir,
        output_dir=args.out,
        baseline_name=args.baseline,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
