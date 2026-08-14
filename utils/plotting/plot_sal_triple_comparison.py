"""
Compare all three SAL Overcooked paradigms (Independent, Individually Collective,
Fully Collective) on matched episode seeds -- the three-way generalization of
plot_sal_pair_comparison.py.

Runs are paired by episode_seed, restricted to seeds present in ALL THREE
conditions. Metrics use team_reward = max(reward_a0, reward_a1) per step to
avoid double-counting shared delivery credit (same convention as
plot_sal_pair_comparison.py / overcooked_log_metrics).

Usage (explicit directories, preferred -- avoids ambiguity if a base dir has
multiple candidate runs per paradigm):

    python3 utils/plotting/plot_sal_triple_comparison.py \\
        --ind thesis_logs/03_ma_overcooked/sal_ind_30seed \\
        --ic  thesis_logs/03_ma_overcooked/sal_ic_XXseed \\
        --fc  thesis_logs/03_ma_overcooked/sal_fc_30seed_collisionfix \\
        -o thesis_plots/03_ma_overcooked/compare_ind_ic_fc

Usage (auto-discover): scans immediate subdirectories of --base-dir for SAL
CSVs matching each paradigm's filename pattern (sal_ind_*.csv / sal_ic_*.csv /
sal_fc_*.csv). If more than one subdirectory matches a paradigm, the most
recently modified one is used and all candidates are printed so you can
re-run with an explicit --ind/--ic/--fc override if the wrong one was picked.

    python3 utils/plotting/plot_sal_triple_comparison.py \\
        --base-dir thesis_logs/03_ma_overcooked \\
        -o thesis_plots/03_ma_overcooked/compare_ind_ic_fc
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

from plot_sal_semantic_action_level import (  # noqa: E402
    PARADIGM_PATTERNS,
    PARADIGM_TITLES,
    W_CURVE,
    load_seed_csvs,
    rolling_mean,
)

DELIVERY_REWARD = 20.0
DPI = 150
PARADIGM_ORDER = ("ind", "ic", "fc")
COLORS = {"ind": "#E67E22", "ic": "#2E86AB", "fc": "#27AE60"}


def ci95(x) -> float:
    x = pd.Series(x).dropna()
    n = len(x)
    if n <= 1:
        return np.nan
    tcrit_table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    }
    tcrit = tcrit_table.get(n, 1.96)
    return tcrit * x.std(ddof=1) / np.sqrt(n)


def prepare_team_rewards(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("step").copy()
    g["team_reward"] = g[["reward_a0", "reward_a1"]].max(axis=1)
    g["cumulative_soups"] = g["team_reward"].cumsum() / DELIVERY_REWARD
    return g


def run_metrics(df: pd.DataFrame) -> dict:
    g = prepare_team_rewards(df)
    ep_seed = int(g["episode_seed"].iloc[0])
    episode_return = float(g["team_reward"].sum())
    soups = episode_return / DELIVERY_REWARD
    return {
        "episode_seed": ep_seed,
        "episode_return": episode_return,
        "soups_delivered": soups,
        "any_delivery": soups > 0,
        "horizon": int(g["step"].max()),
    }


def runs_dataframe(seed_dfs: list[pd.DataFrame], condition_label: str) -> pd.DataFrame:
    out = pd.DataFrame([run_metrics(df) for df in seed_dfs])
    out["condition"] = condition_label
    return out


def cumulative_soup_curves(seed_dfs: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    prepared = [prepare_team_rewards(df) for df in seed_dfs]
    min_step = min(int(df["step"].min()) for df in prepared)
    max_step = max(int(df["step"].max()) for df in prepared)
    steps = np.arange(min_step, max_step + 1)
    mat = np.zeros((len(prepared), len(steps)))
    for i, g in enumerate(prepared):
        s = g.set_index("step")["cumulative_soups"].reindex(steps).ffill().fillna(0.0)
        mat[i] = s.values
    return steps, mat


def aggregate_curve(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = mat.mean(axis=0)
    err = np.array([ci95(mat[:, t]) for t in range(mat.shape[1])])
    return mean, err


def smooth_curve(mean: np.ndarray, err: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    if w <= 1:
        return mean, err
    return rolling_mean(np.asarray(mean, dtype=float), w), rolling_mean(np.asarray(err, dtype=float), w)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def find_common_seeds(runs: dict[str, pd.DataFrame]) -> list[int]:
    sets = [set(df["episode_seed"]) for df in runs.values()]
    common = sorted(set.intersection(*sets))
    if not common:
        raise ValueError(
            "No episode_seed is common to all three conditions. "
            f"Per-condition seeds: { {k: sorted(v['episode_seed']) for k, v in runs.items()} }"
        )
    return common


def plot_cumulative_soups(steps, curves: dict, labels: dict, path: Path, smoothing_window: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in PARADIGM_ORDER:
        if key not in curves:
            continue
        mean, err = curves[key]
        ax.plot(steps, mean, lw=2.5, color=COLORS[key], label=labels[key])
        if np.any(np.isfinite(err)):
            ax.fill_between(steps, mean - err, mean + err, color=COLORS[key], alpha=0.18)
    ax.set_xlabel("Timestep within episode")
    ax.set_ylabel("Cumulative soups delivered (mean across matched runs)")
    title = "Three-way comparison: cumulative deliveries"
    if smoothing_window > 1:
        title += f" (smoothed, window={smoothing_window})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, path)


def plot_soups_bar(runs: dict[str, pd.DataFrame], labels: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = [k for k in PARADIGM_ORDER if k in runs]
    means = [runs[k]["soups_delivered"].mean() for k in keys]
    errs = [ci95(runs[k]["soups_delivered"]) for k in keys]
    colors = [COLORS[k] for k in keys]
    x = list(range(len(keys)))
    ax.bar(x, means, yerr=errs, capsize=6, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys])
    ax.set_ylabel("Soups delivered per episode")
    n = len(next(iter(runs.values())))
    ax.set_title(f"Mean ± 95% CI (n={n} matched runs)")
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, path)


def plot_paired_lines(paired: pd.DataFrame, labels: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    keys = [k for k in PARADIGM_ORDER if f"soups_{k}" in paired.columns]
    xs = list(range(len(keys)))
    for _, row in paired.iterrows():
        ys = [row[f"soups_{k}"] for k in keys]
        ax.plot(xs, ys, "o-", color="#888888", alpha=0.5, lw=1.2, markersize=6, zorder=2)
        ax.text(-0.08, ys[0], str(int(row["episode_seed"])), ha="right", va="center", fontsize=7, color=COLORS[keys[0]])
    for i, k in enumerate(keys):
        ax.scatter([i] * len(paired), paired[f"soups_{k}"], s=70, color=COLORS[k], label=labels[k], zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[k] for k in keys])
    ax.set_ylabel("Soups delivered")
    ax.set_title("Paired by episode_seed")
    ax.set_xlim(-0.4, len(keys) - 0.6)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, path)


def plot_per_seed_grouped_bar(paired: pd.DataFrame, labels: dict, path: Path) -> None:
    keys = [k for k in PARADIGM_ORDER if f"soups_{k}" in paired.columns]
    n_seeds = len(paired)
    n_cond = len(keys)
    width = 0.8 / n_cond
    fig, ax = plt.subplots(figsize=(max(9, n_seeds * 0.9), 5))
    x = np.arange(n_seeds)
    for i, k in enumerate(keys):
        offset = (i - (n_cond - 1) / 2) * width
        ax.bar(x + offset, paired[f"soups_{k}"], width=width, color=COLORS[k], edgecolor="black", alpha=0.85, label=labels[k])
    ax.set_xticks(x)
    ax.set_xticklabels(paired["episode_seed"].astype(int).astype(str))
    ax.set_xlabel("Episode seed")
    ax.set_ylabel("Soups delivered")
    ax.set_title("Per-seed comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, path)


def print_summary(paired: pd.DataFrame, labels: dict) -> None:
    keys = [k for k in PARADIGM_ORDER if f"soups_{k}" in paired.columns]
    print("\n" + "=" * 70)
    print(f"THREE-WAY COMPARISON: {' vs '.join(labels[k] for k in keys)}")
    print("=" * 70)
    print(f"Matched episode_seeds (n={len(paired)}): {list(paired['episode_seed'])}")
    for k in keys:
        col = paired[f"soups_{k}"]
        print(f"Mean soups  {labels[k]:>24s}: {col.mean():.3f} +/- {col.std(ddof=1):.3f}")
    print("\nWin counts (highest soups on that seed; ties count for all tied conditions):")
    wins = {k: 0 for k in keys}
    for _, row in paired.iterrows():
        vals = {k: row[f"soups_{k}"] for k in keys}
        best = max(vals.values())
        for k, v in vals.items():
            if v == best:
                wins[k] += 1
    for k in keys:
        print(f"  {labels[k]:>24s}: best on {wins[k]}/{len(paired)} seeds")


def resolve_auto_dirs(base_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for paradigm in PARADIGM_ORDER:
        pattern = PARADIGM_PATTERNS[paradigm]
        candidates = sorted(
            {p.parent for p in base_dir.glob(f"*/{pattern}")},
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"WARNING: no subdirectory of {base_dir} matches paradigm '{paradigm}' (pattern {pattern}) -- skipping")
            continue
        if len(candidates) > 1:
            print(f"NOTE: multiple candidate directories for '{paradigm}', using the most recently modified:")
            for c in candidates:
                print(f"    {'-> ' if c == candidates[0] else '   '}{c}")
        resolved[paradigm] = candidates[0]
    return resolved


def run_comparison(dirs: dict[str, Path], output_dir: Path, labels_override: dict[str, str], smoothing_window: int) -> None:
    output_dir = Path(output_dir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    dfs: dict[str, list[pd.DataFrame]] = {}
    labels: dict[str, str] = {}
    for key, d in dirs.items():
        seed_dfs, paradigm, _ = load_seed_csvs(d, key)
        dfs[key] = seed_dfs
        labels[key] = labels_override.get(key) or PARADIGM_TITLES.get(paradigm, paradigm.upper())

    runs = {k: runs_dataframe(v, labels[k]) for k, v in dfs.items()}
    common_seeds = find_common_seeds(runs)
    print(f"\nCommon episode_seeds across all {len(dirs)} conditions: {common_seeds}")

    paired_rows = []
    for ep in common_seeds:
        row = {"episode_seed": ep}
        for k in dirs:
            r = runs[k].set_index("episode_seed").loc[ep]
            row[f"soups_{k}"] = r["soups_delivered"]
            row[f"return_{k}"] = r["episode_return"]
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)

    dfs_matched = {
        k: [df for df in v if int(df["episode_seed"].iloc[0]) in set(common_seeds)]
        for k, v in dfs.items()
    }
    curves = {}
    steps_ref = None
    for k, seed_dfs in dfs_matched.items():
        steps, mat = cumulative_soup_curves(seed_dfs)
        if steps_ref is None or len(steps) < len(steps_ref):
            steps_ref = steps
        curves[k] = (steps, mat)

    # Align all conditions to the shortest common step grid before aggregating
    aligned_curves = {}
    for k, (steps, mat) in curves.items():
        mat_trim = mat[:, : len(steps_ref)]
        mean, err = aggregate_curve(mat_trim)
        mean, err = smooth_curve(mean, err, smoothing_window)
        aligned_curves[k] = (mean, err)

    runs_matched = {
        k: runs[k][runs[k]["episode_seed"].isin(common_seeds)].sort_values("episode_seed")
        for k in dirs
    }

    print_summary(paired, labels)

    paired.to_csv(output_dir / "tables" / "matched_by_episode_seed.csv", index=False)
    for k in dirs:
        runs_matched[k].to_csv(output_dir / "tables" / f"runs_{k}.csv", index=False)

    print(f"\nSaving plots -> {output_dir / 'plots'}")
    plot_cumulative_soups(
        steps_ref, aligned_curves, labels,
        output_dir / "plots" / "compare_cumulative_soups.png",
        smoothing_window=smoothing_window,
    )
    plot_soups_bar(runs_matched, labels, output_dir / "plots" / "compare_mean_soups.png")
    plot_paired_lines(paired, labels, output_dir / "plots" / "compare_paired_soups.png")
    plot_per_seed_grouped_bar(paired, labels, output_dir / "plots" / "compare_per_seed_soups.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way SAL Overcooked comparison (matched episode_seed)")
    parser.add_argument("--ind", type=Path, default=None, help="Independent logs directory")
    parser.add_argument("--ic", type=Path, default=None, help="Individually Collective logs directory")
    parser.add_argument("--fc", type=Path, default=None, help="Fully Collective logs directory")
    parser.add_argument("--base-dir", type=Path, default=None, help="Auto-discover --ind/--ic/--fc as subdirectories of this path")
    parser.add_argument("--label-ind", default=None)
    parser.add_argument("--label-ic", default=None)
    parser.add_argument("--label-fc", default=None)
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--smooth-window", type=int, default=W_CURVE,
        help=f"Rolling window for compare_cumulative_soups (default: {W_CURVE}; use 1 for raw)",
    )
    args = parser.parse_args()

    dirs: dict[str, Path] = {}
    if args.base_dir is not None:
        dirs.update(resolve_auto_dirs(args.base_dir))
    for key, val in (("ind", args.ind), ("ic", args.ic), ("fc", args.fc)):
        if val is not None:
            dirs[key] = val  # explicit flag always overrides auto-discovery

    if len(dirs) < 2:
        parser.error(
            "Need at least two of --ind/--ic/--fc (directly or via --base-dir auto-discovery) to compare."
        )
    missing = [k for k in PARADIGM_ORDER if k not in dirs]
    if missing:
        print(f"NOTE: proceeding without {missing} -- comparison will only cover {list(dirs)}")

    labels_override = {"ind": args.label_ind, "ic": args.label_ic, "fc": args.label_fc}
    run_comparison(dirs, args.output_dir, labels_override, args.smooth_window)
    print(f"\nDone. Tables in {args.output_dir / 'tables'}, plots in {args.output_dir / 'plots'}")


if __name__ == "__main__":
    main()
