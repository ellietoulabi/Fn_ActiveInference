"""
Compare two SAL Overcooked conditions (e.g. FC vs IND) on matched episode seeds.

Runs are paired by episode_seed (SLURM array index: ep 76–85). Metrics use
team_reward = max(reward_a0, reward_a1) per step to avoid double-counting
shared delivery credit (same as overcooked_log_metrics).

Usage:
    python utils/plotting/plot_sal_pair_comparison.py \\
        --a logs/sal_fc --b logs/sal_ind \\
        --label-a FC --label-b IND \\
        -o results/Overcooked/compare_fc_ind

    python utils/plotting/plot_sal_pair_comparison.py \\
        logs/sal_fc logs/sal_ind -o results/Overcooked/compare_fc_ind
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
    PARADIGM_TITLES,
    detect_paradigm,
    load_seed_csvs,
)

DELIVERY_REWARD = 20.0
DPI = 150
COLORS = ("#2E86AB", "#E67E22")


def ci95(x):
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
    g["cumulative_team_reward"] = g["team_reward"].cumsum()
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
    rows = [run_metrics(df) for df in seed_dfs]
    out = pd.DataFrame(rows)
    out["condition"] = condition_label
    return out


def cumulative_soup_curves(seed_dfs: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return steps, (n_runs, n_steps) soups matrix, episode_seeds order."""
    prepared = [prepare_team_rewards(df) for df in seed_dfs]
    ep_seeds = [int(df["episode_seed"].iloc[0]) for df in prepared]
    min_step = min(int(df["step"].min()) for df in prepared)
    max_step = max(int(df["step"].max()) for df in prepared)
    steps = np.arange(min_step, max_step + 1)

    mat = np.zeros((len(prepared), len(steps)))
    for i, g in enumerate(prepared):
        s = (
            g.set_index("step")["cumulative_soups"]
            .reindex(steps)
            .ffill()
            .fillna(0.0)
        )
        mat[i] = s.values
    return steps, mat, np.array(ep_seeds)


def aggregate_curve(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = mat.mean(axis=0)
    err = np.array([ci95(mat[:, t]) for t in range(mat.shape[1])])
    return mean, err


def paired_table(runs_a: pd.DataFrame, runs_b: pd.DataFrame) -> pd.DataFrame:
    a = runs_a.set_index("episode_seed")
    b = runs_b.set_index("episode_seed")
    common = sorted(set(a.index) & set(b.index))
    if not common:
        raise ValueError("No matching episode_seed between conditions")

    rows = []
    for ep in common:
        rows.append({
            "episode_seed": ep,
            "soups_a": a.loc[ep, "soups_delivered"],
            "soups_b": b.loc[ep, "soups_delivered"],
            "return_a": a.loc[ep, "episode_return"],
            "return_b": b.loc[ep, "episode_return"],
            "soups_diff_b_minus_a": b.loc[ep, "soups_delivered"] - a.loc[ep, "soups_delivered"],
        })
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cumulative_soups(
    steps: np.ndarray,
    mean_a: np.ndarray,
    err_a: np.ndarray,
    mean_b: np.ndarray,
    err_b: np.ndarray,
    label_a: str,
    label_b: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for mean, err, label, color in (
        (mean_a, err_a, label_a, COLORS[0]),
        (mean_b, err_b, label_b, COLORS[1]),
    ):
        ax.plot(steps, mean, lw=2.5, color=color, label=label)
        if np.any(np.isfinite(err)):
            ax.fill_between(steps, mean - err, mean + err, color=color, alpha=0.2)
    ax.set_xlabel("Timestep within episode")
    ax.set_ylabel("Cumulative soups delivered (mean across matched runs)")
    ax.set_title("Paired comparison: cumulative deliveries")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, path)


def plot_soups_bar(
    runs_a: pd.DataFrame,
    runs_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    means = [runs_a["soups_delivered"].mean(), runs_b["soups_delivered"].mean()]
    errs = [ci95(runs_a["soups_delivered"]), ci95(runs_b["soups_delivered"])]
    x = [0, 1]
    ax.bar(x, means, yerr=errs, capsize=6, color=COLORS, alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Soups delivered per episode")
    ax.set_title(f"Mean ± 95% CI (n={len(runs_a)} matched runs)")
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, path)


def plot_paired_dots(paired: pd.DataFrame, label_a: str, label_b: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ep = paired["episode_seed"].values
    a = paired["soups_a"].values
    b = paired["soups_b"].values

    for i, e in enumerate(ep):
        ax.plot([0, 1], [a[i], b[i]], "o-", color="#888888", alpha=0.6, lw=1.2, markersize=6)
        ax.text(-0.06, a[i], str(int(e)), ha="right", va="center", fontsize=7, color=COLORS[0])
    ax.scatter(np.zeros(len(a)), a, s=70, color=COLORS[0], label=label_a, zorder=3)
    ax.scatter(np.ones(len(b)), b, s=70, color=COLORS[1], label=label_b, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Soups delivered")
    ax.set_title("Paired by episode_seed")
    ax.set_xlim(-0.3, 1.3)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, path)


def plot_paired_difference(paired: pd.DataFrame, label_a: str, label_b: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ep = paired["episode_seed"].values
    diff = paired["soups_diff_b_minus_a"].values
    colors = [COLORS[1] if d >= 0 else COLORS[0] for d in diff]
    ax.bar(ep.astype(str), diff, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Episode seed")
    ax.set_ylabel(f"Soups ({label_b} − {label_a})")
    ax.set_title("Per-seed difference")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    savefig(fig, path)


def print_summary(paired: pd.DataFrame, label_a: str, label_b: str) -> None:
    diff = paired["soups_diff_b_minus_a"]
    print("\n" + "=" * 70)
    print(f"PAIRED COMPARISON: {label_a}  vs  {label_b}")
    print("=" * 70)
    print(f"Matched episode_seeds (n={len(paired)}): {list(paired['episode_seed'])}")
    print(f"\nMean soups  {label_a}: {paired['soups_a'].mean():.3f} ± {paired['soups_a'].std(ddof=1):.3f}")
    print(f"Mean soups  {label_b}: {paired['soups_b'].mean():.3f} ± {paired['soups_b'].std(ddof=1):.3f}")
    print(f"Mean diff ({label_b}−{label_a}): {diff.mean():.3f}")
    print(f"{label_b} better on {int((diff > 0).sum())}/{len(paired)} seeds")
    print(f"{label_a} better on {int((diff < 0).sum())}/{len(paired)} seeds")


def run_comparison(
    logs_dir_a: Path,
    logs_dir_b: Path,
    output_dir: Path,
    label_a: str | None = None,
    label_b: str | None = None,
    paradigm_a: str | None = None,
    paradigm_b: str | None = None,
) -> None:
    output_dir = Path(output_dir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    dfs_a, par_a, _ = load_seed_csvs(logs_dir_a, paradigm_a)
    dfs_b, par_b, _ = load_seed_csvs(logs_dir_b, paradigm_b)

    label_a = label_a or PARADIGM_TITLES.get(par_a, par_a.upper())
    label_b = label_b or PARADIGM_TITLES.get(par_b, par_b.upper())

    runs_a = runs_dataframe(dfs_a, label_a)
    runs_b = runs_dataframe(dfs_b, label_b)
    paired = paired_table(runs_a, runs_b)

    # Restrict curves to matched episode seeds only
    match_eps = set(paired["episode_seed"])
    dfs_a_m = [df for df in dfs_a if int(df["episode_seed"].iloc[0]) in match_eps]
    dfs_b_m = [df for df in dfs_b if int(df["episode_seed"].iloc[0]) in match_eps]

    steps_a, mat_a, _ = cumulative_soup_curves(dfs_a_m)
    steps_b, mat_b, _ = cumulative_soup_curves(dfs_b_m)
    if len(steps_a) != len(steps_b) or not np.array_equal(steps_a, steps_b):
        print("WARNING: step grids differ between conditions; using shared min–max intersection")
        steps = np.arange(max(steps_a[0], steps_b[0]), min(steps_a[-1], steps_b[-1]) + 1)
        # reindex mats — keep simple: use shorter logic
        steps = steps_a if len(steps_a) <= len(steps_b) else steps_b
        mat_a = mat_a[:, : len(steps)]
        mat_b = mat_b[:, : len(steps)]

    mean_a, err_a = aggregate_curve(mat_a)
    mean_b, err_b = aggregate_curve(mat_b)

    runs_a_m = runs_a[runs_a["episode_seed"].isin(match_eps)].sort_values("episode_seed")
    runs_b_m = runs_b[runs_b["episode_seed"].isin(match_eps)].sort_values("episode_seed")

    print_summary(paired, label_a, label_b)

    paired.to_csv(output_dir / "tables" / "paired_by_episode_seed.csv", index=False)
    runs_a_m.to_csv(output_dir / "tables" / f"runs_{par_a}.csv", index=False)
    runs_b_m.to_csv(output_dir / "tables" / f"runs_{par_b}.csv", index=False)

    print(f"\nSaving plots → {output_dir / 'plots'}")
    plot_cumulative_soups(
        steps_a, mean_a, err_a, mean_b, err_b, label_a, label_b,
        output_dir / "plots" / "compare_cumulative_soups.png",
    )
    plot_soups_bar(runs_a_m, runs_b_m, label_a, label_b, output_dir / "plots" / "compare_mean_soups.png")
    plot_paired_dots(paired, label_a, label_b, output_dir / "plots" / "compare_paired_soups.png")
    plot_paired_difference(paired, label_a, label_b, output_dir / "plots" / "compare_soups_difference.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise SAL Overcooked comparison (matched episode_seed)")
    parser.add_argument("logs_a", nargs="?", default=None, help="First logs directory")
    parser.add_argument("logs_b", nargs="?", default=None, help="Second logs directory")
    parser.add_argument("--a", dest="logs_a_flag", type=Path, default=None, help="First logs directory")
    parser.add_argument("--b", dest="logs_b_flag", type=Path, default=None, help="Second logs directory")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--paradigm-a", choices=["fc", "ind", "ic", "mappo"], default=None)
    parser.add_argument("--paradigm-b", choices=["fc", "ind", "ic", "mappo"], default=None)
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    logs_a = args.logs_a_flag or args.logs_a
    logs_b = args.logs_b_flag or args.logs_b
    if logs_a is None or logs_b is None:
        parser.error("Provide two log directories: positional or --a/--b")

    run_comparison(
        Path(logs_a),
        Path(logs_b),
        args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        paradigm_a=args.paradigm_a,
        paradigm_b=args.paradigm_b,
    )
    print(f"\nDone. Tables in {args.output_dir / 'tables'}, plots in {args.output_dir / 'plots'}")


if __name__ == "__main__":
    main()
