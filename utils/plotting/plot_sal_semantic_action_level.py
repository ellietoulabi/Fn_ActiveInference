"""
Plot semantic action-level (SAL) Overcooked results from per-seed step CSV logs.

Each CSV is one evaluation episode (one seed), one row per env step. Produces
RL-paper-style figures (learning curves with bootstrap CI, success rate, ECDFs,
return distributions) and saves each as a separate PNG.

Usage:
    python utils/plotting/plot_sal_semantic_action_level.py logs/sal_fc
    python utils/plotting/plot_sal_semantic_action_level.py logs/sal_fc -o results/plots_sal_fc
    python utils/plotting/plot_sal_semantic_action_level.py logs/sal_ind --paradigm ind
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARADIGM_PATTERNS = {
    "fc": "sal_fc_*.csv",
    "ind": "sal_ind_*.csv",
    "ic": "sal_ic_*.csv",
    "mappo": "sal_mappo_*.csv",
}

PARADIGM_TITLES = {
    "fc": "Fully Collective",
    "ind": "Independent",
    "ic": "Individually Collective",
    "mappo": "MAPPO",
}

# RL plotting defaults (aligned with plot_sa_redbluebuttons_nine.py)
W_CURVE = 50
W_STABLE = 50
THETA = 0.8
BOOTSTRAP_N = 1000
CI_PERCENT = 95
AGG_COLOR = "#2E86AB"
DPI = 150


def detect_paradigm(logs_dir: Path) -> str:
    for paradigm, pattern in PARADIGM_PATTERNS.items():
        if list(logs_dir.glob(pattern)):
            return paradigm
    raise FileNotFoundError(
        f"No SAL CSV files found in {logs_dir}. "
        f"Expected sal_fc_*.csv, sal_ind_*.csv, sal_ic_*.csv, or sal_mappo_*.csv"
    )


def seed_key_from_path(path: Path, paradigm: str) -> tuple:
    name = path.name
    if paradigm == "fc":
        m = re.search(r"sal_fc_ep(\d+)_brain(\d+)_", name)
        if m:
            return ("fc", int(m.group(1)), int(m.group(2)))
    else:
        m = re.search(r"sal_(?:ind|ic|mappo)_ep(\d+)_a0_(\d+)_a1_(\d+)_", name)
        if m:
            return (paradigm, int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return (paradigm, name)


def seed_label(key: tuple) -> str:
    if key[0] == "fc":
        return f"ep{key[1]}/brain{key[2]}"
    return f"ep{key[1]}/a0={key[2]}/a1={key[3]}"


def warn_missing_expected_runs(
    paradigm: str,
    loaded_keys: list[tuple],
    *,
    episode_start: int = 76,
    n_runs: int = 10,
) -> None:
    """Warn when SLURM array seeds (e.g. ep 76–85) are not all present in logs/."""
    if paradigm == "fc":
        expected = {(episode_start + i, 1000 + i) for i in range(n_runs)}
        loaded = {(k[1], k[2]) for k in loaded_keys if k[0] == "fc" and len(k) >= 3}
        missing = sorted(expected - loaded)
        if not missing:
            return
        print("\nWARNING: expected 10 FC runs (ep 76–85 / brain 1000–1009) but some are missing:")
        for ep, brain in missing:
            print(f"  missing: episode_seed={ep}, brain_seed={brain}  (look for sal_fc_ep{ep}_brain{brain}_*.csv)")
        print(f"  loaded {len(loaded)}/{n_runs} runs — plots use only what is on disk.\n")
    elif paradigm in ("ind", "ic", "mappo"):
        expected = {(episode_start + i, 1000 + i, 2000 + i) for i in range(n_runs)}
        loaded = {
            (k[1], k[2], k[3])
            for k in loaded_keys
            if k[0] == paradigm and len(k) >= 4
        }
        missing = sorted(expected - loaded)
        if not missing:
            return
        print(f"\nWARNING: expected {n_runs} {paradigm} runs but {len(missing)} file(s) missing:")
        for ep, a0, a1 in missing:
            print(f"  missing: ep={ep}, a0={a0}, a1={a1}")
        print(f"  loaded {len(loaded)}/{n_runs} runs.\n")


def load_seed_csvs(logs_dir: Path, paradigm: str | None = None) -> tuple[list[pd.DataFrame], str, list[str]]:
    logs_dir = Path(logs_dir)
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {logs_dir}")

    if paradigm is None:
        paradigm = detect_paradigm(logs_dir)

    pattern = PARADIGM_PATTERNS[paradigm]
    files = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {logs_dir}")

    latest_by_seed: dict[tuple, Path] = {}
    for path in files:
        latest_by_seed[seed_key_from_path(path, paradigm)] = path

    seed_dfs: list[pd.DataFrame] = []
    labels: list[str] = []
    for key in sorted(latest_by_seed):
        path = latest_by_seed[key]
        print(f"Loading: {path.name}")
        df = pd.read_csv(path)
        seed_dfs.append(df.sort_values("step").reset_index(drop=True))
        labels.append(seed_label(key))

    print(f"Loaded {len(seed_dfs)} run file(s) for paradigm={paradigm}")
    warn_missing_expected_runs(paradigm, list(latest_by_seed.keys()))
    return seed_dfs, paradigm, labels


def episode_summary(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    total_a0 = float(last["cumulative_reward_a0"])
    total_a1 = float(last["cumulative_reward_a1"])
    total = total_a0 + total_a1
    steps = int(last["step"])
    terminated = bool(last["terminated"])
    return {
        "steps": steps,
        "total_reward_a0": total_a0,
        "total_reward_a1": total_a1,
        "total_reward": total,
        "terminated": terminated,
        "success": terminated or total > 0,
    }


def build_step_series(seed_dfs: list[pd.DataFrame]) -> dict[str, np.ndarray]:
    max_steps = max(int(df["step"].max()) for df in seed_dfs)
    n_seeds = len(seed_dfs)
    steps = np.arange(1, max_steps + 1)

    cum_team = np.full((n_seeds, max_steps), np.nan)
    step_team = np.full((n_seeds, max_steps), np.nan)
    delivered = np.full((n_seeds, max_steps), np.nan)

    for i, df in enumerate(seed_dfs):
        for _, row in df.iterrows():
            s = int(row["step"]) - 1
            if s < 0 or s >= max_steps:
                continue
            r = float(row["reward_a0"]) + float(row["reward_a1"])
            cum_team[i, s] = float(row["cumulative_reward_a0"]) + float(row["cumulative_reward_a1"])
            step_team[i, s] = r
            delivered[i, s] = 1.0 if r > 0 else 0.0

    return {"steps": steps, "cum_team": cum_team, "step_team": step_team, "delivered": delivered}


def bootstrap_ci(
    curves: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    ci_percent: float = CI_PERCENT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_seeds, _ = curves.shape
    rng = np.random.default_rng(42)
    low = (100 - ci_percent) / 2
    high = 100 - low
    boot = np.zeros((n_bootstrap, curves.shape[1]))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot[b] = np.nanmean(curves[idx], axis=0)
    mean = np.nanmean(curves, axis=0)
    ci_low = np.nanpercentile(boot, low, axis=0)
    ci_high = np.nanpercentile(boot, high, axis=0)
    return mean, ci_low, ci_high


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - w + 1)
        out[i] = np.nanmean(x[start : i + 1])
    return out


def smooth_triple(mean: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray, w: int) -> None:
    if w <= 1:
        return
    mean[:] = rolling_mean(mean, w)
    ci_low[:] = rolling_mean(ci_low, w)
    ci_high[:] = rolling_mean(ci_high, w)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def _draw_ci(ax, steps: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray, valid: np.ndarray) -> None:
    if np.any(ci_high[valid] - ci_low[valid] > 1e-9):
        ax.fill_between(steps[valid], ci_low[valid], ci_high[valid], color=AGG_COLOR, alpha=0.25, zorder=2)


def _title_suffix(n_runs: int) -> str:
    return " (1 run — no CI)" if n_runs == 1 else f" ({CI_PERCENT}% bootstrap CI, n={n_runs} runs)"


def time_to_first_delivery(df: pd.DataFrame) -> float:
    team = df["reward_a0"].values + df["reward_a1"].values
    hits = np.nonzero(team > 0)[0]
    if len(hits) == 0:
        return np.inf
    return float(df.iloc[hits[0]]["step"])


def time_to_stable_delivery(df: pd.DataFrame, w_stable: int, theta: float) -> float:
    delivered = (df["reward_a0"].values + df["reward_a1"].values > 0).astype(float)
    steps = df["step"].values
    for i in range(w_stable - 1, len(delivered)):
        if np.mean(delivered[i - w_stable + 1 : i + 1]) >= theta:
            return float(steps[i])
    return np.inf


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.array([0.0]), np.array([0.0])
    xs = np.sort(np.unique(finite))
    counts = np.searchsorted(np.sort(finite), xs, side="right")
    frac = counts / len(values)
    return xs, frac


def run_title(paradigm: str, n_runs: int) -> str:
    return f"SAL Overcooked — {PARADIGM_TITLES.get(paradigm, paradigm)} ({n_runs} runs)"


# ---------------------------------------------------------------------------
# Individual plot functions (one file each)
# ---------------------------------------------------------------------------


def plot_learning_curve_cumulative_return(
    series: dict,
    output_path: Path,
    paradigm: str,
    n_seeds: int,
    smoothing_window: int,
) -> None:
    """RL standard: mean cumulative return vs environment steps."""
    steps = series["steps"]
    mean, ci_low, ci_high = bootstrap_ci(series["cum_team"])
    if smoothing_window > 1:
        smooth_triple(mean, ci_low, ci_high, smoothing_window)
    valid = ~np.isnan(mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps[valid], mean[valid], color=AGG_COLOR, lw=2.5, label="Mean", zorder=3)
    _draw_ci(ax, steps, ci_low, ci_high, valid)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Mean cumulative team return")
    ax.set_title("Learning curve: cumulative return" + _title_suffix(n_seeds))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, steps.max() + 1)
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_learning_curve_step_return(
    series: dict,
    output_path: Path,
    paradigm: str,
    n_seeds: int,
    smoothing_window: int,
) -> None:
    """RL standard: mean per-step return (smoothed) vs steps."""
    steps = series["steps"]
    mean, ci_low, ci_high = bootstrap_ci(series["step_team"])
    if smoothing_window > 1:
        smooth_triple(mean, ci_low, ci_high, smoothing_window)
    valid = ~np.isnan(mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps[valid], mean[valid], color="#E67E22", lw=2.5, label="Mean step return", zorder=3)
    if np.any(ci_high[valid] - ci_low[valid] > 1e-9):
        ax.fill_between(steps[valid], ci_low[valid], ci_high[valid], color="#E67E22", alpha=0.25)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Mean team reward (per step)")
    ax.set_title(f"Learning curve: step return (window={smoothing_window})" + _title_suffix(n_seeds))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, steps.max() + 1)
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_fraction_delivered_by_step(
    series: dict,
    output_path: Path,
    paradigm: str,
    n_seeds: int,
    smoothing_window: int,
) -> None:
    """RL standard: fraction of seeds with ≥1 delivery by step t."""
    cum = series["cum_team"]
    success_by_step = np.nanmean((cum > 0).astype(float), axis=0)
    steps = series["steps"]
    if smoothing_window > 1:
        success_by_step = rolling_mean(success_by_step, smoothing_window)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, success_by_step, color="#06A77D", lw=2.5)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Fraction of seeds with delivery")
    ax.set_title("Success rate over steps (≥1 delivery so far)" + _title_suffix(n_seeds))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, steps.max() + 1)
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_ecdf_first_delivery(
    seed_dfs: list[pd.DataFrame],
    output_path: Path,
    paradigm: str,
    n_seeds: int,
) -> None:
    """RL standard: ECDF of time to first soup delivery."""
    tau = np.array([time_to_first_delivery(df) for df in seed_dfs])
    xs, ys = ecdf(tau)
    max_step = max(int(df["step"].max()) for df in seed_dfs)

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(xs) > 0:
        ax.step(np.r_[0, xs, max_step + 1], np.r_[0, ys, ys[-1]], where="post", color=AGG_COLOR, lw=2)
    n_censored = int(np.sum(~np.isfinite(tau)))
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Fraction of seeds with first delivery")
    ax.set_title(f"ECDF: time to first delivery (censored: {n_censored}/{n_seeds})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max_step + 1)
    ax.grid(True, alpha=0.3)
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_ecdf_stable_delivery(
    seed_dfs: list[pd.DataFrame],
    output_path: Path,
    paradigm: str,
    n_seeds: int,
    w_stable: int,
    theta: float,
) -> None:
    """RL standard: ECDF of time to stable delivery rate."""
    tau = np.array([time_to_stable_delivery(df, w_stable, theta) for df in seed_dfs])
    xs, ys = ecdf(tau)
    max_step = max(int(df["step"].max()) for df in seed_dfs)

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(xs) > 0:
        ax.step(np.r_[0, xs, max_step + 1], np.r_[0, ys, ys[-1]], where="post", color=AGG_COLOR, lw=2)
    n_censored = int(np.sum(~np.isfinite(tau)))
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Fraction of seeds reached stable delivery")
    ax.set_title(
        f"ECDF: stable delivery (window={w_stable}, θ={theta}; censored: {n_censored}/{n_seeds})"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max_step + 1)
    ax.grid(True, alpha=0.3)
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_return_distribution(
    summaries: list[dict],
    output_path: Path,
    paradigm: str,
    n_seeds: int,
) -> None:
    """RL standard: histogram of final episode returns across seeds."""
    totals = np.array([s["total_reward"] for s in summaries])

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = max(5, min(20, len(totals)))
    ax.hist(totals, bins=bins, color=AGG_COLOR, edgecolor="black", alpha=0.75)
    ax.axvline(np.mean(totals), color="red", ls="--", lw=2, label=f"Mean = {np.mean(totals):.1f}")
    ax.axvline(np.median(totals), color="#333", ls=":", lw=2, label=f"Median = {np.median(totals):.1f}")
    ax.set_xlabel("Final team return (episode)")
    ax.set_ylabel("Count (seeds)")
    ax.set_title("Return distribution across seeds")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_return_boxplot(
    summaries: list[dict],
    seed_labels: list[str],
    output_path: Path,
    paradigm: str,
    n_seeds: int,
) -> None:
    """RL standard: box plot of final returns (with per-seed scatter)."""
    totals = np.array([s["total_reward"] for s in summaries])

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([totals], widths=0.4, patch_artist=True, showmeans=True)
    bp["boxes"][0].set_facecolor(AGG_COLOR)
    bp["boxes"][0].set_alpha(0.5)
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(totals))
    ax.scatter(1 + jitter, totals, color="#333", s=40, zorder=3, alpha=0.8)
    for j, (lab, val) in enumerate(zip(seed_labels, totals)):
        if len(seed_labels) <= 12:
            ax.annotate(lab, (1 + jitter[j], val), fontsize=6, ha="center", va="bottom", rotation=45)
    ax.set_xticks([1])
    ax.set_xticklabels([PARADIGM_TITLES.get(paradigm, paradigm)])
    ax.set_ylabel("Final team return")
    ax.set_title("Return across seeds (box plot)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_per_seed_trajectories(
    seed_dfs: list[pd.DataFrame],
    seed_labels: list[str],
    output_path: Path,
    paradigm: str,
    n_seeds: int,
) -> None:
    """RL standard: spaghetti plot of cumulative return per seed."""
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_seeds))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, df in enumerate(seed_dfs):
        team_cum = df["cumulative_reward_a0"] + df["cumulative_reward_a1"]
        ax.plot(df["step"], team_cum, lw=1.5, color=cmap[i], alpha=0.85, label=seed_labels[i])
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Cumulative team return")
    ax.set_title("Per-seed cumulative return trajectories")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_final_return_by_seed(
    summaries: list[dict],
    seed_labels: list[str],
    output_path: Path,
    paradigm: str,
    n_seeds: int,
) -> None:
    """Bar chart of final return per seed."""
    totals = [s["total_reward"] for s in summaries]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_seeds))
    x = np.arange(len(seed_labels))

    fig, ax = plt.subplots(figsize=(max(10, len(seed_labels) * 0.8), 6))
    bars = ax.bar(x, totals, color=cmap, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axhline(np.mean(totals), color="red", ls="--", lw=1.5, label=f"Mean: {np.mean(totals):.1f}")
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Final team return")
    ax.set_title("Final return by seed")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_delivery_rate_rolling(
    series: dict,
    output_path: Path,
    paradigm: str,
    n_seeds: int,
    smoothing_window: int,
) -> None:
    """Rolling fraction of steps with delivery (aggregated across seeds)."""
    steps = series["steps"]
    mean, ci_low, ci_high = bootstrap_ci(series["delivered"])
    if smoothing_window > 1:
        smooth_triple(mean, ci_low, ci_high, smoothing_window)
    valid = ~np.isnan(mean)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps[valid], mean[valid], color="#7209B7", lw=2.5, label="Mean delivery rate", zorder=3)
    if np.any(ci_high[valid] - ci_low[valid] > 1e-9):
        ax.fill_between(steps[valid], ci_low[valid], ci_high[valid], color="#7209B7", alpha=0.2)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("P(delivery at step)")
    ax.set_title(f"Rolling delivery rate (window={smoothing_window})" + _title_suffix(n_seeds))
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, steps.max() + 1)
    fig.suptitle(run_title(paradigm, n_seeds), fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def run_all_plots(
    seed_dfs: list[pd.DataFrame],
    seed_labels: list[str],
    paradigm: str,
    output_dir: Path,
    *,
    smoothing_window: int = W_CURVE,
    w_stable: int = W_STABLE,
    theta: float = THETA,
) -> None:
    n_seeds = len(seed_dfs)
    series = build_step_series(seed_dfs)
    summaries = [episode_summary(df) for df in seed_dfs]

    plot_learning_curve_cumulative_return(
        series, output_dir / "learning_curve_cumulative_return.png", paradigm, n_seeds, smoothing_window
    )
    plot_learning_curve_step_return(
        series, output_dir / "learning_curve_step_return.png", paradigm, n_seeds, smoothing_window
    )
    plot_fraction_delivered_by_step(
        series, output_dir / "success_rate_over_steps.png", paradigm, n_seeds, smoothing_window
    )
    plot_delivery_rate_rolling(
        series, output_dir / "delivery_rate_rolling.png", paradigm, n_seeds, smoothing_window
    )
    plot_ecdf_first_delivery(seed_dfs, output_dir / "ecdf_first_delivery.png", paradigm, n_seeds)
    plot_ecdf_stable_delivery(
        seed_dfs, output_dir / "ecdf_stable_delivery.png", paradigm, n_seeds, w_stable, theta
    )
    plot_return_distribution(summaries, output_dir / "return_distribution.png", paradigm, n_seeds)
    plot_return_boxplot(summaries, seed_labels, output_dir / "return_boxplot.png", paradigm, n_seeds)
    plot_per_seed_trajectories(seed_dfs, seed_labels, output_dir / "per_seed_trajectories.png", paradigm, n_seeds)
    plot_final_return_by_seed(summaries, seed_labels, output_dir / "final_return_by_seed.png", paradigm, n_seeds)


def print_summary(seed_dfs: list[pd.DataFrame], seed_labels: list[str], paradigm: str) -> None:
    summaries = [episode_summary(df) for df in seed_dfs]
    print("\n" + "=" * 80)
    print(f"SUMMARY — {PARADIGM_TITLES.get(paradigm, paradigm.upper())} ({len(seed_dfs)} seeds)")
    print("=" * 80)
    print(f"\n{'Seed':<22} {'Steps':>6} {'R_a0':>8} {'R_a1':>8} {'Total':>8} {'Success':>8}")
    print("-" * 70)
    for label, s in zip(seed_labels, summaries):
        ok = "yes" if s["success"] else "no"
        print(
            f"{label:<22} {s['steps']:>6} {s['total_reward_a0']:>8.1f} "
            f"{s['total_reward_a1']:>8.1f} {s['total_reward']:>8.1f} {ok:>8}"
        )
    totals = [s["total_reward"] for s in summaries]
    successes = [s["success"] for s in summaries]
    print(f"\nMean total reward: {np.mean(totals):.1f} ± {np.std(totals, ddof=1):.1f}")
    print(f"Success rate: {100 * np.mean(successes):.1f}% ({sum(successes)}/{len(successes)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAL Overcooked: RL-paper plots (one PNG per figure)"
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default=str(project_root / "logs" / "sal_fc"),
        help="Directory with sal_*_*.csv files",
    )
    parser.add_argument("--paradigm", choices=["fc", "ind", "ic"], default=None)
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory for PNGs (default: results/plots_sal_<paradigm>_<n>seeds)",
    )
    parser.add_argument("--smooth-window", type=int, default=W_CURVE)
    parser.add_argument("--w-stable", type=int, default=W_STABLE)
    parser.add_argument("--theta", type=float, default=THETA)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    print("=" * 80)
    print("SAL SEMANTIC ACTION LEVEL — RL PAPER PLOTS")
    print("=" * 80)
    print(f"\nLoading from: {logs_dir}")

    seed_dfs, paradigm, seed_labels = load_seed_csvs(logs_dir, args.paradigm)
    print_summary(seed_dfs, seed_labels, paradigm)

    n = len(seed_dfs)
    output_dir = args.output_dir or (project_root / "results" / f"plots_sal_{paradigm}_{n}seeds")

    print(f"\n{'=' * 80}")
    print(f"GENERATING PLOTS → {output_dir}")
    print("=" * 80)
    run_all_plots(
        seed_dfs,
        seed_labels,
        paradigm,
        output_dir,
        smoothing_window=args.smooth_window,
        w_stable=args.w_stable,
        theta=args.theta,
    )

    print(f"\nDone. Ten files in {output_dir}:")
    for name in [
        "learning_curve_cumulative_return.png",
        "learning_curve_step_return.png",
        "success_rate_over_steps.png",
        "delivery_rate_rolling.png",
        "ecdf_first_delivery.png",
        "ecdf_stable_delivery.png",
        "return_distribution.png",
        "return_boxplot.png",
        "per_seed_trajectories.png",
        "final_return_by_seed.png",
    ]:
        print(f"  - {name}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
