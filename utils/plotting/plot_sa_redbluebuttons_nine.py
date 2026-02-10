"""
Single-Agent Red–Blue Doors: 9-algorithm evaluation plots.

Implements the evaluation & plotting spec:
- Plot A: Mean episode return vs episodes (learning curve, 95% bootstrap CI)
- Plot B: Mean success rate vs episodes (learning curve, 95% bootstrap CI)
- Plot: Mean episode length (steps) vs episodes (95% bootstrap CI)
- Plot C: ECDF of time to first success
- Plot D: ECDF of time to stable success

Each plot is saved to a separate file.
Expects CSV from compare_nine_agents.py (columns: seed, agent, episode, step, reward, ...).






# One folder: use every nine_agents_comparison_*.csv inside it
python utils/plotting/plot_sa_redbluebuttons_nine.py logs/logs_sa_nine

# Plots saved in the same folder (or use -o)
python utils/plotting/plot_sa_redbluebuttons_nine.py logs/logs_sa_nine

# Save plots somewhere else
python utils/plotting/plot_sa_redbluebuttons_nine.py logs/logs_sa_nine -o plots/plots_sa_nine

# Still works: pass explicit files
python utils/plotting/plot_sa_redbluebuttons_nine.py logs/seed0.csv logs/seed1.csv -o results/results_sa_nine
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Global constants (from spec)
# ---------------------------------------------------------------------------
W_CURVE = 50       # Rolling window for learning curve smoothing
W_STABLE = 50      # Rolling window for stable success
THETA = 0.8        # Stable success threshold
BOOTSTRAP_N = 1000
CI_PERCENT = 95    # 95% CI -> 2.5 and 97.5 percentiles

# Default for experiments: 200 episodes, change config every 25 episodes, max_step 50
EPISODES_PER_CONFIG_DEFAULT = 25

# Agent order and colors (match plot_nine_agents_aggregated)
AGENT_COLORS = {
    'AIF': '#1A1A1A',
    'QLearning': '#000000',
    'Vanilla': '#2E86AB',
    'Recency0.99': '#06A77D',
    'Recency0.95': '#A23B72',
    'Recency0.9': '#F18F01',
    'Recency0.85': '#E63946',
    'TrajSampling': '#7209B7',
    'OPSRL': '#FF6B35',
}


def load_episode_data(log_files: List[Path]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load one or more comparison CSVs and build per-episode return and success.

    Expects CSV columns: seed, agent, episode, step, reward (and optionally others).
    Returns:
        episode_df: DataFrame with columns [seed, algorithm, episode, episode_return, success]
        agent_names: unique algorithm names in consistent order
        episodes: sorted unique episode numbers (1..E)
    """
    dfs = []
    for p in log_files:
        df = pd.read_csv(p)
        if 'seed' not in df.columns:
            raise ValueError(f"CSV must have 'seed' column: {p}")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Per (seed, agent, episode): sum reward = episode_return, success = (return >= 1.0), episode_length = max(step)
    ep = df.groupby(['seed', 'agent', 'episode'], as_index=False).agg(
        episode_return=('reward', 'sum'),
        episode_length=('step', 'max'),
    )
    ep['success'] = (ep['episode_return'] >= 1.0).astype(int)
    ep = ep.rename(columns={'agent': 'algorithm'})

    # Agent order: use AGENT_COLORS order for known agents, then any extras
    known = [a for a in AGENT_COLORS if a in ep['algorithm'].unique()]
    extras = [a for a in ep['algorithm'].unique() if a not in AGENT_COLORS]
    for a in extras:
        AGENT_COLORS[a] = '#888888'
    agent_names = np.array(known + extras) if known else np.array(ep['algorithm'].unique())
    episodes = np.sort(ep['episode'].unique())

    return ep, agent_names, episodes


def build_curves_per_seed(
    episode_df: pd.DataFrame,
    agent_names: np.ndarray,
    episodes: np.ndarray,
) -> Tuple[dict, dict, dict]:
    """
    Build (n_seeds, n_episodes) arrays of episode_return, success, and episode_length per algorithm.
    Fills missing (seed, episode) with nan; caller can use nanmean or mask.
    """
    seeds = np.sort(episode_df['seed'].unique())
    n_seeds = len(seeds)
    n_episodes = len(episodes)
    seed_to_idx = {float(s): i for i, s in enumerate(seeds)}
    ep_to_idx = {e: i for i, e in enumerate(episodes)}

    returns_per_agent = {}
    success_per_agent = {}
    length_per_agent = {}
    for alg in agent_names:
        ret = np.full((n_seeds, n_episodes), np.nan)
        suc = np.full((n_seeds, n_episodes), np.nan)
        length_arr = np.full((n_seeds, n_episodes), np.nan)
        sub = episode_df[episode_df['algorithm'] == alg]
        for _, row in sub.iterrows():
            sidx = seed_to_idx.get(float(row['seed']))
            if sidx is None:
                continue
            eidx = ep_to_idx.get(row['episode'])
            if eidx is not None:
                ret[sidx, eidx] = row['episode_return']
                suc[sidx, eidx] = row['success']
                length_arr[sidx, eidx] = row['episode_length']
        returns_per_agent[alg] = ret
        success_per_agent[alg] = suc
        length_per_agent[alg] = length_arr
    return returns_per_agent, success_per_agent, length_per_agent


def bootstrap_ci(
    curves: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    ci_percent: float = CI_PERCENT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    curves: (n_seeds, n_episodes). May contain nan; we use nanmean along axis=0 per sample.
    Returns mean, ci_lower, ci_upper (each n_episodes). Uses bootstrap over seeds.
    """
    n_seeds, n_episodes = curves.shape
    rng = np.random.default_rng(42)
    low = (100 - ci_percent) / 2
    high = 100 - low
    boot_means = np.zeros((n_bootstrap, n_episodes))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[b] = np.nanmean(curves[idx], axis=0)
    mean_curve = np.nanmean(curves, axis=0)
    ci_low = np.nanpercentile(boot_means, low, axis=0)
    ci_high = np.nanpercentile(boot_means, high, axis=0)
    return mean_curve, ci_low, ci_high


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean with window w; same length as x, valid only where full window fits."""
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        start = max(0, i - w + 1)
        out[i] = np.nanmean(x[start : i + 1])
    return out


def smooth_curves(mean: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray, w: int = W_CURVE):
    """Apply rolling mean to mean and CI bounds (in place)."""
    m = rolling_mean(mean, w)
    l = rolling_mean(ci_low, w)
    h = rolling_mean(ci_high, w)
    mean[:] = m
    ci_low[:] = l
    ci_high[:] = h


def add_config_boundaries(ax, max_ep: float, episodes_per_config: Optional[int]):
    if episodes_per_config is not None and episodes_per_config > 0:
        for config_ep in range(episodes_per_config, int(max_ep), episodes_per_config):
            ax.axvline(x=config_ep, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=1)


def _draw_ci_if_visible(ax, episodes: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray, valid: np.ndarray, color: str, alpha: float = 0.3):
    """Only draw fill_between when CI has positive width (e.g. when n_seeds >= 2)."""
    if np.any(np.greater(ci_high[valid] - ci_low[valid], 1e-9)):
        ax.fill_between(episodes[valid], ci_low[valid], ci_high[valid], color=color, alpha=alpha, zorder=2)


def plot_a_episode_return(
    returns_per_agent: dict,
    success_per_agent: dict,
    agent_names: np.ndarray,
    episodes: np.ndarray,
    output_path: Path,
    episodes_per_config: Optional[int] = EPISODES_PER_CONFIG_DEFAULT,
    smoothing_window: int = W_CURVE,
    n_bootstrap: int = BOOTSTRAP_N,
) -> None:
    """
    Plot A: Mean episode return vs episodes.
    One curve per algorithm, shaded 95% bootstrap CI, optional rolling mean.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    max_ep = float(episodes.max())
    n_seeds = returns_per_agent[agent_names[0]].shape[0]
    title_suffix = " (1 seed — no CI)" if n_seeds == 1 else " (95% bootstrap CI)"

    for alg in agent_names:
        ret = returns_per_agent[alg]
        mean, ci_low, ci_high = bootstrap_ci(ret, n_bootstrap=n_bootstrap)
        if smoothing_window > 0:
            smooth_curves(mean, ci_low, ci_high, w=smoothing_window)
        valid = ~np.isnan(mean)
        if not np.any(valid):
            continue
        color = AGENT_COLORS.get(alg, '#888')
        ax.plot(episodes[valid], mean[valid], color=color, lw=2, label=alg, zorder=3)
        _draw_ci_if_visible(ax, episodes, ci_low, ci_high, valid, color)

    add_config_boundaries(ax, max_ep, episodes_per_config)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean episode return')
    ax.set_title('Mean episode return vs episodes' + title_suffix)
    ax.legend(loc='best', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_ep + 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_b_mean_success(
    success_per_agent: dict,
    agent_names: np.ndarray,
    episodes: np.ndarray,
    output_path: Path,
    episodes_per_config: Optional[int] = EPISODES_PER_CONFIG_DEFAULT,
    smoothing_window: int = W_CURVE,
    n_bootstrap: int = BOOTSTRAP_N,
) -> None:
    """
    Plot B: Mean success rate vs episodes.
    One curve per algorithm, shaded 95% bootstrap CI.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    max_ep = float(episodes.max())
    n_seeds = success_per_agent[agent_names[0]].shape[0]
    title_suffix = " (1 seed — no CI)" if n_seeds == 1 else " (95% bootstrap CI)"

    for alg in agent_names:
        suc = success_per_agent[alg]
        mean, ci_low, ci_high = bootstrap_ci(suc, n_bootstrap=n_bootstrap)
        if smoothing_window > 0:
            smooth_curves(mean, ci_low, ci_high, w=smoothing_window)
        valid = ~np.isnan(mean)
        if not np.any(valid):
            continue
        color = AGENT_COLORS.get(alg, '#888')
        ax.plot(episodes[valid], mean[valid], color=color, lw=2, label=alg, zorder=3)
        _draw_ci_if_visible(ax, episodes, ci_low, ci_high, valid, color)

    add_config_boundaries(ax, max_ep, episodes_per_config)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean success rate')
    ax.set_title('Mean success rate vs episodes' + title_suffix)
    ax.legend(loc='best', ncol=3, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_ep + 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_episode_length(
    length_per_agent: dict,
    agent_names: np.ndarray,
    episodes: np.ndarray,
    output_path: Path,
    episodes_per_config: Optional[int] = EPISODES_PER_CONFIG_DEFAULT,
    smoothing_window: int = W_CURVE,
    n_bootstrap: int = BOOTSTRAP_N,
) -> None:
    """
    Mean episode length (steps) vs episodes.
    One curve per algorithm, shaded 95% bootstrap CI.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    max_ep = float(episodes.max())
    n_seeds = length_per_agent[agent_names[0]].shape[0]
    title_suffix = " (1 seed — no CI)" if n_seeds == 1 else " (95% bootstrap CI)"

    for alg in agent_names:
        length = length_per_agent[alg]
        mean, ci_low, ci_high = bootstrap_ci(length, n_bootstrap=n_bootstrap)
        if smoothing_window > 0:
            smooth_curves(mean, ci_low, ci_high, w=smoothing_window)
        valid = ~np.isnan(mean)
        if not np.any(valid):
            continue
        color = AGENT_COLORS.get(alg, '#888')
        ax.plot(episodes[valid], mean[valid], color=color, lw=2, label=alg, zorder=3)
        _draw_ci_if_visible(ax, episodes, ci_low, ci_high, valid, color)

    add_config_boundaries(ax, max_ep, episodes_per_config)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean episode length (steps)')
    ax.set_title('Mean episode length vs episodes' + title_suffix)
    ax.legend(loc='best', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_ep + 1)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def time_to_first_success(success_per_seed_episode: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    """
    success_per_seed_episode: (n_seeds, n_episodes), 0/1.
    Returns (n_seeds,) tau_first: min episode index where success==1, else np.inf.
    """
    n_seeds = success_per_seed_episode.shape[0]
    tau = np.full(n_seeds, np.inf)
    for s in range(n_seeds):
        idx = np.nonzero(success_per_seed_episode[s] >= 1.0)[0]
        if len(idx) > 0:
            tau[s] = episodes[idx[0]]
    return tau


def time_to_stable_success(
    success_per_seed_episode: np.ndarray,
    episodes: np.ndarray,
    w_stable: int = W_STABLE,
    theta: float = THETA,
) -> np.ndarray:
    """
    success_per_seed_episode: (n_seeds, n_episodes).
    Rolling success = mean of last w_stable episodes; tau_stable = first e where rolling >= theta.
    """
    n_seeds, n_episodes = success_per_seed_episode.shape
    tau = np.full(n_seeds, np.inf)
    for s in range(n_seeds):
        row = np.nan_to_num(success_per_seed_episode[s], nan=0.0)
        for i in range(w_stable - 1, n_episodes):
            start = max(0, i - w_stable + 1)
            if np.mean(row[start : i + 1]) >= theta:
                tau[s] = episodes[i]
                break
    return tau


def ecdf(x: np.ndarray, censored_inf: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: 1D array of values (may contain np.inf for censored).
    Returns (sorted unique values, cumulative fraction).
    Censored (inf) are excluded from the curve so curve never reaches 1.0 if any censored.
    """
    x_finite = x[np.isfinite(x)]
    if len(x_finite) == 0:
        return np.array([0.0]), np.array([0.0])
    xs = np.sort(np.unique(x_finite))
    n = len(x_finite)
    # Fraction of all seeds (including censored) with value <= x
    counts = np.searchsorted(np.sort(x_finite), xs, side='right')
    frac = counts / len(x)  # divide by full n_seeds so censored don't reach 1
    return xs, frac


def plot_c_ecdf_first_success(
    success_per_agent: dict,
    agent_names: np.ndarray,
    episodes: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot C: ECDF of time to first success.
    x-axis: episodes, y-axis: fraction of runs that had first success by that episode.
    Censored runs (never succeeded) never reach 1.0.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    max_ep = float(episodes.max())

    for alg in agent_names:
        suc = success_per_agent[alg]
        tau = time_to_first_success(suc, episodes)
        xs, ys = ecdf(tau, censored_inf=True)
        if len(xs) > 0:
            ax.step(np.r_[0, xs, max_ep + 1], np.r_[0, ys, ys[-1]], where='post', color=AGENT_COLORS.get(alg, '#888'), lw=1.5, label=alg)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Fraction of runs succeeded (first success)')
    ax.set_title('ECDF: Time to first success')
    ax.legend(loc='lower right', ncol=3, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max_ep + 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_d_ecdf_stable_success(
    success_per_agent: dict,
    agent_names: np.ndarray,
    episodes: np.ndarray,
    output_path: Path,
    w_stable: int = W_STABLE,
    theta: float = THETA,
) -> None:
    """
    Plot D: ECDF of time to stable success (rolling success >= theta).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    max_ep = float(episodes.max())

    for alg in agent_names:
        suc = success_per_agent[alg]
        tau = time_to_stable_success(suc, episodes, w_stable=w_stable, theta=theta)
        xs, ys = ecdf(tau, censored_inf=True)
        if len(xs) > 0:
            ax.step(np.r_[0, xs, max_ep + 1], np.r_[0, ys, ys[-1]], where='post', color=AGENT_COLORS.get(alg, '#888'), lw=1.5, label=alg)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Fraction of runs reached stable success')
    ax.set_title(f'ECDF: Time to stable success (rolling window={w_stable}, θ={theta})')
    ax.legend(loc='lower right', ncol=3, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max_ep + 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# Glob pattern for seed CSV files produced by compare_nine_agents.py
LOG_DIR_GLOB = "nine_agents_comparison_*.csv"


def collect_log_files_from_paths(paths: List[Path]) -> List[Path]:
    """
    Collect log files from a mix of file paths and directory paths.
    - If a path is a file, add it to the list.
    - If a path is a directory, add all files matching LOG_DIR_GLOB inside it (non-recursive).
    Returns a sorted list of unique paths (one per seed file).
    """
    collected: List[Path] = []
    for p in paths:
        p = Path(p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"No such path: {p}")
        if p.is_file():
            collected.append(p)
        else:
            for f in sorted(p.glob(LOG_DIR_GLOB)):
                if f.is_file():
                    collected.append(f)
    if not collected:
        raise ValueError(
            f"No log files found. For a directory, expected files matching '{LOG_DIR_GLOB}'."
        )
    return sorted(set(collected))


def run_all_plots(
    log_files: List[Path],
    output_dir: Optional[Path] = None,
    episodes_per_config: Optional[int] = EPISODES_PER_CONFIG_DEFAULT,
    smoothing_window: int = W_CURVE,
    w_stable: int = W_STABLE,
    theta: float = THETA,
) -> None:
    """
    Load data, build curves, and save Plot A–D plus episode length as separate files.
    If output_dir is None, uses parent of first log file.
    """
    if not log_files:
        raise ValueError("Need at least one log file")
    output_dir = output_dir or Path(log_files[0]).parent

    episode_df, agent_names, episodes = load_episode_data(log_files)
    returns_per_agent, success_per_agent, length_per_agent = build_curves_per_seed(episode_df, agent_names, episodes)

    # Ensure agent order: use AGENT_COLORS order for known agents, then any extras
    order = [a for a in AGENT_COLORS if a in agent_names]
    for a in agent_names:
        if a not in order:
            order.append(a)
    agent_names = np.array(order)

    plot_a_episode_return(
        returns_per_agent,
        success_per_agent,
        agent_names,
        episodes,
        output_dir / "plot_a_episode_return.png",
        episodes_per_config=episodes_per_config,
        smoothing_window=smoothing_window,
    )
    print(f"  Saved {output_dir / 'plot_a_episode_return.png'}")

    plot_b_mean_success(
        success_per_agent,
        agent_names,
        episodes,
        output_dir / "plot_b_mean_success.png",
        episodes_per_config=episodes_per_config,
        smoothing_window=smoothing_window,
    )
    print(f"  Saved {output_dir / 'plot_b_mean_success.png'}")

    plot_episode_length(
        length_per_agent,
        agent_names,
        episodes,
        output_dir / "plot_episode_length.png",
        episodes_per_config=episodes_per_config,
        smoothing_window=smoothing_window,
    )
    print(f"  Saved {output_dir / 'plot_episode_length.png'}")

    plot_c_ecdf_first_success(
        success_per_agent,
        agent_names,
        episodes,
        output_dir / "plot_c_ecdf_first_success.png",
    )
    print(f"  Saved {output_dir / 'plot_c_ecdf_first_success.png'}")

    plot_d_ecdf_stable_success(
        success_per_agent,
        agent_names,
        episodes,
        output_dir / "plot_d_ecdf_stable_success.png",
        w_stable=w_stable,
        theta=theta,
    )
    print(f"  Saved {output_dir / 'plot_d_ecdf_stable_success.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Single-Agent Red–Blue Doors: 9-algorithm evaluation plots (spec). '
        'Pass a log directory (finds all nine_agents_comparison_*.csv) or one or more CSV files.'
    )
    parser.add_argument(
        'paths',
        nargs='+',
        type=Path,
        help='Path to folder of log files (finds nine_agents_comparison_*.csv), or one or more CSV file paths',
    )
    parser.add_argument('--output_dir', '-o', type=Path, default=None, help='Directory to save plots (default: log directory)')
    parser.add_argument('--episodes_per_config', type=int, default=EPISODES_PER_CONFIG_DEFAULT, help='Episodes per config (vertical lines; default 25)')
    parser.add_argument('--smoothing_window', type=int, default=W_CURVE, help=f'Rolling window for learning curves (default {W_CURVE})')
    parser.add_argument('--w_stable', type=int, default=W_STABLE, help=f'Rolling window for stable success (default {W_STABLE})')
    parser.add_argument('--theta', type=float, default=THETA, help=f'Stable success threshold (default {THETA})')
    args = parser.parse_args()

    log_files = collect_log_files_from_paths(args.paths)
    print(f"Using {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  {f.name}")
    print()

    run_all_plots(
        log_files,
        output_dir=args.output_dir,
        episodes_per_config=args.episodes_per_config,
        smoothing_window=args.smoothing_window,
        w_stable=args.w_stable,
        theta=args.theta,
    )
    print("Done. Five files: plot_a_episode_return.png, plot_b_mean_success.png, plot_episode_length.png, plot_c_ecdf_first_success.png, plot_d_ecdf_stable_success.png")


if __name__ == "__main__":
    main()
