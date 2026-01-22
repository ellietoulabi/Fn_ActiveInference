"""
Plot cooperation learning curves in the same style as
`utils/plotting/plot_eight_agents_from_seed_files.py`.

Treat each CSV as one "seed" of the same condition and aggregate
across those seeds to produce:
  1. Average reward convergence over episodes
  2. Cumulative reward over time
  3. Episode length (steps)
  4. Success rate (win) over episodes
  5. Cooperative success rate over episodes

Expected CSV format (from two-agent AIF runs):
    seed,episode,step,action1,...,reward,result,button_pressed,pressed_by
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_episode_stats_from_csv(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load one CSV and compute per-episode statistics.

    Returns a dict:
        {
          'episodes': np.array([...]),
          'total_reward': np.array([...]),
          'steps': np.array([...]),
          'success': np.array([...], dtype=bool),
          'coop_success': np.array([...], dtype=bool),
        }
    where each index corresponds to one episode in sorted order.
    """
    # Group rows by episode
    by_episode: Dict[int, List[Dict[str, str]]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["episode"])
            by_episode.setdefault(ep, []).append(row)

    episodes = sorted(by_episode.keys())
    n = len(episodes)

    total_reward = np.zeros(n, dtype=float)
    steps = np.zeros(n, dtype=int)
    success = np.zeros(n, dtype=bool)
    coop_success = np.zeros(n, dtype=bool)

    for i, ep in enumerate(episodes):
        rows = sorted(by_episode[ep], key=lambda r: int(r["step"]))
        # Total reward and steps
        r_sum = sum(float(r["reward"]) for r in rows)
        total_reward[i] = r_sum
        steps[i] = int(rows[-1]["step"])

        # Success = final result == 'win'
        final_res = str(rows[-1].get("result", "")).strip()
        is_win = final_res == "win"
        success[i] = is_win

        # Cooperative success as in analyze_coop_logs: different agents press red/blue at different steps
        if is_win:
            red_events = []
            blue_events = []
            for r in rows:
                btn = r.get("button_pressed", "")
                who = r.get("pressed_by", "")
                if not btn or not who:
                    continue
                step = int(r["step"])
                if btn == "red":
                    red_events.append((step, who))
                elif btn == "blue":
                    blue_events.append((step, who))

            if red_events and blue_events:
                red_step, red_who = sorted(red_events, key=lambda x: x[0])[0]
                blue_step, blue_who = sorted(blue_events, key=lambda x: x[0])[0]
                coop_success[i] = (red_who != blue_who) and (red_step != blue_step)
            else:
                coop_success[i] = False
        else:
            coop_success[i] = False

    return {
        "episodes": np.array(episodes, dtype=int),
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "coop_success": coop_success,
    }


def aggregate_across_seeds(seed_stats: List[Dict[str, np.ndarray]], ci_level: float = 0.95):
    """
    Aggregate per-episode stats across seeds.

    Assumes all seeds share the same episode indices.
    Returns a dict with 'episodes' and per-metric mean/std/ci bounds.
    """
    from scipy import stats

    n_seeds = len(seed_stats)
    if n_seeds == 0:
        raise ValueError("No seed stats provided")

    episodes = seed_stats[0]["episodes"]
    n_ep = len(episodes)

    # Build matrices: seeds x episodes
    rewards = np.zeros((n_seeds, n_ep))
    steps = np.zeros((n_seeds, n_ep))
    success = np.zeros((n_seeds, n_ep))
    coop = np.zeros((n_seeds, n_ep))

    for i, s in enumerate(seed_stats):
        assert np.array_equal(s["episodes"], episodes), "Episode indices differ between seeds"
        rewards[i, :] = s["total_reward"]
        steps[i, :] = s["steps"]
        success[i, :] = s["success"].astype(float)
        coop[i, :] = s["coop_success"].astype(float)

    z = stats.norm.ppf((1 + ci_level) / 2)

    def summarize(mat):
        mean = mat.mean(axis=0)
        std = mat.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)
        se = std / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean)
        ci_lower = mean - z * se
        ci_upper = mean + z * se
        return {"mean": mean, "std": std, "ci_lower": ci_lower, "ci_upper": ci_upper}

    return {
        "episodes": episodes,
        "reward": summarize(rewards),
        "steps": summarize(steps),
        "success": summarize(success * 100.0),
        "coop": summarize(coop * 100.0),
    }


def add_config_boundaries(ax, max_ep, episodes_per_config: int):
    if episodes_per_config is None or episodes_per_config <= 0:
        return
    for ep in range(episodes_per_config, int(max_ep) + 1, episodes_per_config):
        ax.axvline(x=ep, color="gray", linestyle=":", linewidth=0.5, alpha=0.3, zorder=1)


def plot_coop_learning(agg, episodes_per_config: int, output_path: Path, title: str):
    """
    Create a 5-panel figure similar to plot_eight_agents_from_seed_files.py
    using aggregated cooperation stats across seeds.
    """
    episodes = agg["episodes"]
    max_ep = episodes.max()

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    # Colors
    reward_color = "#2E86AB"
    success_color = "#27ae60"
    coop_color = "#3498db"

    # 1) Average reward convergence (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    r_mean = agg["reward"]["mean"]
    cumsum = np.cumsum(r_mean)
    cumavg = cumsum / np.arange(1, len(cumsum) + 1)

    # Approximate CI for average reward
    ci_lower = np.cumsum(agg["reward"]["ci_lower"]) / np.arange(1, len(cumsum) + 1)
    ci_upper = np.cumsum(agg["reward"]["ci_upper"]) / np.arange(1, len(cumsum) + 1)

    ax1.plot(episodes, cumavg, linestyle="-", linewidth=2.5, marker="o", markersize=4,
             label="Average reward (cum. mean)", color=reward_color, alpha=0.9, zorder=3)
    ax1.fill_between(episodes, ci_lower, ci_upper, color=reward_color, alpha=0.15, zorder=2)

    add_config_boundaries(ax1, max_ep, episodes_per_config)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Episode", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Average Reward", fontsize=14, fontweight="bold")
    ax1.set_title("Average Reward Convergence", fontsize=16, fontweight="bold", pad=15)
    ax1.legend(loc="best", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_xlim(0, max_ep + 1)

    # 2) Cumulative reward over time (second row left)
    ax2 = fig.add_subplot(gs[1, 0])
    cumsum_mean = np.cumsum(r_mean)
    cumsum_ci_lower = np.cumsum(agg["reward"]["ci_lower"])
    cumsum_ci_upper = np.cumsum(agg["reward"]["ci_upper"])

    ax2.plot(episodes, cumsum_mean, linestyle="-", linewidth=2.5,
             color=reward_color, alpha=0.9, label="Cumulative reward")
    ax2.fill_between(episodes, cumsum_ci_lower, cumsum_ci_upper,
                     color=reward_color, alpha=0.15)

    add_config_boundaries(ax2, max_ep, episodes_per_config)
    ax2.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax2.set_title("Cumulative Reward Over Time", fontsize=14, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_ep + 1)

    # 3) Episode length (steps) (second row right)
    ax3 = fig.add_subplot(gs[1, 1])
    s_mean = agg["steps"]["mean"]
    s_ci_lo = agg["steps"]["ci_lower"]
    s_ci_hi = agg["steps"]["ci_upper"]

    ax3.plot(episodes, s_mean, linestyle="-", linewidth=2.5,
             color="#A23B72", alpha=0.9, label="Steps")
    ax3.fill_between(episodes, s_ci_lo, s_ci_hi, color="#A23B72", alpha=0.15)

    add_config_boundaries(ax3, max_ep, episodes_per_config)
    ax3.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Steps", fontsize=12, fontweight="bold")
    ax3.set_title("Episode Length (Steps)", fontsize=14, fontweight="bold")
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_ep + 1)

    # 4) Success rate per episode (third row left)
    ax4 = fig.add_subplot(gs[2, 0])
    succ_mean = agg["success"]["mean"]
    succ_lo = agg["success"]["ci_lower"]
    succ_hi = agg["success"]["ci_upper"]

    ax4.plot(episodes, succ_mean, linestyle="-", linewidth=2.5,
             color=success_color, alpha=0.9, label="Success rate")
    ax4.fill_between(episodes, succ_lo, succ_hi, color=success_color, alpha=0.15)

    add_config_boundaries(ax4, max_ep, episodes_per_config)
    ax4.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax4.set_title("Success Rate Over Episodes", fontsize=14, fontweight="bold")
    ax4.legend(loc="best", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max_ep + 1)
    ax4.set_ylim(-5, 105)

    # 5) Cooperative success rate per episode (third row right)
    ax5 = fig.add_subplot(gs[2, 1])
    coop_mean = agg["coop"]["mean"]
    coop_lo = agg["coop"]["ci_lower"]
    coop_hi = agg["coop"]["ci_upper"]

    ax5.plot(episodes, coop_mean, linestyle="-", linewidth=2.5,
             color=coop_color, alpha=0.9, label="Cooperative success rate")
    ax5.fill_between(episodes, coop_lo, coop_hi, color=coop_color, alpha=0.15)

    add_config_boundaries(ax5, max_ep, episodes_per_config)
    ax5.set_xlabel("Episode", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Coop Success Rate (%)", fontsize=12, fontweight="bold")
    ax5.set_title("Cooperative Success Rate Over Episodes", fontsize=14, fontweight="bold")
    ax5.legend(loc="best", fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, max_ep + 1)
    ax5.set_ylim(-5, 105)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Coop learning plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cooperation learning curves like eight_agents_from_seed_files.py"
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="CSV paths (each treated as one seed of the same condition)",
    )
    parser.add_argument(
        "--episodes-per-config",
        type=int,
        default=100,
        help="Episodes per environment configuration (for vertical lines)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/coop/coop_learning.png",
        help="Output path for the plot",
    )
    args = parser.parse_args()

    seed_stats = []
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue
        print(f"Loading seed from {path}")
        stats = load_episode_stats_from_csv(path)
        seed_stats.append(stats)

    if not seed_stats:
        print("No valid CSV files provided.")
        return

    agg = aggregate_across_seeds(seed_stats, ci_level=0.95)
    output_path = Path(args.out)
    title = "Two-Agent Cooperation: Learning Curves (Aggregated Across Seeds)"
    plot_coop_learning(agg, args.episodes_per_config, output_path, title)


if __name__ == "__main__":
    main()







