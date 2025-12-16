"""
Plot episode length curves for each CSV file individually.

Creates separate episode length plots for each CSV, showing:
- Episode length (steps) over episodes
- Average and confidence intervals if multiple seeds are provided
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_episode_lengths(path: Path) -> Dict[int, int]:
    """
    Load episode lengths from a CSV file.
    
    Returns a dict mapping episode number to episode length (steps).
    """
    by_episode: Dict[int, List[Dict[str, str]]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["episode"])
            by_episode.setdefault(ep, []).append(row)
    
    episode_lengths = {}
    for ep, rows in sorted(by_episode.items()):
        rows = sorted(rows, key=lambda r: int(r["step"]))
        episode_lengths[ep] = int(rows[-1]["step"])
    
    return episode_lengths


def plot_episode_length_single(episode_lengths: Dict[int, int], output_path: Path, title: str):
    """
    Plot episode length curve for a single CSV file.
    """
    episodes = np.array(sorted(episode_lengths.keys()))
    lengths = np.array([episode_lengths[ep] for ep in episodes])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(episodes, lengths, linestyle="-", linewidth=2, marker="o", markersize=3,
            color="#A23B72", alpha=0.8, label="Episode length")
    
    ax.set_xlabel("Episode", fontsize=14, fontweight="bold")
    ax.set_ylabel("Steps", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, episodes.max() + 1)
    ax.set_ylim(0, max(lengths.max() * 1.1, 10))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Episode length plot saved to: {output_path}")


def plot_episode_length_multi(seed_lengths: List[Dict[int, int]], output_path: Path, title: str, episodes_per_config: int = None):
    """
    Plot episode length curves aggregated across multiple seeds with confidence intervals.
    """
    from scipy import stats
    
    # Get union of all episode indices
    all_episodes = set()
    for lengths in seed_lengths:
        all_episodes.update(lengths.keys())
    episodes = np.array(sorted(all_episodes))
    n_ep = len(episodes)
    n_seeds = len(seed_lengths)
    
    # Build matrix: seeds x episodes
    lengths_matrix = np.zeros((n_seeds, n_ep))
    for i, lengths in enumerate(seed_lengths):
        for j, ep in enumerate(episodes):
            lengths_matrix[i, j] = lengths.get(ep, np.nan)
    
    # Compute mean and CI
    z = stats.norm.ppf(0.975)  # 95% CI
    mean_lengths = np.nanmean(lengths_matrix, axis=0)
    std_lengths = np.nanstd(lengths_matrix, axis=0, ddof=1)
    se = std_lengths / np.sqrt(n_seeds)
    ci_lower = mean_lengths - z * se
    ci_upper = mean_lengths + z * se
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(episodes, mean_lengths, linestyle="-", linewidth=2.5, marker="o", markersize=3,
            color="#A23B72", alpha=0.9, label="Mean episode length")
    ax.fill_between(episodes, ci_lower, ci_upper, color="#A23B72", alpha=0.15, label="95% CI")
    
    # Add config boundaries if specified
    if episodes_per_config is not None and episodes_per_config > 0:
        max_ep = episodes.max()
        for ep in range(episodes_per_config, int(max_ep) + 1, episodes_per_config):
            ax.axvline(x=ep, color="gray", linestyle=":", linewidth=0.5, alpha=0.3, zorder=1)
    
    ax.set_xlabel("Episode", fontsize=14, fontweight="bold")
    ax.set_ylabel("Steps", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, episodes.max() + 1)
    ax.set_ylim(0, max(mean_lengths.max() * 1.1, 10))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Episode length plot (aggregated) saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot episode length curves for cooperation CSV files"
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="CSV paths to plot",
    )
    parser.add_argument(
        "--episodes-per-config",
        type=int,
        default=None,
        help="Episodes per environment configuration (for vertical lines)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/coop",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="If set, aggregate all CSVs as seeds and create one plot with CI",
    )
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.aggregate:
        # Load all CSVs and aggregate
        seed_lengths = []
        for p in args.paths:
            path = Path(p)
            if not path.exists():
                print(f"[WARN] File not found: {path}")
                continue
            print(f"Loading {path.name}")
            lengths = load_episode_lengths(path)
            seed_lengths.append(lengths)
        
        if not seed_lengths:
            print("No valid CSV files provided.")
            return
        
        output_path = out_dir / "episode_length_aggregated.png"
        title = "Episode Length Over Episodes (Aggregated Across Seeds)"
        plot_episode_length_multi(seed_lengths, output_path, title, args.episodes_per_config)
    else:
        # Plot each CSV individually
        for p in args.paths:
            path = Path(p)
            if not path.exists():
                print(f"[WARN] File not found: {path}")
                continue
            
            print(f"Loading {path.name}")
            lengths = load_episode_lengths(path)
            
            # Generate output filename from input filename
            stem = path.stem
            output_path = out_dir / f"episode_length_{stem}.png"
            title = f"Episode Length Over Episodes: {stem}"
            
            plot_episode_length_single(lengths, output_path, title)


if __name__ == "__main__":
    main()

