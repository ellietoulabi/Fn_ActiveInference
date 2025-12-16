"""
Analyze cooperation logs for two-agent AIF runs.

Expected input: CSV files with columns like
    seed,episode,step,joint_action1,joint_action2,action1,action1_name,
    action2,action2_name,map,reward,result,button_pressed,pressed_by

For each CSV (treated as one "pair" / condition), this script computes:
  - Per-seed success rate (fraction of episodes with result == 'win')
  - Per-seed count of "true cooperative" wins where:
        * result == 'win'
        * red was pressed by one agent, blue by the other
        * in different steps within the same episode
  - Aggregate stats across seeds
  - Plots summarizing success rates and cooperative-win counts per seed

Usage (from project root):
    python coop/analyze_coop_logs.py \
        --paths coop/two_aif_agents_individually_collective_seeds1_ep100_20251216_034907.csv \
               coop/two_aif_agents_individually_collective_seeds1_ep100_20251216_034908.csv \
               coop/two_aif_agents_individually_collective_seeds1_ep100_20251216_034909.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def load_episodes(path: Path):
    """
    Load CSV and group rows by (seed, episode).

    Returns:
        episodes: dict[(seed, episode)] -> list[dict(row)]
    """
    episodes = defaultdict(list)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(row["seed"])
            episode = int(row["episode"])
            episodes[(seed, episode)].append(row)
    return episodes


def classify_episode(rows: List[Dict[str, str]]) -> Tuple[bool, bool]:
    """
    Classify a single episode as success / cooperative-success.

    Args:
        rows: list of CSV rows (dicts) for one (seed, episode), in time order.

    Returns:
        (is_success, is_coop)
          is_success: True if final result == 'win'
          is_coop: True if:
              - is_success
              - red was pressed by one agent, blue by the other
              - in different steps (no simultaneous double press)
    """
    if not rows:
        return False, False

    # Determine final result from last row
    final = rows[-1]
    result = str(final.get("result", "")).strip()
    is_success = result == "win"
    if not is_success:
        return False, False

    # Track who pressed red / blue and on which step
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

    if not red_events or not blue_events:
        return True, False

    # Take earliest red and earliest blue events
    red_step, red_who = sorted(red_events, key=lambda x: x[0])[0]
    blue_step, blue_who = sorted(blue_events, key=lambda x: x[0])[0]

    # Cooperative if:
    # - pressed by different agents
    # - at different steps (enforces temporal separation)
    is_coop = (red_who != blue_who) and (red_step != blue_step)
    return True, is_coop


def analyze_file(path: Path):
    """
    Analyze one CSV file; returns per-seed stats and aggregates.

    Returns:
        per_seed: dict[int] -> dict with keys:
            'episodes', 'wins', 'coop_wins', 'success_rate', 'coop_rate'
        aggregate: dict with same keys aggregated across seeds
    """
    episodes = load_episodes(path)

    per_seed_counts = defaultdict(lambda: {"episodes": 0, "wins": 0, "coop_wins": 0})

    for (seed, ep), rows in episodes.items():
        is_success, is_coop = classify_episode(sorted(rows, key=lambda r: int(r["step"])))
        stats = per_seed_counts[seed]
        stats["episodes"] += 1
        if is_success:
            stats["wins"] += 1
        if is_coop:
            stats["coop_wins"] += 1

    per_seed = {}
    for seed, stats in per_seed_counts.items():
        n = stats["episodes"]
        wins = stats["wins"]
        coop = stats["coop_wins"]
        success_rate = wins / n if n > 0 else 0.0
        coop_rate = coop / n if n > 0 else 0.0
        per_seed[seed] = {
            "episodes": n,
            "wins": wins,
            "coop_wins": coop,
            "success_rate": success_rate,
            "coop_rate": coop_rate,
        }

    # Aggregate across seeds
    total_episodes = sum(s["episodes"] for s in per_seed.values())
    total_wins = sum(s["wins"] for s in per_seed.values())
    total_coop = sum(s["coop_wins"] for s in per_seed.values())

    aggregate = {
        "episodes": total_episodes,
        "wins": total_wins,
        "coop_wins": total_coop,
        "success_rate": (total_wins / total_episodes) if total_episodes > 0 else 0.0,
        "coop_rate": (total_coop / total_episodes) if total_episodes > 0 else 0.0,
    }

    return per_seed, aggregate


def print_summary(label: str, per_seed, aggregate):
    print(f"\n=== {label} ===")
    print(f"Total episodes: {aggregate['episodes']}")
    print(f"Total wins:     {aggregate['wins']}  "
          f"({aggregate['success_rate']*100:.1f}% success)")
    print(f"Coop wins:      {aggregate['coop_wins']}  "
          f"({aggregate['coop_rate']*100:.1f}% cooperative)")
    print("\nPer-seed:")
    print(f"{'Seed':<6} {'Eps':<6} {'Wins':<8} {'CoopWins':<10} "
          f"{'Succ%':<8} {'Coop%':<8}")
    for seed in sorted(per_seed.keys()):
        s = per_seed[seed]
        print(f"{seed:<6} {s['episodes']:<6} {s['wins']:<8} {s['coop_wins']:<10} "
              f"{s['success_rate']*100:6.1f} {s['coop_rate']*100:6.1f}")


def plot_results(label: str, per_seed, aggregate, out_dir: Path):
    """
    Plot per-seed success and cooperative rates using the project's plotting style.
    """
    # Lazy import matplotlib; make it headless-friendly
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted(per_seed.keys())
    if not seeds:
        return

    succ = [per_seed[s]["success_rate"] * 100 for s in seeds]
    coop = [per_seed[s]["coop_rate"] * 100 for s in seeds]

    # Create 2x2 subplot layout matching project style
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Cooperation Analysis: {os.path.basename(label)}', 
                 fontsize=14, fontweight='bold')

    # Colors matching project style
    success_color = '#27ae60'  # Green
    coop_color = '#3498db'     # Blue

    # =========================================================================
    # Plot 1: Success Rate and Cooperation Rate per Seed (Bar Chart)
    # =========================================================================
    ax = axes[0, 0]
    x = np.arange(len(seeds))
    width = 0.35

    bars1 = ax.bar(x - width/2, succ, width, label='Success Rate', 
                   color=success_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, coop, width, label='Cooperation Rate', 
                   color=coop_color, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel('Seed')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Success and Cooperation Rates per Seed')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(succ) if succ else [0], max(coop) if coop else [0]) + 15)

    # =========================================================================
    # Plot 2: Overall Success vs Cooperation (Aggregate Bar Chart)
    # =========================================================================
    ax = axes[0, 1]
    x_pos = [0, 1]
    means = [aggregate['success_rate'] * 100, aggregate['coop_rate'] * 100]
    labels_bar = ['Overall\nSuccess', 'Overall\nCooperation']
    colors_bar = [success_color, coop_color]

    bars = ax.bar(x_pos, means, color=colors_bar, alpha=0.7, 
                  edgecolor='black', linewidth=1.5, capsize=5)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Aggregate Success and Cooperation Rates')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(means) + 15)

    # =========================================================================
    # Plot 3: Wins Breakdown (Total Wins vs Cooperative Wins)
    # =========================================================================
    ax = axes[1, 0]
    total_wins = aggregate['wins']
    coop_wins = aggregate['coop_wins']
    non_coop_wins = total_wins - coop_wins

    categories = ['Total Wins', 'Cooperative\nWins', 'Non-Cooperative\nWins']
    values = [total_wins, coop_wins, non_coop_wins]
    colors_pie = [success_color, coop_color, '#e74c3c']  # Red for non-coop

    bars = ax.bar(categories, values, color=colors_pie, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Count')
    ax.set_title(f'Win Breakdown (Total: {total_wins} wins, {aggregate["episodes"]} episodes)')
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Plot 4: Cooperation Efficiency (Coop Wins / Total Wins per Seed)
    # =========================================================================
    ax = axes[1, 1]
    coop_efficiency = []
    for s in seeds:
        wins = per_seed[s]['wins']
        coop_w = per_seed[s]['coop_wins']
        eff = (coop_w / wins * 100) if wins > 0 else 0.0
        coop_efficiency.append(eff)

    bars = ax.bar(seeds, coop_efficiency, color=coop_color, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, eff in zip(bars, coop_efficiency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Seed')
    ax.set_ylabel('Cooperation Efficiency (%)')
    ax.set_title('Cooperation Efficiency\n(Coop Wins / Total Wins per Seed)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(coop_efficiency) + 15 if coop_efficiency else 105)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = os.path.basename(label).replace(".csv", "")
    out_path = out_dir / f"{safe_label}_coop_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved to: {out_path}")


def plot_comparison(all_results: List[Tuple[str, Dict, Dict]], out_dir: Path):
    """
    Create comparison plot across multiple CSV files (pairs/conditions).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(all_results) < 2:
        return  # Need at least 2 to compare

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cooperation Comparison Across Pairs', 
                 fontsize=14, fontweight='bold')

    labels = [os.path.basename(label).replace(".csv", "") for label, _, _ in all_results]
    success_rates = [agg['success_rate'] * 100 for _, _, agg in all_results]
    coop_rates = [agg['coop_rate'] * 100 for _, _, agg in all_results]
    total_wins = [agg['wins'] for _, _, agg in all_results]
    coop_wins = [agg['coop_wins'] for _, _, agg in all_results]

    success_color = '#27ae60'
    coop_color = '#3498db'

    # =========================================================================
    # Plot 1: Success Rate Comparison
    # =========================================================================
    ax = axes[0, 0]
    x = np.arange(len(labels))
    bars = ax.bar(x, success_rates, color=success_color, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:30] + '...' if len(l) > 30 else l for l in labels], 
                       rotation=15, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Overall Success Rate Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(success_rates) + 15 if success_rates else 105)

    # =========================================================================
    # Plot 2: Cooperation Rate Comparison
    # =========================================================================
    ax = axes[0, 1]
    bars = ax.bar(x, coop_rates, color=coop_color, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    for bar, rate in zip(bars, coop_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:30] + '...' if len(l) > 30 else l for l in labels], 
                       rotation=15, ha='right')
    ax.set_ylabel('Cooperation Rate (%)')
    ax.set_title('Overall Cooperation Rate Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(coop_rates) + 15 if coop_rates else 105)

    # =========================================================================
    # Plot 3: Side-by-Side Success and Cooperation
    # =========================================================================
    ax = axes[1, 0]
    width = 0.35
    bars1 = ax.bar(x - width/2, success_rates, width, label='Success Rate', 
                   color=success_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, coop_rates, width, label='Cooperation Rate', 
                   color=coop_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:20] + '...' if len(l) > 20 else l for l in labels], 
                       rotation=15, ha='right')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Success vs Cooperation Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(success_rates) if success_rates else [0], 
                       max(coop_rates) if coop_rates else [0]) + 15)

    # =========================================================================
    # Plot 4: Win Counts Comparison
    # =========================================================================
    ax = axes[1, 1]
    width = 0.35
    bars1 = ax.bar(x - width/2, total_wins, width, label='Total Wins', 
                   color=success_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, coop_wins, width, label='Cooperative Wins', 
                   color=coop_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:20] + '...' if len(l) > 20 else l for l in labels], 
                       rotation=15, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Win Counts Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cooperation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Comparison plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze cooperation logs for two-agent AIF runs'
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="CSV paths to analyze (each treated as one pair/condition)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: results/coop/)",
    )
    args = parser.parse_args()

    # Default output directory
    if args.out_dir is None:
        out_dir = project_root / "results" / "coop"
    else:
        out_dir = Path(args.out_dir)

    print("="*80)
    print("COOPERATION ANALYSIS")
    print("="*80)

    all_results = []

    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue

        print(f"\n{'='*80}")
        print(f"Analyzing: {path.name}")
        print(f"{'='*80}")

        per_seed, aggregate = analyze_file(path)
        print_summary(str(path), per_seed, aggregate)
        plot_results(str(path), per_seed, aggregate, out_dir)
        
        all_results.append((str(path), per_seed, aggregate))

    # Create comparison plot if multiple files
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("GENERATING COMPARISON PLOT")
        print(f"{'='*80}")
        plot_comparison(all_results, out_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()


