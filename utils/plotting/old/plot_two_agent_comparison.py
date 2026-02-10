"""
Plot comparison between Q-Learning agents and Random agents in two-agent collaboration.

Loads results from CSV logs and generates comparison plots.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from glob import glob
import argparse


def load_latest_csv(pattern, log_dir):
    """Load the most recent CSV file matching the pattern."""
    files = sorted(glob(str(log_dir / pattern)), key=lambda x: Path(x).stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern} in {log_dir}")
    latest = files[-1]
    print(f"Loading: {latest}")
    return pd.read_csv(latest), latest


def compute_episode_stats(df):
    """Compute per-episode statistics from step-level data."""
    # Group by seed and episode to get episode-level stats
    episode_stats = df.groupby(['seed', 'episode']).agg({
        'reward': 'sum',
        'step': 'max',
        'result': 'last'
    }).reset_index()
    
    episode_stats['success'] = episode_stats['result'] == 'win'
    
    return episode_stats


def compute_learning_curve(episode_stats, num_seeds, num_episodes):
    """Compute average success rate per episode across seeds."""
    success_per_episode = []
    
    for ep in range(1, num_episodes + 1):
        ep_data = episode_stats[episode_stats['episode'] == ep]
        if len(ep_data) > 0:
            success_per_episode.append(ep_data['success'].mean())
        else:
            success_per_episode.append(0)
    
    return success_per_episode


def compute_learning_curve_with_ci(episode_stats, num_episodes):
    """
    Compute average success rate per episode across seeds with confidence interval.
    Returns: episodes, means, ci_lower, ci_upper
    """
    episodes = list(range(1, num_episodes + 1))
    means = []
    ci_lower = []
    ci_upper = []
    
    for ep in episodes:
        ep_data = episode_stats[episode_stats['episode'] == ep]
        if len(ep_data) > 1:
            # Get success rate per seed for this episode
            success_values = ep_data.groupby('seed')['success'].mean().values
            mean = np.mean(success_values)
            
            # 95% confidence interval
            n = len(success_values)
            se = np.std(success_values, ddof=1) / np.sqrt(n)
            ci = 1.96 * se  # 95% CI
            
            means.append(mean)
            ci_lower.append(mean - ci)
            ci_upper.append(mean + ci)
        elif len(ep_data) == 1:
            means.append(ep_data['success'].values[0])
            ci_lower.append(ep_data['success'].values[0])
            ci_upper.append(ep_data['success'].values[0])
        else:
            means.append(0)
            ci_lower.append(0)
            ci_upper.append(0)
    
    return episodes, np.array(means), np.array(ci_lower), np.array(ci_upper)


def smooth_with_ci(episodes, means, ci_lower, ci_upper, window=50):
    """Apply moving average smoothing to mean and CI bounds."""
    if len(means) < window:
        return episodes, means, ci_lower, ci_upper
    
    kernel = np.ones(window) / window
    smoothed_means = np.convolve(means, kernel, mode='valid')
    smoothed_lower = np.convolve(ci_lower, kernel, mode='valid')
    smoothed_upper = np.convolve(ci_upper, kernel, mode='valid')
    smoothed_episodes = episodes[window-1:]
    
    return smoothed_episodes, smoothed_means, smoothed_lower, smoothed_upper


def plot_comparison(random_stats, qlearning_stats, output_path, 
                    num_episodes_random, num_episodes_ql):
    """Generate comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Two-Agent Collaboration: Q-Learning vs Random Agents', 
                 fontsize=14, fontweight='bold')
    
    # Colors
    random_color = '#e74c3c'  # Red
    ql_color = '#3498db'  # Blue
    
    # =========================================================================
    # Plot 1: Learning Curves with Confidence Intervals
    # =========================================================================
    ax = axes[0, 0]
    
    # Random agents learning curve with CI
    r_eps, r_means, r_ci_lo, r_ci_hi = compute_learning_curve_with_ci(
        random_stats, num_episodes_random
    )
    
    # Q-Learning agents learning curve with CI
    q_eps, q_means, q_ci_lo, q_ci_hi = compute_learning_curve_with_ci(
        qlearning_stats, num_episodes_ql
    )
    
    # Smooth the curves
    window = 50
    r_eps_s, r_means_s, r_ci_lo_s, r_ci_hi_s = smooth_with_ci(r_eps, r_means, r_ci_lo, r_ci_hi, window)
    q_eps_s, q_means_s, q_ci_lo_s, q_ci_hi_s = smooth_with_ci(q_eps, q_means, q_ci_lo, q_ci_hi, window)
    
    # Plot Random agents
    ax.plot(r_eps_s, r_means_s, '-', linewidth=2.5, color=random_color, label='Random')
    ax.fill_between(r_eps_s, r_ci_lo_s, r_ci_hi_s, color=random_color, alpha=0.2)
    
    # Plot Q-Learning agents
    ax.plot(q_eps_s, q_means_s, '-', linewidth=2.5, color=ql_color, label='Q-Learning')
    ax.fill_between(q_eps_s, q_ci_lo_s, q_ci_hi_s, color=ql_color, alpha=0.2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title(f'Learning Curve (Mean ± 95% CI, smoothed window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # =========================================================================
    # Plot 2: Overall Success Rate Bar Chart
    # =========================================================================
    ax = axes[0, 1]
    
    # Compute per-seed success rates
    random_by_seed = random_stats.groupby('seed')['success'].mean() * 100
    ql_by_seed = qlearning_stats.groupby('seed')['success'].mean() * 100
    
    random_mean = random_by_seed.mean()
    random_std = random_by_seed.std()
    ql_mean = ql_by_seed.mean()
    ql_std = ql_by_seed.std()
    
    x = [0, 1]
    means = [random_mean, ql_mean]
    stds = [random_std, ql_std]
    colors = [random_color, ql_color]
    labels = ['Random\nAgents', 'Q-Learning\nAgents']
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Overall Success Rate (Mean ± Std)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(means) + max(stds) + 15)
    
    # =========================================================================
    # Plot 3: Reward Distribution
    # =========================================================================
    ax = axes[1, 0]
    
    random_rewards = random_stats['reward'].values
    ql_rewards = qlearning_stats['reward'].values
    
    bins = np.linspace(min(min(random_rewards), min(ql_rewards)),
                       max(max(random_rewards), max(ql_rewards)), 25)
    
    ax.hist(random_rewards, bins=bins, alpha=0.6, color=random_color, 
            label=f'Random (μ={np.mean(random_rewards):+.2f})', edgecolor='black')
    ax.hist(ql_rewards, bins=bins, alpha=0.6, color=ql_color,
            label=f'Q-Learning (μ={np.mean(ql_rewards):+.2f})', edgecolor='black')
    
    ax.axvline(np.mean(random_rewards), color=random_color, linestyle='--', linewidth=2)
    ax.axvline(np.mean(ql_rewards), color=ql_color, linestyle='--', linewidth=2)
    
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 4: Steps to Completion (Wins Only)
    # =========================================================================
    ax = axes[1, 1]
    
    random_win_steps = random_stats[random_stats['success']]['step'].values
    ql_win_steps = qlearning_stats[qlearning_stats['success']]['step'].values
    
    if len(random_win_steps) > 0 and len(ql_win_steps) > 0:
        bins = np.linspace(0, max(max(random_win_steps), max(ql_win_steps)) + 5, 20)
        
        ax.hist(random_win_steps, bins=bins, alpha=0.6, color=random_color,
                label=f'Random (n={len(random_win_steps)}, μ={np.mean(random_win_steps):.1f})',
                edgecolor='black')
        ax.hist(ql_win_steps, bins=bins, alpha=0.6, color=ql_color,
                label=f'Q-Learning (n={len(ql_win_steps)}, μ={np.mean(ql_win_steps):.1f})',
                edgecolor='black')
        
        ax.axvline(np.mean(random_win_steps), color=random_color, linestyle='--', linewidth=2)
        ax.axvline(np.mean(ql_win_steps), color=ql_color, linestyle='--', linewidth=2)
    
    ax.set_xlabel('Steps to Win')
    ax.set_ylabel('Frequency')
    ax.set_title('Efficiency (Steps for Successful Episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    return fig


def print_summary(random_stats, qlearning_stats):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Random Agents':<20} {'Q-Learning Agents':<20}")
    print("-" * 70)
    
    # Success rate
    random_sr = random_stats['success'].mean() * 100
    ql_sr = qlearning_stats['success'].mean() * 100
    print(f"{'Overall Success Rate':<30} {random_sr:>6.1f}%             {ql_sr:>6.1f}%")
    
    # Average reward
    random_reward = random_stats['reward'].mean()
    ql_reward = qlearning_stats['reward'].mean()
    print(f"{'Average Reward':<30} {random_reward:>+7.2f}             {ql_reward:>+7.2f}")
    
    # Average steps
    random_steps = random_stats['step'].mean()
    ql_steps = qlearning_stats['step'].mean()
    print(f"{'Average Steps':<30} {random_steps:>7.1f}             {ql_steps:>7.1f}")
    
    # Win steps only
    random_win_steps = random_stats[random_stats['success']]['step'].mean()
    ql_win_steps = qlearning_stats[qlearning_stats['success']]['step'].mean()
    print(f"{'Average Steps (Wins Only)':<30} {random_win_steps:>7.1f}             {ql_win_steps:>7.1f}")
    
    # Improvement
    improvement = ((ql_sr - random_sr) / random_sr) * 100 if random_sr > 0 else 0
    print(f"\n🎯 Q-Learning improvement over Random: {improvement:+.1f}%")
    
    # Per-seed breakdown
    print(f"\n{'Seed':<6} {'Random (%)':<15} {'Q-Learning (%)':<15} {'Improvement':<15}")
    print("-" * 55)
    
    random_by_seed = random_stats.groupby('seed')['success'].mean() * 100
    ql_by_seed = qlearning_stats.groupby('seed')['success'].mean() * 100
    
    for seed in sorted(set(random_by_seed.index) & set(ql_by_seed.index)):
        r = random_by_seed[seed]
        q = ql_by_seed[seed]
        imp = q - r
        print(f"{seed:<6} {r:>6.1f}          {q:>6.1f}          {imp:>+6.1f}")


def main():
    parser = argparse.ArgumentParser(description='Compare Q-Learning vs Random agents in two-agent environment')
    parser.add_argument('--random-csv', type=str, default=None,
                       help='Path to random agents CSV (default: latest in logs/)')
    parser.add_argument('--ql-csv', type=str, default=None,
                       help='Path to Q-learning agents CSV (default: latest in logs/)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: results/two_agent_comparison.png)')
    args = parser.parse_args()
    
    log_dir = project_root / "logs"
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("TWO-AGENT COMPARISON: Q-LEARNING vs RANDOM")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    
    if args.random_csv:
        random_df = pd.read_csv(args.random_csv)
        print(f"Random agents: {args.random_csv}")
    else:
        random_df, random_path = load_latest_csv("two_random_agents_*.csv", log_dir)
    
    if args.ql_csv:
        ql_df = pd.read_csv(args.ql_csv)
        print(f"Q-Learning agents: {args.ql_csv}")
    else:
        ql_df, ql_path = load_latest_csv("two_qlearning_agents_*.csv", log_dir)
    
    # Compute episode statistics
    print("\nComputing episode statistics...")
    random_stats = compute_episode_stats(random_df)
    ql_stats = compute_episode_stats(ql_df)
    
    num_episodes_random = random_stats['episode'].max()
    num_episodes_ql = ql_stats['episode'].max()
    
    print(f"Random agents: {len(random_stats)} episodes across {random_stats['seed'].nunique()} seeds")
    print(f"Q-Learning agents: {len(ql_stats)} episodes across {ql_stats['seed'].nunique()} seeds")
    
    # Print summary
    print_summary(random_stats, ql_stats)
    
    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    
    output_path = args.output if args.output else output_dir / "two_agent_comparison.png"
    fig = plot_comparison(random_stats, ql_stats, output_path,
                         num_episodes_random, num_episodes_ql)
    
    plt.show()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

