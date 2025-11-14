"""
Plot comparison between Q-Learning, Vanilla Dyna-Q, and Dyna-Q with Multiple Recency Biases.

Shows side-by-side comparison of 6 agents' performance metrics:
- Q-Learning (no planning)
- Vanilla Dyna-Q
- Recency 0.99, 0.95, 0.90, 0.85
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_and_process_comparison_log(log_file):
    """Load comparison log and separate by agent."""
    df = pd.read_csv(log_file)
    
    # Get unique agent names
    agent_names = df['agent'].unique()
    print(f"Found {len(agent_names)} agents: {', '.join(agent_names)}")
    
    # Calculate episode statistics for each agent
    def calc_episode_stats(agent_df):
        episode_stats = agent_df.groupby('episode').agg({
            'reward': 'sum',
            'step': 'max'
        }).reset_index()
        
        episode_stats.columns = ['episode', 'total_reward', 'steps']
        
        # Determine success (win = reward > 1.0)
        episode_stats['success'] = episode_stats['total_reward'] > 1.0
        
        return episode_stats
    
    # Process each agent
    all_agent_stats = {}
    for agent_name in agent_names:
        agent_df = df[df['agent'] == agent_name].copy()
        all_agent_stats[agent_name] = calc_episode_stats(agent_df)
    
    return all_agent_stats


def plot_comparison(all_agent_stats, episodes_per_config=None, output_path=None, log_file=None):
    """Create comparison plots for all agents."""
    
    # Create figure with 5 subplots (3 rows: 1 wide plot on top, then 2x2 grid)
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
    
    # Colors for 6 agents (distinct and colorblind-friendly)
    colors = {
        'QLearning': '#000000',      # Black (baseline)
        'Vanilla': '#2E86AB',        # Blue
        'Recency0.99': '#06A77D',    # Teal
        'Recency0.95': '#A23B72',    # Purple/Magenta
        'Recency0.9': '#F18F01',     # Orange
        'Recency0.85': '#E63946'     # Red
    }
    
    # Line styles for variety
    linestyles = {
        'QLearning': '-',      # Solid (baseline)
        'Vanilla': '-',        # Solid
        'Recency0.99': '--',   # Dashed
        'Recency0.95': '-',    # Solid
        'Recency0.9': '--',    # Dashed
        'Recency0.85': ':'     # Dotted
    }
    
    # Linewidths for visibility
    linewidths = {
        'QLearning': 3.0,      # Thicker for baseline
        'Vanilla': 2.5,
        'Recency0.99': 2.0,
        'Recency0.95': 2.5,
        'Recency0.9': 2.0,
        'Recency0.85': 2.5
    }
    
    agent_names = list(all_agent_stats.keys())
    
    fig.suptitle(f'Q-Learning vs Dyna-Q Variants Comparison ({len(agent_names)} agents)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Helper function to add config change boundaries
    def add_config_boundaries(ax, min_ep, max_ep):
        """Add vertical lines at configuration change boundaries."""
        if episodes_per_config:
            for config_ep in range(episodes_per_config, int(max_ep), episodes_per_config):
                ax.axvline(x=config_ep, color='gray', linestyle=':', 
                          linewidth=1.5, alpha=0.3, zorder=1)
            
            tick_positions = [0] + list(range(episodes_per_config, int(max_ep) + 1, episodes_per_config))
            if tick_positions[-1] < max_ep:
                tick_positions.append(int(max_ep))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(int(t)) for t in tick_positions])
    
    max_ep = max(stats['episode'].max() for stats in all_agent_stats.values())
    
    # Plot 1: Average Reward Convergence (Top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot convergence line for each agent
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        cumsum = stats['total_reward'].cumsum()
        cumavg = cumsum / np.arange(1, len(stats) + 1)
        
        ax1.plot(stats['episode'], cumavg,
                 linestyle=linestyles.get(agent_name, '-'),
                 linewidth=linewidths.get(agent_name, 2.0),
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.9,
                 zorder=3,
                 label=agent_name)
    
    # Reference lines
    ax1.axhline(y=1.5, color='#06A77D', linestyle='--', alpha=0.3, linewidth=1.5, label='Win reward')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.axhline(y=-0.5, color='#E63946', linestyle='--', alpha=0.3, linewidth=1.5, label='Loss reward')
    
    add_config_boundaries(ax1, 0, max_ep)
    
    ax1.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax1.set_title('Average Reward Convergence', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_xlim(0, max_ep + 1)
    ax1.set_ylim(-0.8, 2.3)
    
    # Plot 2: Cumulative Reward Over Time (Second row, left)
    ax2 = fig.add_subplot(gs[1, 0])
    
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        cumsum = stats['total_reward'].cumsum()
        
        ax2.plot(stats['episode'], cumsum,
                 linestyle=linestyles.get(agent_name, '-'),
                 linewidth=linewidths.get(agent_name, 2.0),
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.9,
                 zorder=3,
                 label=agent_name)
    
    add_config_boundaries(ax2, 0, max_ep)
    
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_ep + 1)
    
    # Plot 3: Episode Length (Second row, right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        ax3.plot(stats['episode'], stats['steps'],
                 linestyle=linestyles.get(agent_name, '-'),
                 linewidth=linewidths.get(agent_name, 2.0),
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.8,
                 label=agent_name)
    
    add_config_boundaries(ax3, 0, max_ep)
    
    ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Steps to Completion', fontsize=12, fontweight='bold')
    ax3.set_title('Episode Length (Steps)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_ep + 1)
    
    # Plot 4: Success Rate (Third row, left)
    ax4 = fig.add_subplot(gs[2, 0])
    
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        cumsum_success = stats['success'].cumsum()
        cumrate = 100 * cumsum_success / np.arange(1, len(stats) + 1)
        
        ax4.plot(stats['episode'], cumrate,
                 linestyle=linestyles.get(agent_name, '-'),
                 linewidth=linewidths.get(agent_name, 2.0),
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.9,
                 zorder=3,
                 label=agent_name)
    
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='50% baseline')
    
    add_config_boundaries(ax4, 0, max_ep)
    
    ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Cumulative Success Rate', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max_ep + 1)
    ax4.set_ylim(-5, 105)
    
    # Plot 5: Win Rate per Configuration (Third row, right)
    ax5 = fig.add_subplot(gs[2, 1])
    
    if episodes_per_config:
        num_configs = int(max_ep / episodes_per_config)
        config_indices = np.arange(1, num_configs + 1)
        bar_width = 0.12  # Width per bar (6 agents)
        
        for i, agent_name in enumerate(agent_names):
            stats = all_agent_stats[agent_name]
            config_rates = []
            
            for config_idx in range(num_configs):
                start = config_idx * episodes_per_config
                end = start + episodes_per_config
                config_episodes = stats[(stats['episode'] >= start + 1) & (stats['episode'] <= end)]
                rate = 100 * config_episodes['success'].sum() / len(config_episodes) if len(config_episodes) > 0 else 0
                config_rates.append(rate)
            
            offset = (i - len(agent_names)/2 + 0.5) * bar_width
            ax5.bar(config_indices + offset, config_rates, bar_width,
                    color=colors.get(agent_name, '#888888'),
                    alpha=0.8,
                    label=agent_name)
        
        ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='50% baseline')
        ax5.set_xticks(config_indices)
        ax5.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    else:
        # Fallback: show rolling success rate
        window = 20
        for agent_name in agent_names:
            stats = all_agent_stats[agent_name]
            rolling = stats['success'].rolling(window=window, min_periods=1).mean() * 100
            ax5.plot(stats['episode'], rolling,
                     linestyle=linestyles.get(agent_name, '-'),
                     linewidth=linewidths.get(agent_name, 2.0),
                     color=colors.get(agent_name, '#888888'),
                     alpha=0.8,
                     label=agent_name)
        
        ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Episode', fontsize=12, fontweight='bold')
    
    ax5.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Success Rate per Configuration' if episodes_per_config else 'Rolling Success Rate', fontsize=14, fontweight='bold')
    ax5.legend(loc='best', fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-5, 105)
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        save_path = output_path
    else:
        log_path = Path(log_file)
        save_path = log_path.parent / f"{log_path.stem}_comparison.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved to: {save_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("Comparison Statistics:")
    print(f"{'='*80}")
    
    agent_names = list(all_agent_stats.keys())
    
    # Header
    print(f"\n{'Metric':<25} ", end='')
    for agent_name in agent_names:
        print(f"{agent_name:<16}", end='')
    print()
    print(f"{'-' * (25 + 16 * len(agent_names))}")
    
    # Wins
    print(f"{'Wins':<25} ", end='')
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        wins = stats['success'].sum()
        rate = 100 * wins / len(stats)
        print(f"{wins}/{len(stats)} ({rate:.1f}%){'':<3}", end='')
    print()
    
    # Best episode reward
    print(f"{'Best episode reward':<25} ", end='')
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        best = stats['total_reward'].max()
        best_ep = stats.loc[stats['total_reward'].idxmax(), 'episode']
        print(f"{best:.2f} (Ep {best_ep:.0f}){'':<4}", end='')
    print()
    
    # Shortest episode
    print(f"{'Shortest episode':<25} ", end='')
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        min_steps = stats['steps'].min()
        min_ep = stats.loc[stats['steps'].idxmin(), 'episode']
        print(f"{min_steps:.0f} steps (Ep {min_ep:.0f}){'':<0}", end='')
    print()
    
    # Average reward
    print(f"{'Average reward':<25} ", end='')
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        avg = stats['total_reward'].mean()
        print(f"{avg:+.3f}{'':<10}", end='')
    print()
    
    # Average steps
    print(f"{'Average steps':<25} ", end='')
    for agent_name in agent_names:
        stats = all_agent_stats[agent_name]
        avg = stats['steps'].mean()
        print(f"{avg:.1f}{'':<12}", end='')
    print()
    
    print(f"\n{'='*80}\n")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot 6-agent comparison from log file')
    parser.add_argument('log_file', type=str, help='Path to comparison CSV log file')
    parser.add_argument('--episodes_per_config', type=int, default=20,
                       help='Number of episodes per config (for vertical lines)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: auto-generate from log filename)')
    
    args = parser.parse_args()
    
    print(f"Loading comparison log: {args.log_file}")
    all_agent_stats = load_and_process_comparison_log(args.log_file)
    
    print(f"\n{'='*80}")
    print("Loaded Data:")
    print(f"{'='*80}")
    
    for agent_name, stats in all_agent_stats.items():
        print(f"{agent_name}: {len(stats)} episodes")
    
    if args.episodes_per_config:
        num_episodes = len(next(iter(all_agent_stats.values())))
        num_configs = num_episodes // args.episodes_per_config
        print(f"Environment configurations: {num_configs} (changing every {args.episodes_per_config} episodes)")
    
    plot_comparison(all_agent_stats, args.episodes_per_config, args.output, args.log_file)


if __name__ == "__main__":
    main()

