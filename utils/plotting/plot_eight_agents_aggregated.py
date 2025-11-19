"""
Plot aggregated comparison across multiple seeds for 8 agents.

Loads multi-seed data and plots mean values for each agent.
Shows the statistical reliability of the comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats


def load_and_process_multiseed_log(log_file):
    """Load multi-seed comparison log and aggregate by seed."""
    df = pd.read_csv(log_file)
    
    # Check if seed column exists
    if 'seed' not in df.columns:
        raise ValueError("Log file must contain 'seed' column. Use compare_eight_agents.py with multiple seeds.")
    
    # Get unique seeds and agents
    seeds = df['seed'].unique()
    agent_names = df['agent'].unique()
    
    print(f"Found {len(seeds)} seeds: {seeds}")
    print(f"Found {len(agent_names)} agents: {', '.join(agent_names)}")
    
    # Calculate episode statistics for each seed × agent combination
    def calc_episode_stats(agent_seed_df):
        episode_stats = agent_seed_df.groupby('episode').agg({
            'reward': 'sum',
            'step': 'max'
        }).reset_index()
        
        episode_stats.columns = ['episode', 'total_reward', 'steps']
        episode_stats['success'] = episode_stats['total_reward'] > 1.0
        
        return episode_stats
    
    # Process each agent × seed combination
    all_data = {}
    for agent_name in agent_names:
        all_data[agent_name] = {}
        for seed in seeds:
            agent_seed_df = df[(df['agent'] == agent_name) & (df['seed'] == seed)].copy()
            all_data[agent_name][seed] = calc_episode_stats(agent_seed_df)
    
    return all_data, agent_names, seeds


def aggregate_across_seeds(all_data, agent_names, seeds):
    """Aggregate statistics across seeds for each agent."""
    aggregated = {}
    
    for agent_name in agent_names:
        # Collect data from all seeds
        seed_data = [all_data[agent_name][seed] for seed in seeds]
        
        # Get episode numbers (should be same for all seeds)
        episodes = seed_data[0]['episode'].values
        
        # Aggregate metrics across seeds
        n_episodes = len(episodes)
        n_seeds = len(seeds)
        
        # Initialize arrays
        rewards_matrix = np.zeros((n_seeds, n_episodes))
        steps_matrix = np.zeros((n_seeds, n_episodes))
        success_matrix = np.zeros((n_seeds, n_episodes))
        
        for seed_idx, seed_df in enumerate(seed_data):
            rewards_matrix[seed_idx, :] = seed_df['total_reward'].values
            steps_matrix[seed_idx, :] = seed_df['steps'].values
            success_matrix[seed_idx, :] = seed_df['success'].values.astype(int)
        
        # Calculate statistics
        # Using z-score = 2.576 for 99% confidence interval
        aggregated[agent_name] = {
            'episodes': episodes,
            'rewards': {
                'mean': np.mean(rewards_matrix, axis=0),
                'std': np.std(rewards_matrix, axis=0),
                'ci_lower': np.mean(rewards_matrix, axis=0) - 2.576 * np.std(rewards_matrix, axis=0) / np.sqrt(n_seeds),
                'ci_upper': np.mean(rewards_matrix, axis=0) + 2.576 * np.std(rewards_matrix, axis=0) / np.sqrt(n_seeds)
            },
            'steps': {
                'mean': np.mean(steps_matrix, axis=0),
                'std': np.std(steps_matrix, axis=0),
                'ci_lower': np.mean(steps_matrix, axis=0) - 2.576 * np.std(steps_matrix, axis=0) / np.sqrt(n_seeds),
                'ci_upper': np.mean(steps_matrix, axis=0) + 2.576 * np.std(steps_matrix, axis=0) / np.sqrt(n_seeds)
            },
            'success': {
                'mean': np.mean(success_matrix, axis=0) * 100,  # Convert to percentage
                'std': np.std(success_matrix, axis=0) * 100,
                'ci_lower': (np.mean(success_matrix, axis=0) - 2.576 * np.std(success_matrix, axis=0) / np.sqrt(n_seeds)) * 100,
                'ci_upper': (np.mean(success_matrix, axis=0) + 2.576 * np.std(success_matrix, axis=0) / np.sqrt(n_seeds)) * 100
            }
        }
    
    return aggregated


def plot_aggregated_comparison(aggregated, agent_names, episodes_per_config=None, output_path=None):
    """Create aggregated comparison plots with confidence intervals."""
    
    # Create figure with 5 subplots (3 rows: 1 wide plot on top, then 2x2 grid)
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
    
    # Colors for 8 agents
    colors = {
        'AIF': '#1A1A1A',          # Dark Gray/Black
        'QLearning': '#000000',      # Black
        'Vanilla': '#2E86AB',         # Blue
        'Recency0.99': '#06A77D',     # Green
        'Recency0.95': '#A23B72',     # Magenta
        'Recency0.9': '#F18F01',      # Orange
        'Recency0.85': '#E63946',     # Red
        'TrajSampling': '#7209B7'     # Purple
    }
    
    fig.suptitle(f'Aggregated Comparison Across Multiple Seeds (mean values)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Helper function to add config change boundaries
    def add_config_boundaries(ax, max_ep):
        if episodes_per_config:
            for config_ep in range(episodes_per_config, int(max_ep), episodes_per_config):
                ax.axvline(x=config_ep, color='gray', linestyle=':', 
                          linewidth=0.5, alpha=0.3, zorder=1)
    
    episodes = aggregated[agent_names[0]]['episodes']
    max_ep = episodes.max()
    
    # Plot 1: Average Reward Convergence (Top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    
    for agent_name in agent_names:
        data = aggregated[agent_name]
        
        # Calculate cumulative average
        cumsum = np.cumsum(data['rewards']['mean'])
        cumavg_mean = cumsum / np.arange(1, len(cumsum) + 1)
        
        # Calculate CI for cumulative average (approximate)
        cumsum_ci_lower = np.cumsum(data['rewards']['ci_lower'])
        cumsum_ci_upper = np.cumsum(data['rewards']['ci_upper'])
        cumavg_ci_lower = cumsum_ci_lower / np.arange(1, len(cumsum_ci_lower) + 1)
        cumavg_ci_upper = cumsum_ci_upper / np.arange(1, len(cumsum_ci_upper) + 1)
        
        # Plot mean line
        ax1.plot(episodes, cumavg_mean,
                 linestyle='-',
                 linewidth=2.5,
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.9,
                 label=agent_name,
                 zorder=3)
        
        # Plot confidence interval (commented out)
        # ax1.fill_between(episodes, cumavg_ci_lower, cumavg_ci_upper,
        #                 color=colors.get(agent_name, '#888888'),
        #                 alpha=0.15,
        #                 zorder=2)
    
    # Reference lines
    ax1.axhline(y=1.5, color='#06A77D', linestyle='--', alpha=0.3, linewidth=1.5, label='Win reward')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.axhline(y=-0.5, color='#E63946', linestyle='--', alpha=0.3, linewidth=1.5, label='Loss reward')
    
    add_config_boundaries(ax1, max_ep)
    
    ax1.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax1.set_title('Average Reward Convergence', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_xlim(0, max_ep + 1)
    ax1.set_ylim(-0.8, 2.3)
    
    # Plot 2: Cumulative Reward Over Time (Second row, left)
    ax2 = fig.add_subplot(gs[1, 0])
    
    for agent_name in agent_names:
        data = aggregated[agent_name]
        cumsum_mean = np.cumsum(data['rewards']['mean'])
        cumsum_ci_lower = np.cumsum(data['rewards']['ci_lower'])
        cumsum_ci_upper = np.cumsum(data['rewards']['ci_upper'])
        
        ax2.plot(episodes, cumsum_mean,
                 linestyle='-',
                 linewidth=2.5,
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.9,
                 label=agent_name)
        
        # ax2.fill_between(episodes, cumsum_ci_lower, cumsum_ci_upper,
        #                 color=colors.get(agent_name, '#888888'),
        #                 alpha=0.15)
    
    add_config_boundaries(ax2, max_ep)
    
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_ep + 1)
    
    # Plot 3: Episode Length (Second row, right)
    ax3 = fig.add_subplot(gs[1, 1])
    
    for agent_name in agent_names:
        data = aggregated[agent_name]
        
        ax3.plot(episodes, data['steps']['mean'],
                 linestyle='-',
                 linewidth=2.5,
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.9,
                 label=agent_name)
        
        # ax3.fill_between(episodes, data['steps']['ci_lower'], data['steps']['ci_upper'],
        #                 color=colors.get(agent_name, '#888888'),
        #                 alpha=0.15)
    
    add_config_boundaries(ax3, max_ep)
    
    ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Steps to Completion', fontsize=12, fontweight='bold')
    ax3.set_title('Episode Length (Steps)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_ep + 1)
    
    # Plot 4: Cumulative Success Rate (Third row, left)
    ax4 = fig.add_subplot(gs[2, 0])
    
    for agent_name in agent_names:
        data = aggregated[agent_name]
        
        # Calculate cumulative success rate
        success_array = data['success']['mean'] / 100  # Convert back to 0-1
        cumsum_success = np.cumsum(success_array)
        cumrate_mean = 100 * cumsum_success / np.arange(1, len(cumsum_success) + 1)
        
        # Approximate CI for cumulative rate
        success_ci_lower = data['success']['ci_lower'] / 100
        success_ci_upper = data['success']['ci_upper'] / 100
        cumsum_ci_lower = np.cumsum(success_ci_lower)
        cumsum_ci_upper = np.cumsum(success_ci_upper)
        cumrate_ci_lower = 100 * cumsum_ci_lower / np.arange(1, len(cumsum_ci_lower) + 1)
        cumrate_ci_upper = 100 * cumsum_ci_upper / np.arange(1, len(cumsum_ci_upper) + 1)
        
        ax4.plot(episodes, cumrate_mean,
                 linestyle='-',
                 linewidth=2.5,
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.9,
                 label=agent_name)
        
        # ax4.fill_between(episodes, cumrate_ci_lower, cumrate_ci_upper,
        #                 color=colors.get(agent_name, '#888888'),
        #                 alpha=0.15)
    
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='50% baseline')
    
    add_config_boundaries(ax4, max_ep)
    
    ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Cumulative Success Rate', fontsize=14, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max_ep + 1)
    ax4.set_ylim(-5, 105)
    
    # Plot 5: Episode Rewards with CI (Third row, right)
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Use rolling average for smoothness
    window = 20
    for agent_name in agent_names:
        data = aggregated[agent_name]
        
        # Rolling average of episode rewards
        rewards_mean = data['rewards']['mean']
        rolling_mean = pd.Series(rewards_mean).rolling(window=window, min_periods=1).mean().values
        
        # Approximate rolling CI
        ci_lower_rolling = pd.Series(data['rewards']['ci_lower']).rolling(window=window, min_periods=1).mean().values
        ci_upper_rolling = pd.Series(data['rewards']['ci_upper']).rolling(window=window, min_periods=1).mean().values
        
        ax5.plot(episodes, rolling_mean,
                 linestyle='-',
                 linewidth=2.5,
                 color=colors.get(agent_name, '#888888'),
                 alpha=0.9,
                 label=agent_name)
        
        # ax5.fill_between(episodes, ci_lower_rolling, ci_upper_rolling,
        #                 color=colors.get(agent_name, '#888888'),
        #                 alpha=0.15)
    
    ax5.axhline(y=1.5, color='#06A77D', linestyle='--', alpha=0.3, linewidth=1.5, label='Win')
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
    ax5.axhline(y=-0.5, color='#E63946', linestyle='--', alpha=0.3, linewidth=1.5, label='Loss')
    
    add_config_boundaries(ax5, max_ep)
    
    ax5.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax5.set_title(f'Episode Rewards ({window}-ep rolling avg)', fontsize=14, fontweight='bold')
    ax5.legend(loc='best', fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, max_ep + 1)
    ax5.set_ylim(-1.0, 2.5)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Aggregated plot saved to: {output_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot aggregated multi-seed comparison (mean values) for 8 agents')
    parser.add_argument('log_file', type=str, help='Path to multi-seed comparison CSV log file')
    parser.add_argument('--episodes_per_config', type=int, default=20,
                       help='Number of episodes per config (for vertical lines)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: auto-generate from log filename)')
    
    args = parser.parse_args()
    
    print(f"Loading multi-seed comparison log: {args.log_file}")
    all_data, agent_names, seeds = load_and_process_multiseed_log(args.log_file)
    
    print(f"\n{'='*80}")
    print("Aggregating data across seeds...")
    print(f"{'='*80}")
    
    aggregated = aggregate_across_seeds(all_data, agent_names, seeds)
    
    print(f"\nCreating plots (mean values across seeds)...")
    
    # Generate output path if not provided
    output_path = args.output
    if output_path is None:
        log_path = Path(args.log_file)
        output_path = log_path.parent / f"{log_path.stem}_aggregated.png"
    
    plot_aggregated_comparison(aggregated, agent_names, args.episodes_per_config, output_path)
    
    print(f"\n✓ Aggregated comparison complete!")
    print(f"  Mean values aggregated across {len(seeds)} seeds")


if __name__ == "__main__":
    main()




