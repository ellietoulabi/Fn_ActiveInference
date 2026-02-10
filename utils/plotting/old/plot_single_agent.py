"""
Plot detailed analysis of a single agent's run across multiple episodes.

Usage:
    python utils/plotting/plot_single_agent.py <log_file.csv> [--episodes_per_config N]
    
Example:
    python utils/plotting/plot_single_agent.py logs/qlearning_log_ep25_step50_20251104_153000.csv --episodes_per_config 5
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys
import re

# Set publication-quality plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def detect_agent_type(log_df, log_filename=None):
    """Detect agent type from log filename or data."""
    # Check if 'agent' column exists (comparison logs)
    if 'agent' in log_df.columns:
        agents = log_df['agent'].unique()
        if len(agents) == 1:
            return agents[0]
        else:
            return 'Mixed'
    
    # Try to detect from filename
    if log_filename:
        filename = Path(log_filename).stem.lower()
        if 'qlearning' in filename or 'ql_' in filename:
            return 'Q-Learning'
        elif 'dynaq' in filename:
            return 'DynaQ'
        elif 'active' in filename or 'aif' in filename:
            return 'Active Inference'
    
    # Otherwise return 'Unknown'
    return 'Unknown'


def detect_episodes_per_config(log_filename):
    """
    Try to detect episodes_per_config from filename patterns like:
    - comparison_ep50_step50_config5_...
    - qlearning_log_ep25_step50_...
    """
    filename = Path(log_filename).stem
    
    # Pattern: config followed by a number
    config_match = re.search(r'config(\d+)', filename, re.IGNORECASE)
    if config_match:
        return int(config_match.group(1))
    
    # Pattern: epc followed by a number (episodes per config)
    epc_match = re.search(r'epc(\d+)', filename, re.IGNORECASE)
    if epc_match:
        return int(epc_match.group(1))
    
    return None


def calculate_episode_stats(log_df):
    """Calculate statistics per episode."""
    episodes = []
    
    for ep in sorted(log_df['episode'].unique()):
        ep_df = log_df[log_df['episode'] == ep]
        
        total_reward = ep_df['reward'].sum()
        num_steps = len(ep_df)
        
        # Determine if episode was successful (total reward >= 1.5 means win)
        success = 1 if total_reward >= 1.5 else 0
        
        episodes.append({
            'episode': ep,
            'total_reward': total_reward,
            'steps': num_steps,
            'success': success,
            'avg_reward_per_step': total_reward / num_steps if num_steps > 0 else 0
        })
    
    return pd.DataFrame(episodes)


def main():
    parser = argparse.ArgumentParser(description='Plot single agent analysis')
    parser.add_argument('log_file', help='Path to agent CSV log file')
    parser.add_argument('--output', default=None,
                       help='Output filename (default: auto-generated)')
    parser.add_argument('--episodes_per_config', type=int, default=200,
                       help='Episodes per configuration (for non-stationary environments)')
    
    args = parser.parse_args()
    
    # Load the log file
    if not Path(args.log_file).exists():
        print(f"Error: Log file '{args.log_file}' not found!")
        sys.exit(1)
    
    print(f"Loading log file: {args.log_file}")
    log_df = pd.read_csv(args.log_file)
    
    # Detect or use provided episodes_per_config
    episodes_per_config = args.episodes_per_config
    if episodes_per_config is None:
        episodes_per_config = detect_episodes_per_config(args.log_file)
        if episodes_per_config:
            print(f"Detected episodes_per_config: {episodes_per_config}")
    
    # Filter to single agent if it's a comparison log
    if 'agent' in log_df.columns:
        agents = log_df['agent'].unique()
        if len(agents) > 1:
            print(f"\nMultiple agents detected: {agents}")
            print("Please specify which agent to plot:")
            agent_choice = input("Enter agent name (or 'all' for separate plots): ").strip()
            if agent_choice != 'all':
                log_df = log_df[log_df['agent'] == agent_choice].copy()
                agent_type = agent_choice
            else:
                # Plot each agent separately
                for agent in agents:
                    agent_df = log_df[log_df['agent'] == agent].copy()
                    plot_agent_analysis(agent_df, agent, args.log_file, args.output, episodes_per_config)
                return
        else:
            agent_type = agents[0]
            log_df = log_df[log_df['agent'] == agent_type].copy()
    else:
        agent_type = detect_agent_type(log_df, args.log_file)
    
    plot_agent_analysis(log_df, agent_type, args.log_file, args.output, episodes_per_config)


def plot_agent_analysis(log_df, agent_type, log_file, output_path=None, episodes_per_config=None):
    """Create comprehensive analysis plots for a single agent."""
    
    # Calculate episode statistics
    episode_stats = calculate_episode_stats(log_df)
    
    print(f"\n{'='*80}")
    print(f"Analyzing {agent_type} Agent")
    print(f"{'='*80}")
    print(f"Total episodes: {len(episode_stats)}")
    print(f"Total steps: {len(log_df)}")
    print(f"Total reward: {episode_stats['total_reward'].sum():.2f}")
    print(f"Success rate: {episode_stats['success'].mean()*100:.1f}%")
    print(f"Average reward per episode: {episode_stats['total_reward'].mean():.2f}")
    print(f"Average steps per episode: {episode_stats['steps'].mean():.1f}")
    if episodes_per_config:
        num_configs = len(episode_stats) // episodes_per_config
        print(f"Environment configurations: {num_configs} (changing every {episodes_per_config} episodes)")
    
    # Calculate cumulative success rate (rolling window for smoothing)
    window_size = max(1, len(episode_stats) // 20)  # 5% of episodes
    episode_stats['success_rate'] = episode_stats['success'].rolling(
        window=window_size, min_periods=1).mean() * 100
    
    # Calculate rolling average episode length
    episode_stats['avg_length'] = episode_stats['steps'].rolling(
        window=window_size, min_periods=1).mean()
    
    # Calculate rolling statistics for convergence analysis
    episode_stats['reward_mean'] = episode_stats['total_reward'].rolling(
        window=window_size, min_periods=1).mean()
    episode_stats['reward_std'] = episode_stats['total_reward'].rolling(
        window=window_size, min_periods=1).std()
    episode_stats['success_std'] = episode_stats['success'].rolling(
        window=window_size, min_periods=1).std() * 100
    episode_stats['length_std'] = episode_stats['steps'].rolling(
        window=window_size, min_periods=1).std()
    
    # Create figure with 5 subplots (3 rows: 1 wide plot on top, then 2x2 grid)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
    
    # Colorful colors for better visualization
    colors = {
        'AIF': '#2E86AB',           # Blue
        'Active Inference': '#2E86AB',
        'QL': '#A23B72',            # Purple/Magenta
        'Q-Learning': '#A23B72',
        'DynaQ': '#06A77D',         # Teal/Green
        'Unknown': '#F18F01'        # Orange
    }
    agent_color = colors.get(agent_type, colors['Unknown'])
    
    fig.suptitle(f'{agent_type} Performance ({len(episode_stats)} Episodes)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Helper function to add config change boundaries and set x-ticks
    def add_config_boundaries(ax, min_ep, max_ep):
        """Add vertical lines at configuration change boundaries and set x-ticks."""
        if episodes_per_config:
            # Add vertical lines at boundaries
            for config_ep in range(episodes_per_config, int(max_ep), episodes_per_config):
                ax.axvline(x=config_ep, color='orange', linestyle=':', 
                          linewidth=1.5, alpha=0.4, zorder=1)
            
            # Set x-ticks at config boundaries
            tick_positions = [0] + list(range(episodes_per_config, int(max_ep) + 1, episodes_per_config))
            if tick_positions[-1] < max_ep:
                tick_positions.append(int(max_ep))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(int(t)) for t in tick_positions])
    
    max_ep = episode_stats['episode'].max()
    
    # Plot 1: Average Reward Convergence (Top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Calculate CUMULATIVE average reward (true convergence metric)
    # This is the average of ALL rewards from episode 1 to current episode
    cumsum_reward = episode_stats['total_reward'].cumsum()
    episode_numbers = np.arange(1, len(episode_stats) + 1)
    cumulative_avg_reward = cumsum_reward / episode_numbers
    
    # Plot raw episode rewards (very light, for context)
    ax1.plot(episode_stats['episode'], episode_stats['total_reward'], 
             linestyle='-', linewidth=0.8, color=agent_color, alpha=0.15, zorder=1)
    
    # Plot cumulative average (CONVERGENCE line - this is the key plot!)
    ax1.plot(episode_stats['episode'], cumulative_avg_reward,
             linestyle='-', linewidth=3.5, color=agent_color, alpha=0.95, zorder=3,
             label='Cumulative average')
    
    # Reference lines
    ax1.axhline(y=1.5, color='#06A77D', linestyle='--', alpha=0.3, linewidth=1.5, label='Win reward')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.axhline(y=-0.5, color='#E63946', linestyle='--', alpha=0.3, linewidth=1.5, label='Loss reward')
    
    add_config_boundaries(ax1, episode_stats['episode'].min(), episode_stats['episode'].max())
    
    ax1.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax1.set_title('Average Reward Convergence', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_xlim(0, max_ep + 1)
    ax1.set_ylim(-0.8, 2.3)
    
    # Plot 2: Cumulative Reward Over Time (Second row, left)
    ax2 = fig.add_subplot(gs[1, 0])
    cumulative = episode_stats['total_reward'].cumsum()
    
    ax2.plot(episode_stats['episode'], cumulative,
             linestyle='-', linewidth=2.5, marker='o', markersize=4,
             color=agent_color, alpha=0.9, zorder=3)
    
    # Fill area under curve
    ax2.fill_between(episode_stats['episode'], cumulative, alpha=0.15, color=agent_color)
    
    # Add annotation for final value
    final_reward = cumulative.iloc[-1]
    ax2.text(max_ep, final_reward, f'{final_reward:.1f}', fontsize=11, fontweight='bold', 
             ha='left', va='bottom', color=agent_color)
    
    add_config_boundaries(ax2, episode_stats['episode'].min(), episode_stats['episode'].max())
    
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_ep + 1)
    
    # Plot 3: Step Count per Episode (Second row, right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(episode_stats['episode'], episode_stats['steps'],
             marker='o', linestyle='-', linewidth=2, markersize=4,
             color=agent_color, alpha=0.8)
    
    # Add max steps line if available
    if 'max_steps' in log_df.columns:
        max_steps_value = log_df['max_steps'].max()
        ax3.axhline(y=max_steps_value, color='red', linestyle='--', alpha=0.3, label=f'Max steps ({max_steps_value})')
    
    add_config_boundaries(ax3, episode_stats['episode'].min(), episode_stats['episode'].max())
    
    ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Steps to Completion', fontsize=12, fontweight='bold')
    ax3.set_title('Episode Length (Steps)', fontsize=14, fontweight='bold')
    if 'max_steps' in log_df.columns:
        ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_ep + 1)
    
    # Plot 4: Win/Loss per Episode (Third row, left)
    ax4 = fig.add_subplot(gs[2, 0])
    episodes_range = episode_stats['episode'].values
    bar_width = 0.8
    
    # Create outcome values: +1 for win, -1 for loss
    outcomes = episode_stats['success'].apply(lambda x: 1 if x else -1)
    
    # Create colors: green for wins (+1), red for losses (-1)
    bar_colors = ['#06A77D' if outcome == 1 else '#E63946' for outcome in outcomes]
    
    # Plot bars
    ax4.bar(episodes_range, outcomes, bar_width, 
            color=bar_colors, alpha=0.85, edgecolor='none')
    
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#06A77D', label='Win (up)', alpha=0.85),
        Patch(facecolor='#E63946', label='Loss (down)', alpha=0.85)
    ]
    ax4.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    
    add_config_boundaries(ax4, episode_stats['episode'].min(), episode_stats['episode'].max())
    
    ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Outcome', fontsize=12, fontweight='bold')
    ax4.set_title('Win/Loss per Episode', fontsize=14, fontweight='bold')
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['Loss', '0', 'Win'])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(0, max_ep + 1)
    ax4.set_ylim(-1.3, 1.3)
    
    # Plot 5: Success Rate (Third row, right) - Cumulative only
    ax5 = fig.add_subplot(gs[2, 1])
    
    cumsum = episode_stats['success'].cumsum()
    cumrate = 100 * cumsum / np.arange(1, len(episode_stats) + 1)
    
    ax5.plot(episode_stats['episode'], cumrate,
             linestyle='-', linewidth=3,
             color=agent_color, alpha=0.9, zorder=3)
    
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='50% baseline')
    
    add_config_boundaries(ax5, episode_stats['episode'].min(), episode_stats['episode'].max())
    
    ax5.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Cumulative Success Rate', fontsize=14, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, max_ep + 1)
    ax5.set_ylim(-5, 105)
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        save_path = output_path
    else:
        # Auto-generate filename from log file
        log_path = Path(log_file)
        save_path = log_path.parent / f"{log_path.stem}_analysis.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved to: {save_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("Episode Statistics:")
    print(f"{'='*80}")
    print(f"Best episode reward: {episode_stats['total_reward'].max():.2f} (Episode {episode_stats.loc[episode_stats['total_reward'].idxmax(), 'episode']:.0f})")
    print(f"Worst episode reward: {episode_stats['total_reward'].min():.2f} (Episode {episode_stats.loc[episode_stats['total_reward'].idxmin(), 'episode']:.0f})")
    print(f"Shortest episode: {episode_stats['steps'].min():.0f} steps (Episode {episode_stats.loc[episode_stats['steps'].idxmin(), 'episode']:.0f})")
    print(f"Longest episode: {episode_stats['steps'].max():.0f} steps (Episode {episode_stats.loc[episode_stats['steps'].idxmax(), 'episode']:.0f})")
    print(f"Wins: {episode_stats['success'].sum():.0f}/{len(episode_stats)} ({episode_stats['success'].mean()*100:.1f}%)")
    print(f"{'='*80}\n")
    
    plt.close(fig)


if __name__ == "__main__":
    main()

