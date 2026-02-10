"""
Compare Active Inference vs Q-Learning vs Dyna-Q agents from log files.
Creates 5 comparison plots:
  1. Cumulative Reward Over Time (large, top)
  2. Episode Returns
  3. Step Count per Episode
  4. Win/Loss per Episode
  5. Success Rate

Usage:
    python utils/plotting/plot_three_agents_comparison.py <log_file.csv> [--max_episodes N]
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


def aggregate_episodes(df, agent_name, max_episodes=None):
    """Aggregate step data by episode for a specific agent."""
    agent_df = df[df['agent'] == agent_name].copy()
    
    episodes = agent_df.groupby('episode').agg({
        'reward': 'sum',      # Total reward per episode
        'step': 'max',        # Number of steps per episode
    }).reset_index()
    episodes.columns = ['episode', 'total_reward', 'steps']
    
    # Determine success/failure based on total reward
    # Win: reward = +2.0 (0.5 red + 1.5 blue)
    # Lose: reward = -0.5 (0.5 blue - 1.0 lose)
    # We can detect wins by checking if total_reward >= 1.5
    episodes['success'] = episodes['total_reward'] >= 1.5
    
    # Limit to max episodes if specified
    if max_episodes:
        episodes = episodes[episodes['episode'] <= max_episodes]
    
    return episodes


def main():
    parser = argparse.ArgumentParser(description='Plot comparison of three agents')
    parser.add_argument('log_file', help='Path to comparison CSV log file')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Maximum episode number to plot (default: all)')
    parser.add_argument('--output', default='three_agents_comparison.png',
                       help='Output filename (default: three_agents_comparison.png)')
    
    args = parser.parse_args()
    
    # Load the log file
    if not Path(args.log_file).exists():
        print(f"Error: Log file '{args.log_file}' not found!")
        sys.exit(1)
    
    print(f"Loading log file: {args.log_file}")
    log_df = pd.read_csv(args.log_file)
    
    print(f"Total rows: {len(log_df)}")
    print(f"Agents found: {log_df['agent'].unique()}")
    
    # Aggregate by agent and episode
    aif_episodes = aggregate_episodes(log_df, 'AIF', args.max_episodes)
    ql_episodes = aggregate_episodes(log_df, 'QL', args.max_episodes)
    dynaq_episodes = aggregate_episodes(log_df, 'DynaQ', args.max_episodes)
    
    # Debug: Check if data is loaded
    print(f"\nDebug - Episodes loaded:")
    print(f"  AIF: {len(aif_episodes)} episodes")
    print(f"  QL: {len(ql_episodes)} episodes")
    print(f"  DynaQ: {len(dynaq_episodes)} episodes")
    
    if len(aif_episodes) == 0:
        print("WARNING: No AIF episodes found!")
    if len(ql_episodes) == 0:
        print("WARNING: No QL episodes found!")
    if len(dynaq_episodes) == 0:
        print("WARNING: No DynaQ episodes found!")
    
    num_episodes = max(len(aif_episodes), len(ql_episodes), len(dynaq_episodes))
    
    print(f"\nActive Inference: {len(aif_episodes)} episodes")
    print(f"  Wins: {aif_episodes['success'].sum()}")
    print(f"  Losses: {(~aif_episodes['success']).sum()}")
    print(f"  Success rate: {aif_episodes['success'].mean()*100:.1f}%")
    
    print(f"\nQ-Learning: {len(ql_episodes)} episodes")
    print(f"  Wins: {ql_episodes['success'].sum()}")
    print(f"  Losses: {(~ql_episodes['success']).sum()}")
    print(f"  Success rate: {ql_episodes['success'].mean()*100:.1f}%")
    
    print(f"\nDyna-Q: {len(dynaq_episodes)} episodes")
    print(f"  Wins: {dynaq_episodes['success'].sum()}")
    print(f"  Losses: {(~dynaq_episodes['success']).sum()}")
    print(f"  Success rate: {dynaq_episodes['success'].mean()*100:.1f}%")
    
    # Create 5 plots (3 rows: 1 wide plot on top, then 2x2 grid)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
    title = f'AIF vs Q-Learning vs Dyna-Q ({num_episodes} Episodes)'
    if args.max_episodes:
        title += f' [Limited to first {args.max_episodes}]'
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Colors for three agents
    colors = {
        'AIF': '#2E86AB',      # Blue
        'QL': '#A23B72',       # Purple
        'DynaQ': '#06A77D'     # Green
    }
    
    # Plot 1: DEDICATED Cumulative Reward Over Time (Top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    aif_cumulative = aif_episodes['total_reward'].cumsum()
    ql_cumulative = ql_episodes['total_reward'].cumsum()
    dynaq_cumulative = dynaq_episodes['total_reward'].cumsum()
    
    ax1.plot(aif_episodes['episode'], aif_cumulative,
             linestyle='-', linewidth=3, marker='o', markersize=5,
             label='Active Inference', color=colors['AIF'], alpha=0.9, zorder=3)
    ax1.plot(ql_episodes['episode'], ql_cumulative,
             linestyle='-', linewidth=3, marker='s', markersize=5,
             label='Q-Learning', color=colors['QL'], alpha=0.9, zorder=3)
    ax1.plot(dynaq_episodes['episode'], dynaq_cumulative,
             linestyle='-', linewidth=3, marker='^', markersize=5,
             label='Dyna-Q', color=colors['DynaQ'], alpha=0.9, zorder=3)
    
    # Fill areas under curves
    ax1.fill_between(aif_episodes['episode'], aif_cumulative, alpha=0.15, color=colors['AIF'])
    ax1.fill_between(ql_episodes['episode'], ql_cumulative, alpha=0.15, color=colors['QL'])
    ax1.fill_between(dynaq_episodes['episode'], dynaq_cumulative, alpha=0.15, color=colors['DynaQ'])
    
    # Add annotations for final values
    final_aif = aif_cumulative.iloc[-1]
    final_ql = ql_cumulative.iloc[-1]
    final_dynaq = dynaq_cumulative.iloc[-1]
    
    max_ep = max(aif_episodes['episode'].max(), ql_episodes['episode'].max(), dynaq_episodes['episode'].max())
    ax1.text(max_ep, final_aif, f'{final_aif:.1f}', fontsize=11, fontweight='bold', 
             ha='left', va='bottom', color=colors['AIF'])
    ax1.text(max_ep, final_ql, f'{final_ql:.1f}', fontsize=11, fontweight='bold',
             ha='left', va='center', color=colors['QL'])
    ax1.text(max_ep, final_dynaq, f'{final_dynaq:.1f}', fontsize=11, fontweight='bold',
             ha='left', va='top', color=colors['DynaQ'])
    
    ax1.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Reward', fontsize=14, fontweight='bold')
    ax1.set_title('Cumulative Reward Over Time', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.8)
    ax1.set_xlim(0, max_ep + 1)
    
    # Plot 2: Episode Returns (Second row, left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(aif_episodes['episode'], aif_episodes['total_reward'], 
             marker='o', linestyle='-', linewidth=2, markersize=4, 
             label='Active Inference', color=colors['AIF'], alpha=0.8)
    ax2.plot(ql_episodes['episode'], ql_episodes['total_reward'], 
             marker='s', linestyle='-', linewidth=2, markersize=4,
             label='Q-Learning', color=colors['QL'], alpha=0.8)
    ax2.plot(dynaq_episodes['episode'], dynaq_episodes['total_reward'], 
             marker='^', linestyle='-', linewidth=2, markersize=4,
             label='Dyna-Q', color=colors['DynaQ'], alpha=0.8)
    
    ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.3, label='Win (+2.0)')
    ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3, label='Lose (-0.5)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Episode Returns', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_ep + 1)
    ax2.set_ylim(-0.8, 2.3)
    
    # Plot 3: Step Count per Episode (Second row, right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(aif_episodes['episode'], aif_episodes['steps'],
             marker='o', linestyle='-', linewidth=2, markersize=4,
             label='Active Inference', color=colors['AIF'], alpha=0.8)
    ax3.plot(ql_episodes['episode'], ql_episodes['steps'],
             marker='s', linestyle='-', linewidth=2, markersize=4,
             label='Q-Learning', color=colors['QL'], alpha=0.8)
    ax3.plot(dynaq_episodes['episode'], dynaq_episodes['steps'],
             marker='^', linestyle='-', linewidth=2, markersize=4,
             label='Dyna-Q', color=colors['DynaQ'], alpha=0.8)
    
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Max steps (timeout)')
    ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Steps to Completion', fontsize=12, fontweight='bold')
    ax3.set_title('Episode Length (Steps)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_ep + 1)
    
    # Plot 4: Win/Loss per Episode (Third row, left)
    ax4 = fig.add_subplot(gs[2, 0])
    episodes_range = np.arange(1, len(aif_episodes) + 1)
    bar_width = 0.25
    
    # Create win/loss bars
    aif_wins = aif_episodes['success'].astype(int)
    aif_losses = (~aif_episodes['success']).astype(int)
    ql_wins = ql_episodes['success'].astype(int)
    ql_losses = (~ql_episodes['success']).astype(int)
    dynaq_wins = dynaq_episodes['success'].astype(int)
    dynaq_losses = (~dynaq_episodes['success']).astype(int)
    
    # Stack bars for each agent
    ax4.bar(episodes_range - bar_width, aif_wins, bar_width, 
            label='AIF Win', color=colors['AIF'], alpha=0.8)
    ax4.bar(episodes_range - bar_width, -aif_losses, bar_width,
            label='AIF Loss', color=colors['AIF'], alpha=0.3)
    
    ax4.bar(episodes_range, ql_wins, bar_width,
            label='QL Win', color=colors['QL'], alpha=0.8)
    ax4.bar(episodes_range, -ql_losses, bar_width,
            label='QL Loss', color=colors['QL'], alpha=0.3)
    
    ax4.bar(episodes_range + bar_width, dynaq_wins, bar_width,
            label='DQ Win', color=colors['DynaQ'], alpha=0.8)
    ax4.bar(episodes_range + bar_width, -dynaq_losses, bar_width,
            label='DQ Loss', color=colors['DynaQ'], alpha=0.3)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Outcome (1=Win, -1=Loss)', fontsize=12, fontweight='bold')
    ax4.set_title('Win/Loss per Episode', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9, ncol=3)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(0, max_ep + 1)
    ax4.set_ylim(-1.5, 1.5)
    
    # Plot 5: Success Rate (Third row, right)
    ax5 = fig.add_subplot(gs[2, 1])
    window = 5  # 5-episode rolling window
    
    aif_cumsum = aif_episodes['success'].cumsum()
    aif_cumrate = 100 * aif_cumsum / np.arange(1, len(aif_episodes) + 1)
    
    ql_cumsum = ql_episodes['success'].cumsum()
    ql_cumrate = 100 * ql_cumsum / np.arange(1, len(ql_episodes) + 1)
    
    dynaq_cumsum = dynaq_episodes['success'].cumsum()
    dynaq_cumrate = 100 * dynaq_cumsum / np.arange(1, len(dynaq_episodes) + 1)
    
    ax5.plot(aif_episodes['episode'], aif_cumrate,
             linestyle='-', linewidth=2.5,
             label='AIF Cumulative', color=colors['AIF'], alpha=0.8)
    ax5.plot(ql_episodes['episode'], ql_cumrate,
             linestyle='-', linewidth=2.5,
             label='QL Cumulative', color=colors['QL'], alpha=0.8)
    ax5.plot(dynaq_episodes['episode'], dynaq_cumrate,
             linestyle='-', linewidth=2.5,
             label='DQ Cumulative', color=colors['DynaQ'], alpha=0.8)
    
    # Add rolling average
    aif_rolling = aif_episodes['success'].rolling(window=window, min_periods=1).mean() * 100
    ql_rolling = ql_episodes['success'].rolling(window=window, min_periods=1).mean() * 100
    dynaq_rolling = dynaq_episodes['success'].rolling(window=window, min_periods=1).mean() * 100
    
    ax5.plot(aif_episodes['episode'], aif_rolling,
             linestyle='--', linewidth=1.5, alpha=0.5,
             label=f'AIF {window}-ep avg', color=colors['AIF'])
    ax5.plot(ql_episodes['episode'], ql_rolling,
             linestyle='--', linewidth=1.5, alpha=0.5,
             label=f'QL {window}-ep avg', color=colors['QL'])
    ax5.plot(dynaq_episodes['episode'], dynaq_rolling,
             linestyle='--', linewidth=1.5, alpha=0.5,
             label=f'DQ {window}-ep avg', color=colors['DynaQ'])
    
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Success Rate', fontsize=14, fontweight='bold')
    ax5.legend(loc='lower right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, max_ep + 1)
    ax5.set_ylim(-5, 105)
    
    # Add vertical lines for config changes (every 20 episodes for 80-episode runs)
    # Detect config boundaries
    config_size = 20  # Default
    if num_episodes == 80:
        config_size = 20
    elif num_episodes == 200:
        config_size = 40
    
    all_axes = [ax1, ax2, ax3, ax4, ax5]
    config_lines = list(range(config_size, num_episodes, config_size))
    
    for ax in all_axes:
        for ep in config_lines:
            ax.axvline(x=ep, color='orange', linestyle=':', alpha=0.4, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {args.output}")
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    print("\nActive Inference:")
    print(f"  Total episodes: {len(aif_episodes)}")
    print(f"  Wins: {aif_episodes['success'].sum()} ({aif_episodes['success'].mean()*100:.1f}%)")
    print(f"  Losses: {(~aif_episodes['success']).sum()} ({(~aif_episodes['success']).mean()*100:.1f}%)")
    print(f"  Average reward: {aif_episodes['total_reward'].mean():.2f}")
    print(f"  Average steps: {aif_episodes['steps'].mean():.1f}")
    
    print("\nQ-Learning:")
    print(f"  Total episodes: {len(ql_episodes)}")
    print(f"  Wins: {ql_episodes['success'].sum()} ({ql_episodes['success'].mean()*100:.1f}%)")
    print(f"  Losses: {(~ql_episodes['success']).sum()} ({(~ql_episodes['success']).mean()*100:.1f}%)")
    print(f"  Average reward: {ql_episodes['total_reward'].mean():.2f}")
    print(f"  Average steps: {ql_episodes['steps'].mean():.1f}")
    
    print("\nDyna-Q:")
    print(f"  Total episodes: {len(dynaq_episodes)}")
    print(f"  Wins: {dynaq_episodes['success'].sum()} ({dynaq_episodes['success'].mean()*100:.1f}%)")
    print(f"  Losses: {(~dynaq_episodes['success']).sum()} ({(~dynaq_episodes['success']).mean()*100:.1f}%)")
    print(f"  Average reward: {dynaq_episodes['total_reward'].mean():.2f}")
    print(f"  Average steps: {dynaq_episodes['steps'].mean():.1f}")
    
    # Per-config statistics
    print(f"\nPer-Configuration Statistics (every {config_size} episodes):")
    num_configs = num_episodes // config_size
    
    for i in range(num_configs):
        start = i * config_size + 1
        end = (i + 1) * config_size
        
        aif_config = aif_episodes[(aif_episodes['episode'] >= start) & (aif_episodes['episode'] <= end)]
        ql_config = ql_episodes[(ql_episodes['episode'] >= start) & (ql_episodes['episode'] <= end)]
        dynaq_config = dynaq_episodes[(dynaq_episodes['episode'] >= start) & (dynaq_episodes['episode'] <= end)]
        
        print(f"\n  Config {i+1} (Episodes {start}-{end}):")
        print(f"    AIF:    {aif_config['success'].sum()}/{len(aif_config)} wins ({aif_config['success'].mean()*100:.1f}%)")
        print(f"    QL:     {ql_config['success'].sum()}/{len(ql_config)} wins ({ql_config['success'].mean()*100:.1f}%)")
        print(f"    Dyna-Q: {dynaq_config['success'].sum()}/{len(dynaq_config)} wins ({dynaq_config['success'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()

