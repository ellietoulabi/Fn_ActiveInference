"""
Plot detailed comparison of all three agents for a single episode.
Shows step-by-step actions, rewards, and map states.

Usage:
    python plot_single_episode.py <log_file.csv> <episode_number>
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys


def parse_map(map_str):
    """Parse map string to extract agent and button positions."""
    rows = map_str.split('|')
    agent_pos = None
    red_pos = None
    blue_pos = None
    red_pressed = False
    blue_pressed = False
    
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell == 'A':
                agent_pos = (x, y)
            elif cell == 'r':
                red_pos = (x, y)
            elif cell == 'R':
                red_pos = (x, y)
                red_pressed = True
            elif cell == 'b':
                blue_pos = (x, y)
            elif cell == 'B':
                blue_pos = (x, y)
                blue_pressed = True
    
    return {
        'agent': agent_pos,
        'red': red_pos,
        'blue': blue_pos,
        'red_pressed': red_pressed,
        'blue_pressed': blue_pressed
    }


def main():
    parser = argparse.ArgumentParser(description='Plot single episode comparison')
    parser.add_argument('log_file', help='Path to comparison CSV log file')
    parser.add_argument('episode', type=int, help='Episode number to plot')
    parser.add_argument('--output', default=None,
                       help='Output filename (default: episode_N_comparison.png)')
    
    args = parser.parse_args()
    
    # Load the log file
    if not Path(args.log_file).exists():
        print(f"Error: Log file '{args.log_file}' not found!")
        sys.exit(1)
    
    print(f"Loading log file: {args.log_file}")
    log_df = pd.read_csv(args.log_file)
    
    # Filter to specific episode
    episode_df = log_df[log_df['episode'] == args.episode].copy()
    
    if len(episode_df) == 0:
        print(f"Error: Episode {args.episode} not found in log file!")
        print(f"Available episodes: {sorted(log_df['episode'].unique())}")
        sys.exit(1)
    
    print(f"\nAnalyzing Episode {args.episode}")
    print(f"Total steps in episode: {len(episode_df)}")
    
    # Separate by agent
    aif_df = episode_df[episode_df['agent'] == 'AIF'].copy()
    ql_df = episode_df[episode_df['agent'] == 'QL'].copy()
    dynaq_df = episode_df[episode_df['agent'] == 'DynaQ'].copy()
    
    print(f"  AIF: {len(aif_df)} steps")
    print(f"  QL: {len(ql_df)} steps")
    print(f"  DynaQ: {len(dynaq_df)} steps")
    
    # Calculate cumulative rewards
    aif_df['cumulative_reward'] = aif_df['reward'].cumsum()
    ql_df['cumulative_reward'] = ql_df['reward'].cumsum()
    dynaq_df['cumulative_reward'] = dynaq_df['reward'].cumsum()
    
    # Final rewards
    aif_total = aif_df['reward'].sum()
    ql_total = ql_df['reward'].sum()
    dynaq_total = dynaq_df['reward'].sum()
    
    print(f"\nFinal rewards:")
    print(f"  AIF: {aif_total:+.2f} ({'WIN' if aif_total >= 1.5 else 'FAIL'})")
    print(f"  QL: {ql_total:+.2f} ({'WIN' if ql_total >= 1.5 else 'FAIL'})")
    print(f"  DynaQ: {dynaq_total:+.2f} ({'WIN' if dynaq_total >= 1.5 else 'FAIL'})")
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Episode {args.episode} - Detailed Comparison', fontsize=16, fontweight='bold')
    
    colors = {
        'AIF': '#2E86AB',      # Blue
        'QL': '#A23B72',       # Purple
        'DynaQ': '#06A77D'     # Green
    }
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    
    # Plot 1: Cumulative Reward over Steps
    ax1 = axes[0, 0]
    ax1.plot(aif_df['step'], aif_df['cumulative_reward'], 
             marker='o', linewidth=2.5, markersize=6,
             label=f'AIF (total: {aif_total:+.2f})', color=colors['AIF'])
    ax1.plot(ql_df['step'], ql_df['cumulative_reward'], 
             marker='s', linewidth=2.5, markersize=6,
             label=f'QL (total: {ql_total:+.2f})', color=colors['QL'])
    ax1.plot(dynaq_df['step'], dynaq_df['cumulative_reward'], 
             marker='^', linewidth=2.5, markersize=6,
             label=f'DynaQ (total: {dynaq_total:+.2f})', color=colors['DynaQ'])
    
    ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.3, label='Win threshold')
    ax1.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Cumulative Reward Over Steps', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward per Step
    ax2 = axes[0, 1]
    width = 0.25
    
    # Get max step to set x-axis range
    max_step = max(aif_df['step'].max(), ql_df['step'].max(), dynaq_df['step'].max())
    
    # Plot each agent with their own step numbers
    ax2.bar(aif_df['step'] - width, aif_df['reward'], width, 
            label=f'AIF ({len(aif_df)} steps)', color=colors['AIF'], alpha=0.8)
    ax2.bar(ql_df['step'], ql_df['reward'], width, 
            label=f'QL ({len(ql_df)} steps)', color=colors['QL'], alpha=0.8)
    ax2.bar(dynaq_df['step'] + width, dynaq_df['reward'], width, 
            label=f'DynaQ ({len(dynaq_df)} steps)', color=colors['DynaQ'], alpha=0.8)
    
    ax2.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Reward per Step', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max_step + 1)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 3: Action Distribution
    ax3 = axes[1, 0]
    
    # Count actions for each agent
    aif_actions = [action_names[a] for a in aif_df['action']]
    ql_actions = [action_names[a] for a in ql_df['action']]
    dynaq_actions = [action_names[a] for a in dynaq_df['action']]
    
    action_types = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PRESS', 'NOOP']
    aif_counts = [aif_actions.count(a) for a in action_types]
    ql_counts = [ql_actions.count(a) for a in action_types]
    dynaq_counts = [dynaq_actions.count(a) for a in action_types]
    
    x = np.arange(len(action_types))
    width = 0.25
    
    ax3.bar(x - width, aif_counts, width, label='AIF', color=colors['AIF'], alpha=0.8)
    ax3.bar(x, ql_counts, width, label='QL', color=colors['QL'], alpha=0.8)
    ax3.bar(x + width, dynaq_counts, width, label='DynaQ', color=colors['DynaQ'], alpha=0.8)
    
    ax3.set_xlabel('Action Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Action Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(action_types, rotation=45)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Agent Trajectories on Grid
    ax4 = axes[1, 1]
    
    # Parse first map to get grid layout
    first_map = aif_df.iloc[0]['map']
    map_info = parse_map(first_map)
    
    # Extract trajectories
    aif_trajectory = []
    ql_trajectory = []
    dynaq_trajectory = []
    
    for idx, row in aif_df.iterrows():
        map_info = parse_map(row['map'])
        if map_info['agent']:
            aif_trajectory.append(map_info['agent'])
    
    for idx, row in ql_df.iterrows():
        map_info = parse_map(row['map'])
        if map_info['agent']:
            ql_trajectory.append(map_info['agent'])
    
    for idx, row in dynaq_df.iterrows():
        map_info = parse_map(row['map'])
        if map_info['agent']:
            dynaq_trajectory.append(map_info['agent'])
    
    # Get button positions from final state
    final_map = aif_df.iloc[-1]['map']
    map_info = parse_map(final_map)
    
    # Plot grid
    ax4.set_xlim(-0.5, 2.5)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_aspect('equal')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.set_xlabel('X', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax4.set_title('Agent Trajectories (3×3 Grid)', fontsize=14, fontweight='bold')
    
    # Plot button positions
    if map_info['red']:
        marker = 'R' if map_info['red_pressed'] else 'r'
        ax4.plot(map_info['red'][0], map_info['red'][1], 'rs', markersize=20, 
                label=f'Red Button ({marker})', alpha=0.5)
    if map_info['blue']:
        marker = 'B' if map_info['blue_pressed'] else 'b'
        ax4.plot(map_info['blue'][0], map_info['blue'][1], 'bs', markersize=20,
                label=f'Blue Button ({marker})', alpha=0.5)
    
    # Plot trajectories
    if aif_trajectory:
        aif_x = [p[0] for p in aif_trajectory]
        aif_y = [p[1] for p in aif_trajectory]
        ax4.plot(aif_x, aif_y, 'o-', color=colors['AIF'], linewidth=2, 
                markersize=4, label='AIF', alpha=0.7)
        ax4.plot(aif_x[0], aif_y[0], 'o', color=colors['AIF'], markersize=10, alpha=1.0)  # Start
    
    if ql_trajectory:
        ql_x = [p[0] for p in ql_trajectory]
        ql_y = [p[1] for p in ql_trajectory]
        ax4.plot(ql_x, ql_y, 's-', color=colors['QL'], linewidth=2,
                markersize=4, label='QL', alpha=0.7)
        ax4.plot(ql_x[0], ql_y[0], 's', color=colors['QL'], markersize=10, alpha=1.0)  # Start
    
    if dynaq_trajectory:
        dynaq_x = [p[0] for p in dynaq_trajectory]
        dynaq_y = [p[1] for p in dynaq_trajectory]
        ax4.plot(dynaq_x, dynaq_y, '^-', color=colors['DynaQ'], linewidth=2,
                markersize=4, label='DynaQ', alpha=0.7)
        ax4.plot(dynaq_x[0], dynaq_y[0], '^', color=colors['DynaQ'], markersize=10, alpha=1.0)  # Start
    
    ax4.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    if args.output:
        output_path = args.output
    else:
        output_path = f'episode_{args.episode}_comparison.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()

