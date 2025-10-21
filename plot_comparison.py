"""
Compare Active Inference vs Q-Learning agents from log files.
Creates 5 comparison plots:
  1. Cumulative Reward Over Time (large, top)
  2. Episode Returns
  3. Step Count per Episode
  4. Win/Loss per Episode
  5. Success Rate
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the log files
aif_log = pd.read_csv('active_inference_log_ep50_step50_20251021_133609.csv')
ql_log = pd.read_csv('qlearning_log_ep50_step50_20251021_135747.csv')

print(f"Active Inference log: {len(aif_log)} rows")
print(f"Q-Learning log: {len(ql_log)} rows")

# Aggregate by episode
def aggregate_episodes(df):
    """Aggregate step data by episode."""
    episodes = df.groupby('episode').agg({
        'reward': 'sum',      # Total reward per episode
        'step': 'max',        # Number of steps per episode
    }).reset_index()
    episodes.columns = ['episode', 'total_reward', 'steps']
    
    # Determine success/failure based on total reward
    # Win: reward = +2.0 (0.5 red + 1.5 blue)
    # Lose: reward = -0.5 (0.5 blue - 1.0 lose)
    # We can detect wins by checking if total_reward >= 1.5
    episodes['success'] = episodes['total_reward'] >= 1.5
    
    return episodes

aif_episodes = aggregate_episodes(aif_log)
ql_episodes = aggregate_episodes(ql_log)

print(f"\nActive Inference: {len(aif_episodes)} episodes")
print(f"  Wins: {aif_episodes['success'].sum()}")
print(f"  Losses: {(~aif_episodes['success']).sum()}")
print(f"  Success rate: {aif_episodes['success'].mean()*100:.1f}%")

print(f"\nQ-Learning: {len(ql_episodes)} episodes")
print(f"  Wins: {ql_episodes['success'].sum()}")
print(f"  Losses: {(~ql_episodes['success']).sum()}")
print(f"  Success rate: {ql_episodes['success'].mean()*100:.1f}%")

# Create 5 plots (3 rows: 1 wide plot on top, then 2x2 grid)
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
fig.suptitle('Active Inference vs Q-Learning Comparison (50 Episodes)', fontsize=18, fontweight='bold', y=0.98)

# Plot 1: DEDICATED Cumulative Reward Over Time (Top row, full width)
ax1 = fig.add_subplot(gs[0, :])
aif_cumulative = aif_episodes['total_reward'].cumsum()
ql_cumulative = ql_episodes['total_reward'].cumsum()

ax1.plot(aif_episodes['episode'], aif_cumulative,
         linestyle='-', linewidth=3, marker='o', markersize=5,
         label='Active Inference', color='#2E86AB', alpha=0.9, zorder=3)
ax1.plot(ql_episodes['episode'], ql_cumulative,
         linestyle='-', linewidth=3, marker='s', markersize=5,
         label='Q-Learning', color='#A23B72', alpha=0.9, zorder=3)

# Fill areas under curves
ax1.fill_between(aif_episodes['episode'], aif_cumulative, alpha=0.15, color='#2E86AB')
ax1.fill_between(ql_episodes['episode'], ql_cumulative, alpha=0.15, color='#A23B72')

# Add annotations for final values
final_aif = aif_cumulative.iloc[-1]
final_ql = ql_cumulative.iloc[-1]
ax1.text(50, final_aif, f'{final_aif:.1f}', fontsize=12, fontweight='bold', 
         ha='left', va='bottom', color='#2E86AB')
ax1.text(50, final_ql, f'{final_ql:.1f}', fontsize=12, fontweight='bold',
         ha='left', va='top', color='#A23B72')

ax1.set_xlabel('Episode', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Reward', fontsize=14, fontweight='bold')
ax1.set_title('Cumulative Reward Over Time', fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax1.grid(True, alpha=0.3, linewidth=0.8)
ax1.set_xlim(0, 51)
ax1.set_ylim(min(0, ql_cumulative.min() - 5), max(aif_cumulative.max(), ql_cumulative.max()) * 1.1)

# Plot 2: Episode Returns (Second row, left)
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(aif_episodes['episode'], aif_episodes['total_reward'], 
         marker='o', linestyle='-', linewidth=2, markersize=4, 
         label='Active Inference', color='#2E86AB', alpha=0.8)
ax2.plot(ql_episodes['episode'], ql_episodes['total_reward'], 
         marker='s', linestyle='-', linewidth=2, markersize=4,
         label='Q-Learning', color='#A23B72', alpha=0.8)
ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.3, label='Win (+2.0)')
ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3, label='Lose (-0.5)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
ax2.set_title('Episode Returns', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 51)
ax2.set_ylim(-0.8, 2.3)

# Plot 3: Step Count per Episode (Second row, right)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(aif_episodes['episode'], aif_episodes['steps'],
         marker='o', linestyle='-', linewidth=2, markersize=4,
         label='Active Inference', color='#2E86AB', alpha=0.8)
ax3.plot(ql_episodes['episode'], ql_episodes['steps'],
         marker='s', linestyle='-', linewidth=2, markersize=4,
         label='Q-Learning', color='#A23B72', alpha=0.8)
ax3.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Max steps (timeout)')
ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax3.set_ylabel('Steps to Completion', fontsize=12, fontweight='bold')
ax3.set_title('Episode Length (Steps)', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 51)

# Plot 4: Win/Loss per Episode (Third row, left)
ax4 = fig.add_subplot(gs[2, 0])
episodes_range = np.arange(1, len(aif_episodes) + 1)
bar_width = 0.35

# Create win/loss bars
aif_wins = aif_episodes['success'].astype(int)
aif_losses = (~aif_episodes['success']).astype(int)
ql_wins = ql_episodes['success'].astype(int)
ql_losses = (~ql_episodes['success']).astype(int)

# Stack bars for each agent
ax4.bar(episodes_range - bar_width/2, aif_wins, bar_width, 
        label='AIF Win', color='#2E86AB', alpha=0.8)
ax4.bar(episodes_range - bar_width/2, -aif_losses, bar_width,
        label='AIF Loss', color='#2E86AB', alpha=0.3)

ax4.bar(episodes_range + bar_width/2, ql_wins, bar_width,
        label='QL Win', color='#A23B72', alpha=0.8)
ax4.bar(episodes_range + bar_width/2, -ql_losses, bar_width,
        label='QL Loss', color='#A23B72', alpha=0.3)

ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax4.set_ylabel('Outcome (1=Win, -1=Loss)', fontsize=12, fontweight='bold')
ax4.set_title('Win/Loss per Episode', fontsize=14, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10, ncol=2)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xlim(0, 51)
ax4.set_ylim(-1.5, 1.5)

# Plot 5: Success Rate (Third row, right)
ax5 = fig.add_subplot(gs[2, 1])
window = 5  # 5-episode rolling window

aif_cumsum = aif_episodes['success'].cumsum()
aif_cumrate = 100 * aif_cumsum / np.arange(1, len(aif_episodes) + 1)

ql_cumsum = ql_episodes['success'].cumsum()
ql_cumrate = 100 * ql_cumsum / np.arange(1, len(ql_episodes) + 1)

ax5.plot(aif_episodes['episode'], aif_cumrate,
         linestyle='-', linewidth=2.5,
         label='AIF Cumulative', color='#2E86AB', alpha=0.8)
ax5.plot(ql_episodes['episode'], ql_cumrate,
         linestyle='-', linewidth=2.5,
         label='QL Cumulative', color='#A23B72', alpha=0.8)

# Add rolling average
aif_rolling = aif_episodes['success'].rolling(window=window, min_periods=1).mean() * 100
ql_rolling = ql_episodes['success'].rolling(window=window, min_periods=1).mean() * 100

ax5.plot(aif_episodes['episode'], aif_rolling,
         linestyle='--', linewidth=1.5, alpha=0.5,
         label=f'AIF {window}-ep avg', color='#2E86AB')
ax5.plot(ql_episodes['episode'], ql_rolling,
         linestyle='--', linewidth=1.5, alpha=0.5,
         label=f'QL {window}-ep avg', color='#A23B72')

ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax5.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax5.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax5.set_title('Success Rate', fontsize=14, fontweight='bold')
ax5.legend(loc='lower right', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 51)
ax5.set_ylim(-5, 105)

# Add vertical lines every 10 episodes (config changes)
all_axes = [ax1, ax2, ax3, ax4, ax5]
for ax in all_axes:
    for ep in [10, 20, 30, 40]:
        ax.axvline(x=ep, color='orange', linestyle=':', alpha=0.4, linewidth=1.5)
    # Add text annotation
    ax.text(5, ax.get_ylim()[1]*0.95, 'Config 1', fontsize=8, ha='center', alpha=0.6)
    ax.text(15, ax.get_ylim()[1]*0.95, 'Config 2', fontsize=8, ha='center', alpha=0.6)
    ax.text(25, ax.get_ylim()[1]*0.95, 'Config 3', fontsize=8, ha='center', alpha=0.6)
    ax.text(35, ax.get_ylim()[1]*0.95, 'Config 4', fontsize=8, ha='center', alpha=0.6)
    ax.text(45, ax.get_ylim()[1]*0.95, 'Config 5', fontsize=8, ha='center', alpha=0.6)

plt.tight_layout()
plt.savefig('agent_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: agent_comparison.png")
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

# Per-config statistics
print("\nPer-Configuration Statistics:")
configs = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
for i, (start, end) in enumerate(configs, 1):
    aif_config = aif_episodes[(aif_episodes['episode'] >= start) & (aif_episodes['episode'] <= end)]
    ql_config = ql_episodes[(ql_episodes['episode'] >= start) & (ql_episodes['episode'] <= end)]
    
    print(f"\n  Config {i} (Episodes {start}-{end}):")
    print(f"    AIF: {aif_config['success'].sum()}/{len(aif_config)} wins ({aif_config['success'].mean()*100:.1f}%)")
    print(f"    QL:  {ql_config['success'].sum()}/{len(ql_config)} wins ({ql_config['success'].mean()*100:.1f}%)")

