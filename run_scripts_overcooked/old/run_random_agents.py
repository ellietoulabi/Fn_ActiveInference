"""
Run Random agents on Overcooked.

Runs random agents for a specified number of episodes, saves logs,
and generates statistics.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import environment
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
from overcooked_ai_py.mdp.actions import Action


def action_to_string(action_idx):
    """Convert action index to human-readable string."""
    action_names = {
        0: "NORTH",
        1: "SOUTH", 
        2: "EAST",
        3: "WEST",
        4: "STAY",
        5: "INTERACT"
    }
    return action_names.get(action_idx, f"UNKNOWN({action_idx})")


def run_episodes_random(env, num_episodes, log_dir, verbose=True):
    """
    Run episodes with random agents and log results.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        num_episodes: Number of episodes to run
        log_dir: Directory to save logs
        verbose: Whether to print progress
    
    Returns:
        List of episode results (rewards, lengths, etc.)
    """
    episode_results = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {num_episodes} episodes with RANDOM agents")
        print(f"{'='*80}")
    
    for episode in range(num_episodes):
        # Reset environment
        observations, infos = env.reset()
        
        episode_log = {
            "episode": episode,
            "agent_type": "random",
            "steps": [],
            "total_reward": 0.0,
            "episode_length": 0
        }
        
        step_count = 0
        total_reward = 0.0
        
        while step_count < env.horizon:
            # Random actions
            actions = {}
            for agent_id in ["agent_0", "agent_1"]:
                actions[agent_id] = env.action_space[agent_id].sample()
            
            # Step environment
            observations_next, rewards, terminated, truncated, infos = env.step(actions)
            
            # Log step
            step_log = {
                "step": step_count,
                "actions": {
                    "agent_0": {
                        "action_idx": int(actions["agent_0"]),
                        "action_name": action_to_string(actions["agent_0"])
                    },
                    "agent_1": {
                        "action_idx": int(actions["agent_1"]),
                        "action_name": action_to_string(actions["agent_1"])
                    }
                },
                "rewards": {
                    "agent_0": float(rewards["agent_0"]),
                    "agent_1": float(rewards["agent_1"]),
                    "total": float(rewards["agent_0"])  # Shared reward
                }
            }
            
            episode_log["steps"].append(step_log)
            total_reward += rewards["agent_0"]
            step_count += 1
            observations = observations_next
            
            # Check if done
            if terminated["__all__"] or truncated["__all__"]:
                break
        
        episode_log["total_reward"] = float(total_reward)
        episode_log["episode_length"] = step_count
        
        episode_results.append({
            "episode": episode,
            "total_reward": total_reward,
            "episode_length": step_count
        })
        
        # Save episode log
        log_file = log_dir / f"random_episode_{episode:04d}.json"
        with open(log_file, 'w') as f:
            json.dump(episode_log, f, indent=2)
        
        if verbose:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={total_reward:.4f}, Length={step_count}")
    
    return episode_results


def plot_results(random_results, output_dir, layout, horizon):
    """Create plots for random agents."""
    output_dir = Path(output_dir)
    
    # Extract data
    random_rewards = [r["total_reward"] for r in random_results]
    random_lengths = [r["episode_length"] for r in random_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Random Agents Results\nLayout: {layout}, Horizon: {horizon}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Episode Rewards Over Time
    ax1 = axes[0, 0]
    episodes = range(1, len(random_rewards) + 1)
    ax1.plot(episodes, random_rewards, 'r-s', label='Random', linewidth=2, markersize=6)
    ax1.axhline(y=np.mean(random_rewards), color='r', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(random_rewards):.4f}')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=11)
    ax1.set_title('Episode Rewards Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Lengths Over Time
    ax2 = axes[0, 1]
    ax2.plot(episodes, random_lengths, 'r-s', label='Random', linewidth=2, markersize=6)
    ax2.axhline(y=np.mean(random_lengths), color='r', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(random_lengths):.1f}')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Episode Length (steps)', fontsize=11)
    ax2.set_title('Episode Lengths Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward Distribution (Histogram)
    ax3 = axes[1, 0]
    ax3.hist(random_rewards, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(random_rewards), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(random_rewards):.4f}')
    ax3.set_xlabel('Total Reward', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats_text = f"""
    SUMMARY STATISTICS
    
    Random Agents:
    • Mean Reward: {np.mean(random_rewards):.4f}
    • Std Reward:  {np.std(random_rewards):.4f}
    • Min Reward:  {np.min(random_rewards):.4f}
    • Max Reward:  {np.max(random_rewards):.4f}
    • Mean Length: {np.mean(random_lengths):.1f}
    • Std Length:  {np.std(random_lengths):.1f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'random_agents_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    # Also save as PDF
    plot_file_pdf = output_dir / 'random_agents_plot.pdf'
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Random agents on Overcooked")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--horizon", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for logs and plots")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/random_agents_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RANDOM AGENTS")
    print("="*80)
    print(f"Layout: {args.layout}")
    print(f"Horizon: {args.horizon}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Create environment
    env = OvercookedMultiAgentEnv(layout=args.layout, horizon=args.horizon)
    
    # Run Random episodes
    random_results = run_episodes_random(
        env, args.num_episodes, output_dir, verbose=args.verbose
    )
    
    # Create plots
    print(f"\n{'='*80}")
    print("Generating plots...")
    plot_results(random_results, output_dir, args.layout, args.horizon)
    
    # Extract data for summary
    random_rewards = [r["total_reward"] for r in random_results]
    random_lengths = [r["episode_length"] for r in random_results]
    
    # Save summary
    summary = {
        "config": {
            "layout": args.layout,
            "horizon": args.horizon,
            "num_episodes": args.num_episodes,
            "seed": args.seed
        },
        "results": {
            "rewards": random_rewards,
            "lengths": random_lengths,
            "mean_reward": float(np.mean(random_rewards)),
            "std_reward": float(np.std(random_rewards)),
            "mean_length": float(np.mean(random_lengths)),
            "std_length": float(np.std(random_lengths))
        }
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Mean reward: {summary['results']['mean_reward']:.4f} ± {summary['results']['std_reward']:.4f}")
    print(f"Mean length: {summary['results']['mean_length']:.1f} ± {summary['results']['std_length']:.1f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Plots: {output_dir / 'random_agents_plot.png'}")
    print(f"{'='*80}\n")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
