"""
Compare PPO agents vs Random agents on Overcooked.

Runs both agent types for the same number of episodes, saves logs,
and generates comparison plots.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Import for PPO
try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.algorithm import Algorithm
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Error: Ray RLlib not available. Please install: pip install 'ray[rllib]'")
    sys.exit(1)

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


def run_episodes_with_agents(env, agents, num_episodes, agent_type, log_dir, verbose=True, args=None):
    """
    Run episodes with given agents and log results.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        agents: Dict mapping agent_id to agent (or None for random)
        num_episodes: Number of episodes to run
        agent_type: "ppo" or "random"
        log_dir: Directory to save logs
        verbose: Whether to print progress
    
    Returns:
        List of episode results (rewards, lengths, etc.)
    """
    episode_results = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {num_episodes} episodes with {agent_type.upper()} agents")
        print(f"{'='*80}")
    
    for episode in range(num_episodes):
        # Reset environment
        observations, infos = env.reset()
        
        episode_log = {
            "episode": episode,
            "agent_type": agent_type,
            "steps": [],
            "total_reward": 0.0,
            "episode_length": 0
        }
        
        step_count = 0
        total_reward = 0.0
        
        while step_count < env.horizon:
            # Get actions
            actions = {}
            if agents and agents.get("agent_0") is not None:
                algo = agents["agent_0"]
                
                # Use new API: get_module() and forward_inference()
                try:
                    for agent_id in ["agent_0", "agent_1"]:
                        try:
                            # Get the RLModule for this agent
                            module = algo.get_module(agent_id)
                            
                            # Get observation and convert to tensor
                            obs = observations[agent_id]
                            if isinstance(obs, (list, tuple)):
                                obs = np.array(obs)
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
                            
                            # Compute action using forward_inference
                            with torch.no_grad():
                                fwd_outputs = module.forward_inference({"obs": obs_tensor})
                                
                                # Try to extract action from output
                                if "action_dist_inputs" in fwd_outputs:
                                    # Get action distribution class and create distribution
                                    try:
                                        action_dist_class = module.get_inference_action_dist_cls()
                                        action_dist = action_dist_class.from_logits(
                                            fwd_outputs["action_dist_inputs"]
                                        )
                                        # Try deterministic_sample first, then sample
                                        if hasattr(action_dist, 'deterministic_sample'):
                                            action = action_dist.deterministic_sample()
                                        elif hasattr(action_dist, 'sample'):
                                            action = action_dist.sample()
                                        else:
                                            # Fallback: argmax on logits
                                            logits = fwd_outputs["action_dist_inputs"]
                                            action = torch.argmax(logits, dim=-1)
                                        
                                        # Extract scalar from tensor
                                        if isinstance(action, torch.Tensor):
                                            action = action[0].item() if action.dim() > 0 else action.item()
                                        actions[agent_id] = int(action)
                                    except Exception as e:
                                        # Fallback: use argmax on logits directly
                                        logits = fwd_outputs["action_dist_inputs"]
                                        action = torch.argmax(logits, dim=-1)
                                        if isinstance(action, torch.Tensor):
                                            action = action[0].item() if action.dim() > 0 else action.item()
                                        actions[agent_id] = int(action)
                                elif "action" in fwd_outputs:
                                    # Direct action output
                                    action = fwd_outputs["action"]
                                    if isinstance(action, torch.Tensor):
                                        action = action[0].item() if action.dim() > 0 else action.item()
                                    actions[agent_id] = int(action)
                                else:
                                    raise ValueError(f"No action or action_dist_inputs in output. Keys: {fwd_outputs.keys()}")
                        except Exception as e:
                            # If one agent fails, try the other, but log the error
                            if verbose and step_count == 0:
                                print(f"Warning: Could not get PPO action for {agent_id} ({type(e).__name__}: {e}). Using random.")
                            actions[agent_id] = env.action_space[agent_id].sample()
                except Exception as e:
                    # All methods failed - use random
                    if verbose and step_count == 0:
                        print(f"Warning: Could not get PPO actions ({type(e).__name__}: {e}). Using random.")
                    for agent_id in ["agent_0", "agent_1"]:
                        actions[agent_id] = env.action_space[agent_id].sample()
            else:
                # Random actions
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
        log_file = log_dir / f"{agent_type}_episode_{episode:04d}.json"
        with open(log_file, 'w') as f:
            json.dump(episode_log, f, indent=2)
        
        if verbose:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={total_reward:.4f}, Length={step_count}")
    
    return episode_results


def load_ppo_agents(checkpoint_path):
    """Load trained PPO agents from checkpoint."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Convert to absolute path if needed
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Load algorithm from checkpoint
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Store algorithm - we'll use it to compute actions
    # In new API stack, we need to use the algorithm's compute_actions method
    # or access modules via learner_group
    agents = {
        "agent_0": algo,
        "agent_1": algo
    }
    
    return agents, algo


def train_ppo_quick(layout="cramped_room", horizon=150, num_iterations=1000, seed=0):
    """
    Quick PPO training for comparison.
    
    Returns checkpoint path.
    """
    print(f"\n{'='*80}")
    print("Training PPO agents (quick training for comparison)")
    print(f"{'='*80}")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=3)
    
    # Environment config
    env_config = {
        "layout": layout,
        "horizon": horizon,
        "num_pots": 2
    }
    
    # Create environment instance
    env_instance = OvercookedMultiAgentEnv(layout=layout, horizon=horizon)
    
    # PPO config
    config = (
        PPOConfig()
        .environment(env=OvercookedMultiAgentEnv, env_config=env_config)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            train_batch_size=2000,  # Smaller for quick training
            minibatch_size=128,
            num_epochs=10,
        )
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
        )
        .multi_agent(
            policies={
                "agent_0": (
                    None,
                    env_instance.observation_space["agent_0"],
                    env_instance.action_space["agent_0"],
                    {}
                ),
                "agent_1": (
                    None,
                    env_instance.observation_space["agent_1"],
                    env_instance.action_space["agent_1"],
                    {}
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
        )
        .debugging(seed=seed)
    )
    
    # Checkpoint directory
    checkpoint_dir = f"checkpoints/ppo_quick_{layout}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build and train
    algo = config.build_algo()
    
    print(f"Training for {num_iterations} iterations...")
    for i in range(num_iterations):
        result = algo.train()
        if (i + 1) % 10 == 0:
            # Try different metric names (new API stack uses different names)
            reward = (result.get('env_runners/episode_return_mean') or 
                     result.get('episode_reward_mean') or
                     result.get('env_runners/episode_return_mean') or
                     None)
            
            episode_len = (result.get('env_runners/episode_len_mean') or
                          result.get('episode_len_mean') or
                          None)
            
            if reward is not None:
                print(f"  Iteration {i + 1}/{num_iterations} - Reward: {reward:.4f}", end="")
                if episode_len is not None:
                    print(f", Length: {episode_len:.1f}")
                else:
                    print()
            else:
                # Show what metrics are available
                print(f"  Iteration {i + 1}/{num_iterations} - Training...")
                if i == 0:  # Show available metrics only once
                    available_metrics = [k for k in result.keys() if any(x in k.lower() for x in ['reward', 'return', 'episode', 'len'])]
                    if available_metrics:
                        print(f"    Available metrics: {', '.join(available_metrics[:5])}")
    
    # Save checkpoint
    checkpoint_result = algo.save(checkpoint_dir)
    # Extract path from TrainingResult object (new API returns TrainingResult, not string)
    try:
        if hasattr(checkpoint_result, 'checkpoint'):
            # TrainingResult has a checkpoint attribute (Checkpoint object)
            checkpoint_obj = checkpoint_result.checkpoint
            if hasattr(checkpoint_obj, 'path'):
                checkpoint_path = checkpoint_obj.path
            elif hasattr(checkpoint_obj, 'filesystem'):
                # Checkpoint object - convert to string path
                checkpoint_path = str(checkpoint_obj)
            else:
                checkpoint_path = str(checkpoint_obj)
        elif isinstance(checkpoint_result, str):
            # Old API might return string directly
            checkpoint_path = checkpoint_result
        else:
            # Fallback: use the checkpoint_dir we created
            checkpoint_path = checkpoint_dir
    except Exception as e:
        # If anything fails, use checkpoint_dir
        print(f"Warning: Could not extract checkpoint path from result: {e}")
        checkpoint_path = checkpoint_dir
    
    # Ensure it's a string and absolute path
    checkpoint_path = str(checkpoint_path) if checkpoint_path else checkpoint_dir
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    
    print(f"Training complete. Checkpoint: {checkpoint_path}")
    
    algo.stop()
    
    return checkpoint_path


def plot_comparison(ppo_results, random_results, output_dir, layout, horizon):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    
    # Extract data
    ppo_rewards = [r["total_reward"] for r in ppo_results]
    ppo_lengths = [r["episode_length"] for r in ppo_results]
    random_rewards = [r["total_reward"] for r in random_results]
    random_lengths = [r["episode_length"] for r in random_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PPO vs Random Agents Comparison\nLayout: {layout}, Horizon: {horizon}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Episode Rewards Over Time
    ax1 = axes[0, 0]
    episodes = range(1, len(ppo_rewards) + 1)
    ax1.plot(episodes, ppo_rewards, 'b-o', label='PPO', linewidth=2, markersize=6)
    ax1.plot(episodes, random_rewards, 'r-s', label='Random', linewidth=2, markersize=6)
    ax1.axhline(y=np.mean(ppo_rewards), color='b', linestyle='--', alpha=0.5, label=f'PPO Mean: {np.mean(ppo_rewards):.4f}')
    ax1.axhline(y=np.mean(random_rewards), color='r', linestyle='--', alpha=0.5, label=f'Random Mean: {np.mean(random_rewards):.4f}')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=11)
    ax1.set_title('Episode Rewards Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Lengths Over Time
    ax2 = axes[0, 1]
    ax2.plot(episodes, ppo_lengths, 'b-o', label='PPO', linewidth=2, markersize=6)
    ax2.plot(episodes, random_lengths, 'r-s', label='Random', linewidth=2, markersize=6)
    ax2.axhline(y=np.mean(ppo_lengths), color='b', linestyle='--', alpha=0.5, label=f'PPO Mean: {np.mean(ppo_lengths):.1f}')
    ax2.axhline(y=np.mean(random_lengths), color='r', linestyle='--', alpha=0.5, label=f'Random Mean: {np.mean(random_lengths):.1f}')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Episode Length (steps)', fontsize=11)
    ax2.set_title('Episode Lengths Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward Distribution (Box Plot)
    ax3 = axes[1, 0]
    data = [ppo_rewards, random_rewards]
    bp = ax3.boxplot(data, labels=['PPO', 'Random'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Total Reward', fontsize=11)
    ax3.set_title('Reward Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    ax3.plot([1], [np.mean(ppo_rewards)], 'bD', markersize=10, label='PPO Mean')
    ax3.plot([2], [np.mean(random_rewards)], 'rD', markersize=10, label='Random Mean')
    ax3.legend()
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats_text = f"""
    SUMMARY STATISTICS
    
    PPO Agents:
    • Mean Reward: {np.mean(ppo_rewards):.4f}
    • Std Reward:  {np.std(ppo_rewards):.4f}
    • Min Reward:  {np.min(ppo_rewards):.4f}
    • Max Reward:  {np.max(ppo_rewards):.4f}
    • Mean Length: {np.mean(ppo_lengths):.1f}
    • Std Length:  {np.std(ppo_lengths):.1f}
    
    Random Agents:
    • Mean Reward: {np.mean(random_rewards):.4f}
    • Std Reward:  {np.std(random_rewards):.4f}
    • Min Reward:  {np.min(random_rewards):.4f}
    • Max Reward:  {np.max(random_rewards):.4f}
    • Mean Length: {np.mean(random_lengths):.1f}
    • Std Length:  {np.std(random_lengths):.1f}
    
    Improvement:
    • Reward: {((np.mean(ppo_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'comparison_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    # Also save as PDF
    plot_file_pdf = output_dir / 'comparison_plot.pdf'
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare PPO vs Random agents on Overcooked")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--horizon", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes per agent type")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to PPO checkpoint (if None, trains new agents)")
    parser.add_argument("--train-iterations", type=int, default=1000,
                       help="Number of training iterations if training new PPO agents")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for logs and plots")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"comparison_results/ppo_vs_random_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PPO vs RANDOM AGENTS COMPARISON")
    print("="*80)
    print(f"Layout: {args.layout}")
    print(f"Horizon: {args.horizon}")
    print(f"Episodes per agent type: {args.num_episodes}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Create environment
    env = OvercookedMultiAgentEnv(layout=args.layout, horizon=args.horizon)
    
    # Load or train PPO agents
    if args.checkpoint:
        print(f"\nLoading PPO agents from: {args.checkpoint}")
        ppo_agents, ppo_algo = load_ppo_agents(args.checkpoint)
    else:
        print(f"\nTraining new PPO agents ({args.train_iterations} iterations)...")
        checkpoint_path = train_ppo_quick(
            layout=args.layout,
            horizon=args.horizon,
            num_iterations=args.train_iterations,
            seed=args.seed
        )
        ppo_agents, ppo_algo = load_ppo_agents(checkpoint_path)
    
    # Run PPO episodes
    print(f"\n{'='*80}")
    print("Running PPO agents...")
    ppo_results = run_episodes_with_agents(
        env, ppo_agents, args.num_episodes, "ppo", output_dir, verbose=args.verbose, args=args
    )
    
    # Run Random episodes
    print(f"\n{'='*80}")
    print("Running Random agents...")
    random_results = run_episodes_with_agents(
        env, None, args.num_episodes, "random", output_dir, verbose=args.verbose, args=args
    )
    
    # Create plots
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    plot_comparison(ppo_results, random_results, output_dir, args.layout, args.horizon)
    
    # Extract data for summary
    ppo_rewards = [r["total_reward"] for r in ppo_results]
    ppo_lengths = [r["episode_length"] for r in ppo_results]
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
        "ppo": {
            "rewards": ppo_rewards,
            "lengths": ppo_lengths,
            "mean_reward": float(np.mean(ppo_rewards)),
            "std_reward": float(np.std(ppo_rewards)),
            "mean_length": float(np.mean(ppo_lengths)),
            "std_length": float(np.std(ppo_lengths))
        },
        "random": {
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
    print(f"\nPPO Agents:")
    print(f"  Mean reward: {summary['ppo']['mean_reward']:.4f} ± {summary['ppo']['std_reward']:.4f}")
    print(f"  Mean length: {summary['ppo']['mean_length']:.1f} ± {summary['ppo']['std_length']:.1f}")
    print(f"\nRandom Agents:")
    print(f"  Mean reward: {summary['random']['mean_reward']:.4f} ± {summary['random']['std_reward']:.4f}")
    print(f"  Mean length: {summary['random']['mean_length']:.1f} ± {summary['random']['std_length']:.1f}")
    if abs(summary['random']['mean_reward']) > 1e-6:
        improvement = ((summary['ppo']['mean_reward'] - summary['random']['mean_reward']) / abs(summary['random']['mean_reward']) * 100)
        print(f"\nImprovement: {improvement:.1f}%")
    else:
        diff = summary['ppo']['mean_reward'] - summary['random']['mean_reward']
        print(f"\nDifference: {diff:.4f} (both near zero)")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Plots: {output_dir / 'comparison_plot.png'}")
    print(f"{'='*80}\n")
    
    # Cleanup
    env.close()
    if 'ppo_algo' in locals():
        ppo_algo.stop()
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
