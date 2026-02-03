"""
Fair comparison script between PPO and Active Inference on Overcooked.

This script ensures a fair comparison by:
1. Using the same environment configuration
2. Matching total training steps
3. Using the same evaluation protocol
4. Running multiple seeds for statistical significance
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import json
from datetime import datetime
import os

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
    print("Warning: Ray RLlib not available. PPO comparison will be skipped.")

# Import environment
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv


def train_ppo_fair(
    layout="cramped_room",
    horizon=400,
    total_steps=1_000_000,
    train_batch_size=4000,
    num_workers=4,
    seed=0,
    checkpoint_dir=None
):
    """
    Train PPO with a fixed total number of environment steps.
    
    Args:
        layout: Layout name
        horizon: Max steps per episode
        total_steps: Total environment steps to train on
        train_batch_size: Steps per training iteration
        num_workers: Number of parallel workers
        seed: Random seed
        checkpoint_dir: Where to save checkpoint
    
    Returns:
        Path to checkpoint
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray RLlib not available")
    
    # Calculate iterations needed
    num_iterations = max(1, total_steps // train_batch_size)
    
    print(f"Training PPO:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Batch size: {train_batch_size}")
    print(f"  Expected steps: {num_iterations * train_batch_size:,}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=num_workers + 1)
    
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
            train_batch_size=train_batch_size,
            minibatch_size=128,
            num_epochs=10,
        )
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=num_workers,
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
    
    # Set checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/ppo_fair_{layout}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build and train
    algo = config.build_algo()
    
    print(f"\nTraining for {num_iterations} iterations...")
    for i in range(num_iterations):
        result = algo.train()
        
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{num_iterations} - "
                  f"Reward: {result.get('env_runners/episode_return_mean', 'N/A')}")
    
    # Save final checkpoint
    checkpoint_path = algo.save(checkpoint_dir)
    print(f"\nPPO training complete. Checkpoint: {checkpoint_path}")
    
    algo.stop()
    
    return checkpoint_path


def evaluate_ppo(checkpoint_path, num_episodes=100, seed=0, verbose=False):
    """
    Evaluate trained PPO agent.
    
    Returns:
        List of episode rewards and lengths
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray RLlib not available")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Load algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Create environment
    env = OvercookedMultiAgentEnv(layout="cramped_room", horizon=400)
    
    rewards = []
    lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        step_count = 0
        
        while step_count < env.horizon:
            # Get actions (deterministic - no exploration)
            actions = {}
            for agent_id in ["agent_0", "agent_1"]:
                policy = algo.get_policy(agent_id)
                action = policy.compute_single_action(obs[agent_id], explore=False)
                if isinstance(action, tuple):
                    action = action[0]
                actions[agent_id] = int(action)
            
            obs, reward_dict, terminated, truncated, infos = env.step(actions)
            total_reward += reward_dict["agent_0"]
            step_count += 1
            
            if terminated["__all__"] or truncated["__all__"]:
                break
        
        rewards.append(total_reward)
        lengths.append(step_count)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes} - "
                  f"Reward: {total_reward:.4f}, Length: {step_count}")
    
    env.close()
    algo.stop()
    
    return rewards, lengths


def evaluate_aif(num_episodes, total_steps, eval_frequency, seed=0, verbose=False):
    """
    Run Active Inference agent with online learning.
    
    Key: AIF learns online, so we track performance throughout interaction.
    No separate evaluation phase - performance improves as it learns.
    
    Args:
        num_episodes: Number of episodes to run
        total_steps: Total interaction steps (for learning curve tracking)
        eval_frequency: Track performance every N steps
        seed: Random seed
        verbose: Whether to print progress
    
    Returns:
        rewards: List of episode rewards (final performance)
        lengths: List of episode lengths
        learning_curve: Dict with 'steps' and 'rewards' for learning curve
    """
    # TODO: Integrate with your AIF online learning code
    # This should:
    # 1. Initialize AIF agents
    # 2. Run episodes while learning online
    # 3. Track performance every eval_frequency steps
    # 4. Return final performance and learning curve
    
    print("Warning: AIF online learning not yet implemented in this script.")
    print("Please integrate with your AIF training code (e.g., run_overcooked_independent.py).")
    print("\nExpected behavior:")
    print(f"  - Run {num_episodes} episodes ({total_steps:,} total steps)")
    print(f"  - Learn online (no separate training/eval phases)")
    print(f"  - Track performance every {eval_frequency:,} steps")
    print(f"  - Return final performance (last N episodes) and learning curve")
    
    # Placeholder
    learning_curve = {
        'steps': [],
        'rewards': []
    }
    return [], [], learning_curve


def run_comparison(
    layout="cramped_room",
    horizon=400,
    total_interactions=1_000_000,  # Total environment interactions for BOTH
    training_ratio=0.9,  # For PPO: fraction used for training (rest for eval)
    eval_frequency=10_000,  # Track performance every N steps
    num_seeds=5,
    train_ppo=True,
    train_aif=False,  # Set to True when AIF training is integrated
    results_dir=None
):
    """
    Run fair comparison between PPO and Active Inference.
    
    Key: Both algorithms get the SAME total number of environment interactions.
    - PPO: Uses training_ratio for training, (1-training_ratio) for evaluation
    - AIF: Uses 100% for online learning (no separate eval phase)
    
    Args:
        layout: Layout name
        horizon: Max steps per episode
        total_interactions: Total environment interactions for BOTH algorithms
        training_ratio: For PPO, fraction of interactions used for training
        eval_frequency: Track performance every N steps (for learning curves)
        num_seeds: Number of random seeds to run
        train_ppo: Whether to train/evaluate PPO
        train_aif: Whether to train/evaluate AIF
        results_dir: Directory to save results
    """
    if results_dir is None:
        results_dir = f"comparison_results/ppo_vs_aif_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate step budgets
    ppo_training_steps = int(total_interactions * training_ratio)
    ppo_eval_steps = total_interactions - ppo_training_steps
    aif_total_steps = total_interactions  # AIF uses all for online learning
    
    print("="*80)
    print("FAIR COMPARISON: PPO vs Active Inference")
    print("="*80)
    print(f"Layout: {layout}")
    print(f"Horizon: {horizon}")
    print(f"Total interactions (BOTH): {total_interactions:,}")
    print(f"\nPPO Budget Split:")
    print(f"  Training steps: {ppo_training_steps:,} ({training_ratio*100:.0f}%)")
    print(f"  Evaluation steps: {ppo_eval_steps:,} ({(1-training_ratio)*100:.0f}%)")
    print(f"\nAIF Budget:")
    print(f"  Total interactions: {aif_total_steps:,} (100% online learning)")
    print(f"\nPerformance tracking: Every {eval_frequency:,} steps")
    print(f"Number of seeds: {num_seeds}")
    print(f"Results directory: {results_dir}")
    print("="*80)
    
    all_results = {
        "config": {
            "layout": layout,
            "horizon": horizon,
            "total_interactions": total_interactions,
            "training_ratio": training_ratio,
            "ppo_training_steps": ppo_training_steps,
            "ppo_eval_steps": ppo_eval_steps,
            "aif_total_steps": aif_total_steps,
            "eval_frequency": eval_frequency,
            "num_seeds": num_seeds
        },
        "ppo": [],
        "aif": []
    }
    
    # Run experiments for each seed
    for seed in range(num_seeds):
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")
        
        # PPO
        if train_ppo:
            print(f"\n[PPO] Training and evaluation...")
            try:
                # Train (using training portion of budget)
                checkpoint = train_ppo_fair(
                    layout=layout,
                    horizon=horizon,
                    total_steps=ppo_training_steps,  # Only training portion
                    seed=seed
                )
                
                # Evaluate (using eval portion of budget - counted in total!)
                num_eval_episodes = ppo_eval_steps // horizon
                print(f"[PPO] Final evaluation ({num_eval_episodes} episodes, {ppo_eval_steps:,} steps)...")
                rewards, lengths = evaluate_ppo(
                    checkpoint,
                    num_episodes=num_eval_episodes,
                    seed=seed,
                    verbose=True
                )
                
                all_results["ppo"].append({
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "rewards": rewards,
                    "lengths": lengths,
                    "mean_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "mean_length": float(np.mean(lengths)),
                    "std_length": float(np.std(lengths))
                })
                
                print(f"[PPO] Seed {seed} - "
                      f"Mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}, "
                      f"Mean length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
                
            except Exception as e:
                print(f"[PPO] Error on seed {seed}: {e}")
                import traceback
                traceback.print_exc()
        
        # Active Inference
        if train_aif:
            print(f"\n[AIF] Online learning (no separate evaluation phase)...")
            try:
                # AIF learns online - use all interactions for learning
                num_aif_episodes = aif_total_steps // horizon
                print(f"[AIF] Running {num_aif_episodes} episodes ({aif_total_steps:,} total steps)...")
                print(f"[AIF] Performance tracked throughout (online learning)")
                
                # TODO: Integrate AIF online learning
                # This should:
                # 1. Run episodes while learning online
                # 2. Track performance every eval_frequency steps
                # 3. Return learning curve and final performance
                rewards, lengths, learning_curve = evaluate_aif(
                    num_episodes=num_aif_episodes,
                    total_steps=aif_total_steps,
                    eval_frequency=eval_frequency,
                    seed=seed,
                    verbose=True
                )
                
                if rewards:
                    all_results["aif"].append({
                        "seed": seed,
                        "rewards": rewards,
                        "lengths": lengths,
                        "learning_curve": learning_curve,
                        "mean_reward": float(np.mean(rewards)),
                        "std_reward": float(np.std(rewards)),
                        "mean_length": float(np.mean(lengths)),
                        "std_length": float(np.std(lengths)),
                        "final_performance": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
                    })
                    
                    print(f"[AIF] Seed {seed} - "
                          f"Mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}, "
                          f"Mean length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
                
            except Exception as e:
                print(f"[AIF] Error on seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # Compute summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    if all_results["ppo"]:
        ppo_rewards = [r["mean_reward"] for r in all_results["ppo"]]
        ppo_lengths = [r["mean_length"] for r in all_results["ppo"]]
        
        print(f"\nPPO (across {len(ppo_rewards)} seeds):")
        print(f"  Mean reward: {np.mean(ppo_rewards):.4f} ± {np.std(ppo_rewards):.4f}")
        print(f"  Mean length: {np.mean(ppo_lengths):.1f} ± {np.std(ppo_lengths):.1f}")
        
        all_results["ppo_summary"] = {
            "mean_reward": float(np.mean(ppo_rewards)),
            "std_reward": float(np.std(ppo_rewards)),
            "mean_length": float(np.mean(ppo_lengths)),
            "std_length": float(np.std(ppo_lengths))
        }
    
    if all_results["aif"]:
        aif_rewards = [r["mean_reward"] for r in all_results["aif"]]
        aif_lengths = [r["mean_length"] for r in all_results["aif"]]
        
        print(f"\nActive Inference (across {len(aif_rewards)} seeds):")
        print(f"  Mean reward: {np.mean(aif_rewards):.4f} ± {np.std(aif_rewards):.4f}")
        print(f"  Mean length: {np.mean(aif_lengths):.1f} ± {np.std(aif_lengths):.1f}")
        
        all_results["aif_summary"] = {
            "mean_reward": float(np.mean(aif_rewards)),
            "std_reward": float(np.std(aif_rewards)),
            "mean_length": float(np.mean(aif_lengths)),
            "std_length": float(np.std(aif_lengths))
        }
    
    # Save results
    results_file = results_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Fair comparison: PPO vs Active Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Key Fairness Principle:
  Both algorithms get the SAME total number of environment interactions.
  - PPO: Uses --training-ratio for training, rest for evaluation
  - AIF: Uses 100%% for online learning (no separate eval phase)
  
Example:
  python compare_ppo_vs_aif.py --total-interactions 1000000 --training-ratio 0.9
  This gives PPO: 900K training + 100K eval = 1M total
           AIF:  1M online learning = 1M total
        """
    )
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--horizon", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--total-interactions", type=int, default=1_000_000, 
                       help="Total environment interactions for BOTH algorithms")
    parser.add_argument("--training-ratio", type=float, default=0.9,
                       help="For PPO: fraction of interactions used for training (rest for eval)")
    parser.add_argument("--eval-frequency", type=int, default=10_000,
                       help="Track performance every N steps (for learning curves)")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--train-ppo", action="store_true", default=True, help="Train/evaluate PPO")
    parser.add_argument("--train-aif", action="store_true", help="Train/evaluate Active Inference")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")
    
    args = parser.parse_args()
    
    run_comparison(
        layout=args.layout,
        horizon=args.horizon,
        total_interactions=args.total_interactions,
        training_ratio=args.training_ratio,
        eval_frequency=args.eval_frequency,
        num_seeds=args.num_seeds,
        train_ppo=args.train_ppo,
        train_aif=args.train_aif,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()
