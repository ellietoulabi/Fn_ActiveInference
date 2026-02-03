"""
Train multi-agent PPO on Overcooked using Ray RLlib.

This script uses Ray RLlib to train two PPO agents to collaborate
in the Overcooked environment.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from datetime import datetime
import os

# Import RLlib
try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    import ray
except ImportError:
    print("Error: Ray RLlib not installed. Please install it with:")
    print("  pip install 'ray[rllib]'")
    sys.exit(1)

# Import our environment
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv


def train_ppo(
    layout="cramped_room",
    horizon=400,
    num_iterations=1000,
    num_workers=4,
    num_gpus=0,
    learning_rate=3e-4,
    gamma=0.99,
    lambda_=0.95,
    clip_param=0.2,
    entropy_coeff=0.01,
    vf_loss_coeff=0.5,
    train_batch_size=4000,
    sgd_minibatch_size=128,
    num_sgd_iter=10,
    checkpoint_dir=None,
    checkpoint_freq=50,
    seed=0,
    verbose=True
):
    """
    Train multi-agent PPO on Overcooked.
    
    Args:
        layout: Overcooked layout name
        horizon: Maximum steps per episode
        num_iterations: Number of training iterations
        num_workers: Number of parallel workers
        num_gpus: Number of GPUs to use
        learning_rate: PPO learning rate
        gamma: Discount factor
        lambda_: GAE lambda parameter
        clip_param: PPO clip parameter
        entropy_coeff: Entropy coefficient for exploration
        vf_loss_coeff: Value function loss coefficient
        train_batch_size: Training batch size
        sgd_minibatch_size: SGD minibatch size
        num_sgd_iter: Number of SGD iterations per update
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Frequency of checkpointing
        seed: Random seed
        verbose: Whether to print progress
    """
    print("="*80)
    print("MULTI-AGENT PPO TRAINING - OVERCOOKED")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Layout: {layout}")
    print(f"  Horizon: {horizon}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Workers: {num_workers}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Seed: {seed}")
    print("="*80)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=num_workers + 1)
    
    # Environment configuration
    env_config = {
        "layout": layout,
        "horizon": horizon,
        "num_pots": 2
    }
    
    # Create environment instance to get observation/action spaces
    env_instance = OvercookedMultiAgentEnv(
        layout=layout,
        horizon=horizon
    )
    
    # PPO configuration
    config = (
        PPOConfig()
        .environment(
            env=OvercookedMultiAgentEnv,
            env_config=env_config,
        )
        .training(
            lr=learning_rate,
            gamma=gamma,
            lambda_=lambda_,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=vf_loss_coeff,
            train_batch_size=train_batch_size,
            minibatch_size=sgd_minibatch_size,
            num_epochs=num_sgd_iter,
        )
        .resources(
            num_gpus=num_gpus,
        )
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
    
    # Set checkpoint directory (must be absolute path for Ray)
    if checkpoint_dir is None:
        checkpoint_dir = os.path.abspath(f"checkpoints/overcooked_ppo_{layout}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build algorithm
    algo = config.build_algo()
    
    # Training loop
    print(f"\nStarting training...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}\n")
    
    try:
        for i in range(num_iterations):
            result = algo.train()
            
            if verbose and (i + 1) % 10 == 0:
                print(f"\nIteration {i + 1}/{num_iterations}")
                print(f"  Episode reward mean: {result.get('episode_reward_mean', 'N/A')}")
                print(f"  Episode len mean: {result.get('episode_len_mean', 'N/A')}")
                
                # Get policy rewards if available
                policy_reward_mean = result.get('policy_reward_mean', {})
                if policy_reward_mean:
                    print(f"  Policy reward mean (agent_0): {policy_reward_mean.get('agent_0', 'N/A')}")
                    print(f"  Policy reward mean (agent_1): {policy_reward_mean.get('agent_1', 'N/A')}")
            
            # Save checkpoint
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_path = algo.save(checkpoint_dir)
                if verbose:
                    print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Final checkpoint
        final_checkpoint = algo.save(checkpoint_dir)
        print(f"\nTraining complete!")
        print(f"Final checkpoint: {final_checkpoint}")
        
    finally:
        algo.stop()
        ray.shutdown()
    
    return algo, checkpoint_dir


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train multi-agent PPO on Overcooked")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--horizon", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_", help="GAE lambda")
    parser.add_argument("--clip-param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--train-batch-size", type=int, default=4000, help="Training batch size")
    parser.add_argument("--sgd-minibatch-size", type=int, default=128, help="SGD minibatch size")
    parser.add_argument("--num-sgd-iter", type=int, default=10, help="Number of SGD iterations")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="Checkpoint frequency")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    try:
        train_ppo(
            layout=args.layout,
            horizon=args.horizon,
            num_iterations=args.iterations,
            num_workers=args.workers,
            num_gpus=args.gpus,
            learning_rate=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_freq=args.checkpoint_freq,
            seed=args.seed,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
