"""
Train and evaluate PPO agents on RedBlueButton environment.

This script trains two PPO agents to collaboratively press buttons in the correct order.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import os
from datetime import datetime

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
from environments.RedBlueButton.RedBlueButtonMultiAgentEnv import RedBlueButtonMultiAgentEnv


def train_ppo(
    width=3,
    height=3,
    max_steps=50,
    num_iterations=1000,
    seed=0,
    checkpoint_dir=None,
    checkpoint_freq=100,
    verbose=True
):
    """
    Train PPO agents on RedBlueButton environment.
    
    Returns checkpoint path.
    """
    print(f"\n{'='*80}")
    print("Training PPO agents on RedBlueButton")
    print(f"{'='*80}")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=3)
    
    # Environment config
    env_config = {
        "width": width,
        "height": height,
        "max_steps": max_steps,
        "allow_same_position": False
    }
    
    # Create environment instance to get spaces
    env_instance = RedBlueButtonMultiAgentEnv(**env_config)
    
    # PPO config
    config = (
        PPOConfig()
        .environment(env=RedBlueButtonMultiAgentEnv, env_config=env_config)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            train_batch_size=2000,
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
    if checkpoint_dir is None:
        checkpoint_dir = f"checkpoints/ppo_redbluebutton_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build and train
    algo = config.build_algo()
    
    print(f"Training for {num_iterations} iterations...")
    checkpoint_path = checkpoint_dir  # Default fallback
    
    try:
        for i in range(num_iterations):
            result = algo.train()
            
            if (i + 1) % 10 == 0:
                # Try different metric names (new API stack uses different names)
                reward = (result.get('env_runners/episode_return_mean') or 
                         result.get('episode_reward_mean') or
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
                    print(f"  Iteration {i + 1}/{num_iterations} - Training...")
                    if i == 0:  # Show available metrics only once
                        available_metrics = [k for k in result.keys() if any(x in k.lower() for x in ['reward', 'return', 'episode', 'len'])]
                        if available_metrics:
                            print(f"    Available metrics: {', '.join(available_metrics[:5])}")
            
            # Save checkpoint
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_result = algo.save(checkpoint_dir)
                if verbose:
                    print(f"  Checkpoint saved at iteration {i + 1}")
        
        # Final checkpoint
        checkpoint_result = algo.save(checkpoint_dir)
        
        # Extract path from TrainingResult object
        try:
            if hasattr(checkpoint_result, 'checkpoint'):
                checkpoint_obj = checkpoint_result.checkpoint
                if hasattr(checkpoint_obj, 'path'):
                    checkpoint_path = checkpoint_obj.path
                else:
                    checkpoint_path = str(checkpoint_obj)
            elif isinstance(checkpoint_result, str):
                checkpoint_path = checkpoint_result
            else:
                checkpoint_path = checkpoint_dir
        except Exception as e:
            checkpoint_path = checkpoint_dir
        
        checkpoint_path = str(checkpoint_path) if checkpoint_path else checkpoint_dir
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.abspath(checkpoint_path)
        
        print(f"\nTraining complete!")
        print(f"Final checkpoint: {checkpoint_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving checkpoint...")
        try:
            checkpoint_result = algo.save(checkpoint_dir)
            # Extract path from TrainingResult object
            try:
                if hasattr(checkpoint_result, 'checkpoint'):
                    checkpoint_obj = checkpoint_result.checkpoint
                    if hasattr(checkpoint_obj, 'path'):
                        checkpoint_path = checkpoint_obj.path
                    else:
                        checkpoint_path = str(checkpoint_obj)
                elif isinstance(checkpoint_result, str):
                    checkpoint_path = checkpoint_result
                else:
                    checkpoint_path = checkpoint_dir
            except Exception:
                checkpoint_path = checkpoint_dir
            
            checkpoint_path = str(checkpoint_path) if checkpoint_path else checkpoint_dir
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.abspath(checkpoint_path)
            
            print(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
            checkpoint_path = checkpoint_dir
    finally:
        try:
            algo.stop()
        except Exception:
            pass
    
    return checkpoint_path


def evaluate_ppo(checkpoint_path, num_episodes=100, max_steps=50, seed=0, verbose=True):
    """
    Evaluate trained PPO agents.
    
    Returns list of episode results.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating PPO agents from checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Convert to absolute path if needed
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Load algorithm from checkpoint
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Environment config
    env_config = {
        "width": 3,
        "height": 3,
        "max_steps": max_steps,
        "allow_same_position": False
    }
    
    # Create environment
    env = RedBlueButtonMultiAgentEnv(**env_config)
    
    results = []
    
    print(f"Running {num_episodes} evaluation episodes...")
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            # Get actions from PPO agents
            try:
                import torch
                actions = {}
                for agent_id in ["agent_0", "agent_1"]:
                    try:
                        # Get the RLModule for this agent
                        module = algo.get_module(agent_id)
                        
                        # Get observation and convert to tensor
                        obs_agent = obs[agent_id]
                        if isinstance(obs_agent, (list, tuple)):
                            obs_agent = np.array(obs_agent)
                        obs_tensor = torch.FloatTensor(obs_agent).unsqueeze(0)
                        
                        # Compute action using forward_inference
                        with torch.no_grad():
                            fwd_outputs = module.forward_inference({"obs": obs_tensor})
                            
                            if "action_dist_inputs" in fwd_outputs:
                                try:
                                    action_dist_class = module.get_inference_action_dist_cls()
                                    action_dist = action_dist_class.from_logits(
                                        fwd_outputs["action_dist_inputs"]
                                    )
                                    if hasattr(action_dist, 'deterministic_sample'):
                                        action = action_dist.deterministic_sample()
                                    elif hasattr(action_dist, 'sample'):
                                        action = action_dist.sample()
                                    else:
                                        logits = fwd_outputs["action_dist_inputs"]
                                        action = torch.argmax(logits, dim=-1)
                                    
                                    if isinstance(action, torch.Tensor):
                                        action = action[0].item() if action.dim() > 0 else action.item()
                                    actions[agent_id] = int(action)
                                except Exception:
                                    logits = fwd_outputs["action_dist_inputs"]
                                    action = torch.argmax(logits, dim=-1)
                                    if isinstance(action, torch.Tensor):
                                        action = action[0].item() if action.dim() > 0 else action.item()
                                    actions[agent_id] = int(action)
                            elif "action" in fwd_outputs:
                                action = fwd_outputs["action"]
                                if isinstance(action, torch.Tensor):
                                    action = action[0].item() if action.dim() > 0 else action.item()
                                actions[agent_id] = int(action)
                            else:
                                # Fallback to random
                                actions[agent_id] = env.action_space[agent_id].sample()
                    except Exception as e:
                        if verbose and episode == 0:
                            print(f"Warning: Could not get PPO action for {agent_id} ({type(e).__name__}). Using random.")
                        actions[agent_id] = env.action_space[agent_id].sample()
            except Exception as e:
                if verbose and episode == 0:
                    print(f"Warning: Could not get PPO actions ({type(e).__name__}). Using random.")
                for agent_id in ["agent_0", "agent_1"]:
                    actions[agent_id] = env.action_space[agent_id].sample()
            
            # Step environment
            obs, rewards, terminated, truncated, infos = env.step(actions)
            episode_reward += rewards["agent_0"]  # Shared reward
            episode_length += 1
            done = terminated["__all__"] or truncated["__all__"]
        
        results.append({
            "episode": episode + 1,
            "reward": episode_reward,
            "length": episode_length,
            "won": episode_reward > 0,
            "lost": episode_reward < 0
        })
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}: Reward={episode_reward:.2f}, Length={episode_length}, Won={episode_reward > 0}")
    
    algo.stop()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PPO agents on RedBlueButton")
    parser.add_argument("--width", type=int, default=3, help="Grid width")
    parser.add_argument("--height", type=int, default=3, help="Grid height")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to checkpoint (if None, trains new agents)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=100,
                       help="Frequency of checkpoint saving")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Only evaluate, don't train (requires --checkpoint)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")
    
    args = parser.parse_args()
    
    if args.evaluate_only:
        if args.checkpoint is None:
            print("Error: --checkpoint required when using --evaluate-only")
            return
        results = evaluate_ppo(
            checkpoint_path=args.checkpoint,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            verbose=args.verbose
        )
    else:
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = train_ppo(
                width=args.width,
                height=args.height,
                max_steps=args.max_steps,
                num_iterations=args.num_iterations,
                seed=args.seed,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_freq=args.checkpoint_freq,
                verbose=args.verbose
            )
        
        # Evaluate
        results = evaluate_ppo(
            checkpoint_path=checkpoint_path,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            verbose=args.verbose
        )
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    rewards = [r["reward"] for r in results]
    wins = sum(1 for r in results if r["won"])
    losses = sum(1 for r in results if r["lost"])
    
    print(f"Episodes: {len(results)}")
    print(f"Mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"Wins: {wins} ({wins/len(results)*100:.1f}%)")
    print(f"Losses: {losses} ({losses/len(results)*100:.1f}%)")
    print(f"Neutral: {len(results) - wins - losses} ({(len(results) - wins - losses)/len(results)*100:.1f}%)")
    print(f"Mean episode length: {np.mean([r['length'] for r in results]):.2f}")
    
    ray.shutdown()


if __name__ == "__main__":
    main()
