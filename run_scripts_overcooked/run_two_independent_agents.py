"""
Run two independent Active Inference agents in Overcooked environment.

Each agent uses a single-agent generative model and selects only its own action.
The other agent is treated as part of the environment.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import json
import csv
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import Active Inference agent
from agents.ActiveInference.agent import Agent

# Import Independent paradigm model
from generative_models.MA_ActiveInference.Overcooked.cramped_room.Independent import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)

# Import environment
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
from overcooked_ai_py.mdp.actions import Action


ACTION_NAMES = {
    0: "NORTH",
    1: "SOUTH",
    2: "EAST",
    3: "WEST",
    4: "STAY",
    5: "INTERACT"
}


def create_independent_agent(agent_idx, seed=None):
    """
    Create an independent Active Inference agent.
    
    Args:
        agent_idx: Agent index (0 or 1)
        seed: Random seed for reproducibility
    
    Returns:
        Agent instance
    """
    if seed is not None:
        np.random.seed(seed + agent_idx)
    
    # Environment parameters
    env_params = {
        "width": model_init.GRID_WIDTH,
        "height": model_init.GRID_HEIGHT,
    }
    
    # State factors and sizes
    state_factors = list(model_init.states.keys())
    state_sizes = {f: len(v) for f, v in model_init.states.items()}
    
    # Observation labels
    observation_labels = model_init.observations
    
    # Create agent
    agent = Agent(
        A_fn=A_fn,
        B_fn=B_fn,
        C_fn=C_fn,
        D_fn=D_fn,
        state_factors=state_factors,
        state_sizes=state_sizes,
        observation_labels=observation_labels,
        env_params=env_params,
        observation_state_dependencies=model_init.observation_state_dependencies,
        actions=list(range(model_init.N_ACTIONS)),
        gamma=2.0,  # Policy precision
        alpha=1.0,  # Action precision
        policy_len=2,
        inference_horizon=2,
        action_selection="deterministic",
        sampling_mode="full",
        inference_algorithm="VANILLA",
        num_iter=16,
        dF_tol=0.001,
    )
    
    return agent


def run_episode(env, agent0, agent1, episode_num, max_steps, csv_writer=None, seed=None):
    """
    Run a single episode with two independent agents.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        agent0: Agent 0 instance
        agent1: Agent 1 instance
        episode_num: Episode number
        max_steps: Maximum steps per episode
        csv_writer: Optional CSV writer for logging
        seed: Random seed
    
    Returns:
        Dictionary with episode results
    """
    # Reset environment
    observations, infos = env.reset()
    
    # Get initial state for D config
    state = infos["agent_0"]["state"]
    
    # Reset agents with initial configuration
    config0 = env_utils.get_D_config_from_mdp(env.mdp, state, agent_idx=0)
    config1 = env_utils.get_D_config_from_mdp(env.mdp, state, agent_idx=1)
    
    agent0.reset(config=config0)
    agent1.reset(config=config1)
    
    episode_reward = 0.0
    episode_length = 0
    prev_reward_info = {"sparse_reward_by_agent": [0, 0]}
    
    for step in range(1, max_steps + 1):
        # Convert environment observations to model observations
        # Use previous step's reward info for soup delivery detection
        obs0_model = env_utils.env_obs_to_model_obs(state, agent_idx=0, reward_info=prev_reward_info)
        obs1_model = env_utils.env_obs_to_model_obs(state, agent_idx=1, reward_info=prev_reward_info)
        
        # Get actions from agents (model actions are already indices 0-5)
        action0_idx = int(agent0.step(obs0_model))
        action1_idx = int(agent1.step(obs1_model))
        
        # Step environment (expects action indices 0-5)
        actions = {
            "agent_0": action0_idx,
            "agent_1": action1_idx
        }
        
        observations, rewards, terminated, truncated, infos = env.step(actions)
        
        # Update state for next iteration
        state = infos["agent_0"]["state"]
        
        # Update reward info for next step (sparse rewards indicate soup delivery)
        prev_reward_info = {
            "sparse_reward_by_agent": [
                infos["agent_0"].get("sparse_reward", 0),
                infos["agent_1"].get("sparse_reward", 0)
            ]
        }
        
        # Update episode statistics
        episode_reward += rewards["agent_0"]  # Shared reward
        episode_length = step
        
        # Log step if CSV writer provided
        if csv_writer is not None:
            csv_writer.writerow({
                "episode": episode_num,
                "step": step,
                "agent0_action": action0_idx,
                "agent0_action_name": ACTION_NAMES.get(action0_idx, str(action0_idx)),
                "agent1_action": action1_idx,
                "agent1_action_name": ACTION_NAMES.get(action1_idx, str(action1_idx)),
                "reward": rewards["agent_0"],
                "cumulative_reward": episode_reward,
                "terminated": terminated["__all__"],
                "truncated": truncated["__all__"],
            })
        
        # Check if episode is done
        if terminated["__all__"] or truncated["__all__"]:
            break
    
    return {
        "episode": episode_num,
        "reward": episode_reward,
        "length": episode_length,
        "success": episode_reward > 0,  # Success if any soup was delivered
    }


def main():
    parser = argparse.ArgumentParser(description="Run two independent Active Inference agents in Overcooked")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    
    args = parser.parse_args()
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV log file
    csv_path = log_dir / f"two_independent_agents_{timestamp}.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode", "step", "agent0_action", "agent0_action_name",
        "agent1_action", "agent1_action_name", "reward", "cumulative_reward",
        "terminated", "truncated"
    ])
    csv_writer.writeheader()
    
    # Create environment
    if args.verbose:
        print(f"Creating Overcooked environment: layout={args.layout}, horizon={args.max_steps}")
    
    env = OvercookedMultiAgentEnv(
        config={"layout": args.layout, "horizon": args.max_steps}
    )
    
    # Create agents
    if args.verbose:
        print("Creating two independent Active Inference agents...")
    
    agent0 = create_independent_agent(agent_idx=0, seed=args.seed)
    agent1 = create_independent_agent(agent_idx=1, seed=args.seed if args.seed is None else args.seed + 1000)
    
    # Run episodes
    if args.verbose:
        print(f"\n{'='*80}")
        print(f"Running {args.num_episodes} episodes")
        print(f"{'='*80}\n")
    
    episode_results = []
    
    for episode in range(args.num_episodes):
        result = run_episode(
            env, agent0, agent1, episode, args.max_steps,
            csv_writer=csv_writer, seed=args.seed
        )
        episode_results.append(result)
        
        if args.verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean([r["reward"] for r in episode_results[-10:]])
            avg_length = np.mean([r["length"] for r in episode_results[-10:]])
            success_rate = np.mean([r["success"] for r in episode_results[-10:]])
            print(f"Episode {episode + 1}/{args.num_episodes}: "
                  f"Reward={avg_reward:.2f}, Length={avg_length:.1f}, Success={success_rate:.2%}")
    
    csv_file.close()
    
    # Calculate statistics
    total_rewards = [r["reward"] for r in episode_results]
    episode_lengths = [r["length"] for r in episode_results]
    successes = [r["success"] for r in episode_results]
    
    stats = {
        "num_episodes": args.num_episodes,
        "max_steps_per_episode": args.max_steps,
        "layout": args.layout,
        "seed": args.seed,
        "total_reward_mean": float(np.mean(total_rewards)),
        "total_reward_std": float(np.std(total_rewards)),
        "total_reward_min": float(np.min(total_rewards)),
        "total_reward_max": float(np.max(total_rewards)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "total_soups_delivered": float(np.sum(total_rewards) / 20.0),  # Assuming 20 points per soup
    }
    
    # Save statistics
    stats_path = log_dir / f"two_independent_agents_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EPISODE SUMMARY")
    print(f"{'='*80}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Layout: {args.layout}")
    print(f"\nRewards:")
    print(f"  Mean: {stats['total_reward_mean']:.2f} ± {stats['total_reward_std']:.2f}")
    print(f"  Range: [{stats['total_reward_min']:.2f}, {stats['total_reward_max']:.2f}]")
    print(f"\nEpisode Lengths:")
    print(f"  Mean: {stats['episode_length_mean']:.1f} ± {stats['episode_length_std']:.1f}")
    print(f"\nSuccess Rate: {stats['success_rate']:.2%}")
    print(f"Total Soups Delivered: {stats['total_soups_delivered']:.1f}")
    print(f"\nLogs saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Stats: {stats_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
