"""
Run Trajectory Sampling Dyna-Q agents on Overcooked.

Runs two independent Trajectory Sampling Dyna-Q agents for a specified number of episodes, 
saves logs, and generates statistics. Each agent learns independently from its own observations.
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

# Import Trajectory Sampling Dyna-Q agent
from agents.QLearning.dynaq_agent_trajectory_sampling import DynaQAgent
import gymnasium as gym
from gymnasium import spaces


class OvercookedQLearningWrapper(gym.Env):
    """
    Wrapper to adapt Overcooked featurized observations for QLearning agents.
    
    QLearning agents expect observations with:
    - agent_pos: int (flat grid index 0-8 for 3x3, but Overcooked uses larger grids)
    - on_red_button: int (0 or 1)
    - on_blue_button: int (0 or 1)
    - red_button_state: int (0 or 1)
    - blue_button_state: int (0 or 1)
    
    This wrapper extracts discrete features from Overcooked's featurized state
    and creates a compatible observation format.
    """
    
    def __init__(self, base_env, agent_id):
        """
        Initialize wrapper.
        
        Args:
            base_env: OvercookedMultiAgentEnv instance
            agent_id: "agent_0" or "agent_1"
        """
        super().__init__()
        self.base_env = base_env
        self.agent_id = agent_id
        
        # Get layout dimensions from MDP
        self.mdp = base_env.mdp
        self.width = self.mdp.width
        self.height = self.mdp.height
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: dict with discrete features
        # Note: agent_pos will be a flat index based on grid position
        max_pos = self.width * self.height - 1
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Discrete(max_pos + 1),
            "on_red_button": spaces.Discrete(2),
            "on_blue_button": spaces.Discrete(2),
            "red_button_state": spaces.Discrete(2),
            "blue_button_state": spaces.Discrete(2),
        })
        
        # Track current state info
        self.current_state = None
        self.current_obs = None
    
    def _extract_discrete_features(self, featurized_obs, state):
        """
        Extract discrete features from featurized observation and state.
        
        Args:
            featurized_obs: Featurized observation vector from Overcooked
            state: Overcooked state object
            
        Returns:
            dict: Observation dict compatible with QLearning agents
        """
        # Get agent position from state
        if self.agent_id == "agent_0":
            player = state.players[0]
        else:
            player = state.players[1]
        
        pos = player.position
        x, y = int(pos[0]), int(pos[1])
        
        # Convert (x, y) to flat index: agent_pos = y * width + x
        agent_pos = y * self.width + x
        
        # Extract features from Overcooked state
        # We'll create a simplified discrete representation compatible with QLearning
        
        # Check if agent is at a pot location (as proxy for "button")
        pot_locations = self.mdp.get_pot_locations()
        
        on_pot = 0
        for pot_pos in pot_locations:
            if (x, y) == pot_pos:
                on_pot = 1
                break
        
        # Check if any pot is cooking (as proxy for "button pressed")
        pot_cooking = 0
        for pot_pos in pot_locations:
            if state.has_object(pot_pos):
                obj = state.get_object(pot_pos)
                # is_cooking is a boolean property (not a method), access it directly
                if hasattr(obj, 'is_cooking'):
                    is_cooking_value = obj.is_cooking
                    # Check if it's a boolean value (not callable)
                    if isinstance(is_cooking_value, bool) and is_cooking_value:
                        pot_cooking = 1
                        break
        
        # Create observation dict compatible with QLearning
        # QLearning expects: agent_pos, on_red_button, on_blue_button, red_button_state, blue_button_state
        obs = {
            "agent_pos": agent_pos,
            "on_red_button": on_pot,  # Using pot as proxy
            "on_blue_button": 0,  # Not used in Overcooked, set to 0
            "red_button_state": pot_cooking,  # Using cooking state as proxy
            "blue_button_state": 0,  # Not used in Overcooked, set to 0
        }
        
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment and return observation."""
        observations, infos = self.base_env.reset(seed=seed, options=options)
        
        # Get state from info
        state = infos[self.agent_id]["state"]
        self.current_state = state
        
        # Extract discrete features
        featurized_obs = observations[self.agent_id]
        obs = self._extract_discrete_features(featurized_obs, state)
        self.current_obs = obs
        
        return obs, infos.get(self.agent_id, {})
    
    def step(self, action):
        """
        Step environment with action.
        
        Note: This wrapper doesn't actually step the base environment.
        The main loop coordinates both agents and steps the base env.
        This method is called after the base env has been stepped to
        extract the observation for this agent.
        
        Args:
            action: Action index (0-5) - not used, kept for interface compatibility
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # This method should not be called directly - the main loop handles stepping
        # But we'll return the current observation if called
        if self.current_obs is not None:
            return self.current_obs, 0.0, False, False, {}
        else:
            # Fallback: return a default observation
            return {
                "agent_pos": 0,
                "on_red_button": 0,
                "on_blue_button": 0,
                "red_button_state": 0,
                "blue_button_state": 0,
            }, 0.0, False, False, {}
    
    def update_from_base_step(self, observations_next, rewards, terminated, truncated, infos):
        """
        Update wrapper state after base environment step.
        
        This is called by the main loop after stepping the base environment.
        """
        # Get state from info
        state = infos[self.agent_id]["state"]
        self.current_state = state
        
        # Extract discrete features
        featurized_obs = observations_next[self.agent_id]
        obs = self._extract_discrete_features(featurized_obs, state)
        self.current_obs = obs
        
        return obs, rewards[self.agent_id], terminated["__all__"] or truncated["__all__"], truncated.get(self.agent_id, False), infos.get(self.agent_id, {})


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


def run_episodes_trajectory_sampling_dynaq(env, agents, num_episodes, log_dir, verbose=True):
    """
    Run episodes with Trajectory Sampling Dyna-Q agents and log results.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        agents: Dict mapping agent_id to DynaQAgent
        num_episodes: Number of episodes to run
        log_dir: Directory to save logs
        verbose: Whether to print progress
    
    Returns:
        List of episode results (rewards, lengths, etc.)
    """
    episode_results = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {num_episodes} episodes with Trajectory Sampling Dyna-Q agents")
        print(f"{'='*80}")
    
    # Create wrapped environments for each agent
    wrapped_env_0 = OvercookedQLearningWrapper(env, "agent_0")
    wrapped_env_1 = OvercookedQLearningWrapper(env, "agent_1")
    
    for episode in range(num_episodes):
        # Reset environment
        observations, infos = env.reset()
        
        # Reset wrapped environments (they'll sync with base env)
        obs_0, _ = wrapped_env_0.reset()
        obs_1, _ = wrapped_env_1.reset()
        
        episode_log = {
            "episode": episode,
            "agent_type": "trajectory_sampling_dynaq",
            "steps": [],
            "total_reward": 0.0,
            "episode_length": 0
        }
        
        step_count = 0
        total_reward = 0.0
        
        while step_count < env.horizon:
            # Get actions from Trajectory Sampling Dyna-Q agents
            actions = {}
            
            try:
                # Agent 0: get state and choose action
                state_0 = agents["agent_0"].get_state(obs_0)
                action_0 = agents["agent_0"].choose_action(state_0)
                actions["agent_0"] = int(action_0)
                
                # Agent 1: same
                state_1 = agents["agent_1"].get_state(obs_1)
                action_1 = agents["agent_1"].choose_action(state_1)
                actions["agent_1"] = int(action_1)
                
            except Exception as e:
                if verbose and step_count == 0:
                    print(f"Warning: Could not get Dyna-Q actions ({type(e).__name__}: {e}). Using random.")
                actions["agent_0"] = wrapped_env_0.action_space.sample()
                actions["agent_1"] = wrapped_env_1.action_space.sample()
            
            # Step environment
            observations_next, rewards, terminated, truncated, infos = env.step(actions)
            
            # Update wrapped environments with results from base env step
            obs_0_next, reward_0, done_0, truncated_0, info_0 = wrapped_env_0.update_from_base_step(
                observations_next, rewards, terminated, truncated, infos
            )
            obs_1_next, reward_1, done_1, truncated_1, info_1 = wrapped_env_1.update_from_base_step(
                observations_next, rewards, terminated, truncated, infos
            )
            
            # Update Trajectory Sampling Dyna-Q agents
            state_0 = agents["agent_0"].get_state(obs_0)
            next_state_0 = agents["agent_0"].get_state(obs_0_next) if not done_0 else None
            
            # Update Q-table and model
            agents["agent_0"].update_q_table(state_0, actions["agent_0"], reward_0, next_state_0)
            agents["agent_0"].update_model(state_0, actions["agent_0"], next_state_0, reward_0, done_0)
            
            # Perform planning (trajectory sampling)
            agents["agent_0"].planning()
            
            # Same for agent 1
            state_1 = agents["agent_1"].get_state(obs_1)
            next_state_1 = agents["agent_1"].get_state(obs_1_next) if not done_1 else None
            
            agents["agent_1"].update_q_table(state_1, actions["agent_1"], reward_1, next_state_1)
            agents["agent_1"].update_model(state_1, actions["agent_1"], next_state_1, reward_1, done_1)
            agents["agent_1"].planning()
            
            # Decay epsilon
            agents["agent_0"].epsilon = max(agents["agent_0"].min_epsilon, 
                                            agents["agent_0"].epsilon * agents["agent_0"].epsilon_decay)
            agents["agent_1"].epsilon = max(agents["agent_1"].min_epsilon,
                                            agents["agent_1"].epsilon * agents["agent_1"].epsilon_decay)
            
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
                    "agent_0": float(reward_0),
                    "agent_1": float(reward_1),
                    "total": float(rewards["agent_0"])  # Shared reward
                }
            }
            
            episode_log["steps"].append(step_log)
            total_reward += rewards["agent_0"]
            step_count += 1
            
            obs_0 = obs_0_next
            obs_1 = obs_1_next
            
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
        log_file = log_dir / f"trajectory_sampling_dynaq_episode_{episode:04d}.json"
        with open(log_file, 'w') as f:
            json.dump(episode_log, f, indent=2)
        
        if verbose:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={total_reward:.4f}, Length={step_count}, "
                  f"Epsilon_0={agents['agent_0'].epsilon:.4f}, Epsilon_1={agents['agent_1'].epsilon:.4f}")
    
    return episode_results


def create_trajectory_sampling_dynaq_agents(env, action_space_size=6, planning_steps=10, 
                                            use_trajectory_sampling=True, n_trajectories=5,
                                            rollout_length=3, learning_rate=0.3, 
                                            discount_factor=0.95, epsilon=0.3, 
                                            epsilon_decay=0.995, min_epsilon=0.1,
                                            planning_epsilon=0.1, seed=0):
    """
    Create two independent Trajectory Sampling Dyna-Q agents.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        action_space_size: Number of actions (6 for Overcooked)
        planning_steps: Number of planning steps per real step
        use_trajectory_sampling: Enable trajectory sampling mode
        n_trajectories: Number of trajectories to simulate per planning step
        rollout_length: Maximum length of each simulated trajectory
        learning_rate: Q-learning learning rate
        discount_factor: Discount factor (gamma)
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate per step
        min_epsilon: Minimum epsilon value
        planning_epsilon: Exploration rate during simulated rollouts
        seed: Random seed
        
    Returns:
        Dict mapping agent_id to DynaQAgent
    """
    # Create wrapped environments
    wrapped_env_0 = OvercookedQLearningWrapper(env, "agent_0")
    wrapped_env_1 = OvercookedQLearningWrapper(env, "agent_1")
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Create Trajectory Sampling Dyna-Q agents
    agent_0 = DynaQAgent(
        action_space_size=action_space_size,
        planning_steps=planning_steps,
        q_table_path=None,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        load_existing=False,
        use_trajectory_sampling=use_trajectory_sampling,
        n_trajectories=n_trajectories,
        rollout_length=rollout_length,
        planning_epsilon=planning_epsilon
    )
    
    # Different seed for agent 1
    np.random.seed(seed + 1)
    random.seed(seed + 1)
    
    agent_1 = DynaQAgent(
        action_space_size=action_space_size,
        planning_steps=planning_steps,
        q_table_path=None,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        load_existing=False,
        use_trajectory_sampling=use_trajectory_sampling,
        n_trajectories=n_trajectories,
        rollout_length=rollout_length,
        planning_epsilon=planning_epsilon
    )
    
    return {
        "agent_0": agent_0,
        "agent_1": agent_1
    }


def plot_results(dynaq_results, output_dir, layout, horizon):
    """Create plots for Trajectory Sampling Dyna-Q agents."""
    output_dir = Path(output_dir)
    
    # Extract data
    dynaq_rewards = [r["total_reward"] for r in dynaq_results]
    dynaq_lengths = [r["episode_length"] for r in dynaq_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Trajectory Sampling Dyna-Q Agents Results\nLayout: {layout}, Horizon: {horizon}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Episode Rewards Over Time
    ax1 = axes[0, 0]
    episodes = range(1, len(dynaq_rewards) + 1)
    ax1.plot(episodes, dynaq_rewards, 'm-o', label='Trajectory Sampling Dyna-Q', linewidth=2, markersize=6)
    ax1.axhline(y=np.mean(dynaq_rewards), color='m', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(dynaq_rewards):.4f}')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=11)
    ax1.set_title('Episode Rewards Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Lengths Over Time
    ax2 = axes[0, 1]
    ax2.plot(episodes, dynaq_lengths, 'm-o', label='Trajectory Sampling Dyna-Q', linewidth=2, markersize=6)
    ax2.axhline(y=np.mean(dynaq_lengths), color='m', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(dynaq_lengths):.1f}')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Episode Length (steps)', fontsize=11)
    ax2.set_title('Episode Lengths Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward Distribution (Histogram)
    ax3 = axes[1, 0]
    ax3.hist(dynaq_rewards, bins=20, color='plum', edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(dynaq_rewards), color='m', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(dynaq_rewards):.4f}')
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
    
    Trajectory Sampling Dyna-Q Agents:
    • Mean Reward: {np.mean(dynaq_rewards):.4f}
    • Std Reward:  {np.std(dynaq_rewards):.4f}
    • Min Reward:  {np.min(dynaq_rewards):.4f}
    • Max Reward:  {np.max(dynaq_rewards):.4f}
    • Mean Length: {np.mean(dynaq_lengths):.1f}
    • Std Length:  {np.std(dynaq_lengths):.1f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'trajectory_sampling_dynaq_agents_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    # Also save as PDF
    plot_file_pdf = output_dir / 'trajectory_sampling_dynaq_agents_plot.pdf'
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Trajectory Sampling Dyna-Q agents on Overcooked")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--horizon", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for logs and plots")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--planning-steps", type=int, default=10, help="Planning steps per real step")
    parser.add_argument("--n-trajectories", type=int, default=5, help="Number of trajectories per planning step")
    parser.add_argument("--rollout-length", type=int, default=3, help="Maximum rollout length")
    parser.add_argument("--learning-rate", type=float, default=0.3, help="Q-learning learning rate")
    parser.add_argument("--discount-factor", type=float, default=0.95, help="Discount factor (gamma)")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--min-epsilon", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--planning-epsilon", type=float, default=0.1, help="Exploration rate during planning")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/trajectory_sampling_dynaq_agents_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRAJECTORY SAMPLING DYNA-Q AGENTS")
    print("="*80)
    print(f"Layout: {args.layout}")
    print(f"Horizon: {args.horizon}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Planning steps: {args.planning_steps}")
    print(f"Trajectories per step: {args.n_trajectories}")
    print(f"Rollout length: {args.rollout_length}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Discount factor: {args.discount_factor}")
    print(f"Epsilon: {args.epsilon} (decay: {args.epsilon_decay}, min: {args.min_epsilon})")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Create environment
    env = OvercookedMultiAgentEnv(layout=args.layout, horizon=args.horizon)
    
    # Create Trajectory Sampling Dyna-Q agents
    print("\nCreating Trajectory Sampling Dyna-Q agents...")
    dynaq_agents = create_trajectory_sampling_dynaq_agents(
        env,
        action_space_size=6,
        planning_steps=args.planning_steps,
        use_trajectory_sampling=True,
        n_trajectories=args.n_trajectories,
        rollout_length=args.rollout_length,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        planning_epsilon=args.planning_epsilon,
        seed=args.seed
    )
    print("✓ Agents ready")
    
    # Run episodes
    dynaq_results = run_episodes_trajectory_sampling_dynaq(
        env, dynaq_agents, args.num_episodes, output_dir, verbose=args.verbose
    )
    
    # Create plots
    print(f"\n{'='*80}")
    print("Generating plots...")
    plot_results(dynaq_results, output_dir, args.layout, args.horizon)
    
    # Extract data for summary
    dynaq_rewards = [r["total_reward"] for r in dynaq_results]
    dynaq_lengths = [r["episode_length"] for r in dynaq_results]
    
    # Save summary
    summary = {
        "config": {
            "layout": args.layout,
            "horizon": args.horizon,
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "planning_steps": args.planning_steps,
            "n_trajectories": args.n_trajectories,
            "rollout_length": args.rollout_length,
            "learning_rate": args.learning_rate,
            "discount_factor": args.discount_factor,
            "epsilon": args.epsilon,
            "epsilon_decay": args.epsilon_decay,
            "min_epsilon": args.min_epsilon,
            "planning_epsilon": args.planning_epsilon
        },
        "results": {
            "rewards": dynaq_rewards,
            "lengths": dynaq_lengths,
            "mean_reward": float(np.mean(dynaq_rewards)),
            "std_reward": float(np.std(dynaq_rewards)),
            "mean_length": float(np.mean(dynaq_lengths)),
            "std_length": float(np.std(dynaq_lengths))
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
    print(f"  - Plots: {output_dir / 'trajectory_sampling_dynaq_agents_plot.png'}")
    print(f"{'='*80}\n")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
