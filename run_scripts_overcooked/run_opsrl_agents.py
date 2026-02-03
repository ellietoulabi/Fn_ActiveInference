"""
Run OPSRL agents on Overcooked.

Runs two independent OPSRL agents for a specified number of episodes, saves logs,
and generates statistics. Each agent learns independently from its own observations.
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

# Import OPSRL agent
from agents.OPSRL import OPSRLAgent
import gymnasium as gym
from gymnasium import spaces


class OvercookedOPSRLWrapper(gym.Env):
    """
    Wrapper to adapt Overcooked featurized observations for OPSRL.
    
    OPSRL expects discrete state spaces with dict observations containing:
    - position: (x, y) coordinates
    - on_red_button: bool
    - on_blue_button: bool
    - red_button_pressed: bool
    - blue_button_pressed: bool
    
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
        # Overcooked uses (x, y) coordinates where x is column and y is row
        # Get the actual grid dimensions
        self.width = self.mdp.width
        self.height = self.mdp.height
        
        # Store these as attributes for OPSRL agent to access
        # OPSRL expects env.width and env.height attributes
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: dict with discrete features
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=0,
                high=max(self.width - 1, self.height - 1),
                shape=(2,),
                dtype=int
            ),
            "on_red_button": spaces.Discrete(2),
            "on_blue_button": spaces.Discrete(2),
            "red_button_pressed": spaces.Discrete(2),
            "blue_button_pressed": spaces.Discrete(2),
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
            dict: Observation dict compatible with OPSRL
        """
        # Get agent position from state
        if self.agent_id == "agent_0":
            player = state.players[0]
        else:
            player = state.players[1]
        
        pos = player.position
        x, y = int(pos[0]), int(pos[1])
        
        # Extract features from Overcooked state
        # We'll create a simplified discrete representation compatible with OPSRL
        
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
        
        # Create observation dict compatible with OPSRL
        # OPSRL expects: position, on_red_button, on_blue_button, red_button_pressed, blue_button_pressed
        obs = {
            "position": np.array([x, y], dtype=int),
            "on_red_button": on_pot,  # Using pot as proxy
            "on_blue_button": 0,  # Not used in Overcooked, set to 0
            "red_button_pressed": pot_cooking,  # Using cooking state as proxy
            "blue_button_pressed": 0,  # Not used in Overcooked, set to 0
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
                "position": np.array([0, 0], dtype=int),
                "on_red_button": 0,
                "on_blue_button": 0,
                "red_button_pressed": 0,
                "blue_button_pressed": 0,
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


def run_episodes_opsrl(env, agents, num_episodes, log_dir, verbose=True):
    """
    Run episodes with OPSRL agents and log results.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        agents: Dict mapping agent_id to OPSRLAgent
        num_episodes: Number of episodes to run
        log_dir: Directory to save logs
        verbose: Whether to print progress
    
    Returns:
        List of episode results (rewards, lengths, etc.)
    """
    episode_results = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {num_episodes} episodes with OPSRL agents")
        print(f"{'='*80}")
    
    # Create wrapped environments for each agent
    wrapped_env_0 = OvercookedOPSRLWrapper(env, "agent_0")
    wrapped_env_1 = OvercookedOPSRLWrapper(env, "agent_1")
    
    for episode in range(num_episodes):
        # Reset environment
        observations, infos = env.reset()
        
        # Reset wrapped environments (they'll sync with base env)
        obs_0, _ = wrapped_env_0.reset()
        obs_1, _ = wrapped_env_1.reset()
        
        episode_log = {
            "episode": episode,
            "agent_type": "opsrl",
            "steps": [],
            "total_reward": 0.0,
            "episode_length": 0
        }
        
        step_count = 0
        total_reward = 0.0
        
        # Train agents for one episode (OPSRL's _run_episode handles the episode loop)
        # But we need to coordinate both agents, so we'll step manually
        while step_count < env.horizon:
            # Get actions from OPSRL agents
            actions = {}
            
            try:
                # Agent 0: get action using policy (for evaluation) or _get_action (for exploration)
                # For now, use policy after initial training
                if hasattr(agents["agent_0"], 'Q_policy') and agents["agent_0"].Q_policy is not None:
                    action_0 = agents["agent_0"].policy(obs_0)
                else:
                    # Use random or exploration policy
                    state_0 = agents["agent_0"]._obs_to_state(obs_0)
                    action_0 = agents["agent_0"]._get_action(state_0, step_count) if state_0 is not None else wrapped_env_0.action_space.sample()
                actions["agent_0"] = int(action_0)
                
                # Agent 1: same
                if hasattr(agents["agent_1"], 'Q_policy') and agents["agent_1"].Q_policy is not None:
                    action_1 = agents["agent_1"].policy(obs_1)
                else:
                    state_1 = agents["agent_1"]._obs_to_state(obs_1)
                    action_1 = agents["agent_1"]._get_action(state_1, step_count) if state_1 is not None else wrapped_env_1.action_space.sample()
                actions["agent_1"] = int(action_1)
                
            except Exception as e:
                if verbose and step_count == 0:
                    print(f"Warning: Could not get OPSRL actions ({type(e).__name__}: {e}). Using random.")
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
            
            # Update OPSRL agents
            state_0 = agents["agent_0"]._obs_to_state(obs_0)
            next_state_0 = agents["agent_0"]._obs_to_state(obs_0_next) if not done_0 else None
            agents["agent_0"]._update(state_0, actions["agent_0"], next_state_0, reward_0, step_count)
            
            state_1 = agents["agent_1"]._obs_to_state(obs_1)
            next_state_1 = agents["agent_1"]._obs_to_state(obs_1_next) if not done_1 else None
            agents["agent_1"]._update(state_1, actions["agent_1"], next_state_1, reward_1, step_count)
            
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
        
        # After episode, update OPSRL agents (sample new posterior and recompute Q)
        # This is done by calling _run_episode internally, but we've already stepped manually
        # So we'll trigger the posterior sampling and Q update here
        try:
            # Sample new posterior and recompute Q for next episode
            B = agents["agent_0"].thompson_samples
            if agents["agent_0"].stage_dependent:
                M_sab_zero = np.repeat(agents["agent_0"].M_sa[..., 0, np.newaxis], B, -1)
                M_sab_one = np.repeat(agents["agent_0"].M_sa[..., 1, np.newaxis], B, -1)
                N_sasb = np.repeat(agents["agent_0"].N_sas[..., np.newaxis], B, axis=-1)
            else:
                M_sab_zero = np.repeat(agents["agent_0"].M_sa[..., 0, np.newaxis], B, -1)
                M_sab_one = np.repeat(agents["agent_0"].M_sa[..., 1, np.newaxis], B, -1)
                N_sasb = np.repeat(agents["agent_0"].N_sas[..., np.newaxis], B, axis=-1)
            
            # Sample rewards and transitions
            R_samples_0 = agents["agent_0"].rng.beta(M_sab_zero, M_sab_one)
            P_samples_0 = agents["agent_0"].rng.gamma(N_sasb) + 1e-10
            if agents["agent_0"].stage_dependent:
                sums = P_samples_0.sum(-2, keepdims=True)
                P_samples_0 = P_samples_0 / sums
            else:
                sums = P_samples_0.sum(-1, keepdims=True)
                P_samples_0 = P_samples_0 / sums
            R_samples_0 = 2.0 * R_samples_0 - 1.0
            
            # Update Q for agent 0
            from agents.OPSRL.utils import backward_induction_in_place, backward_induction_sd
            if agents["agent_0"].stage_dependent:
                backward_induction_sd(agents["agent_0"].Q, agents["agent_0"].V, R_samples_0, P_samples_0, 
                                     agents["agent_0"].gamma, agents["agent_0"].v_max[0])
            else:
                backward_induction_in_place(agents["agent_0"].Q, agents["agent_0"].V, R_samples_0, P_samples_0,
                                           agents["agent_0"].horizon, agents["agent_0"].gamma, agents["agent_0"].v_max[0])
            
            # Same for agent 1
            B = agents["agent_1"].thompson_samples
            if agents["agent_1"].stage_dependent:
                M_sab_zero = np.repeat(agents["agent_1"].M_sa[..., 0, np.newaxis], B, -1)
                M_sab_one = np.repeat(agents["agent_1"].M_sa[..., 1, np.newaxis], B, -1)
                N_sasb = np.repeat(agents["agent_1"].N_sas[..., np.newaxis], B, axis=-1)
            else:
                M_sab_zero = np.repeat(agents["agent_1"].M_sa[..., 0, np.newaxis], B, -1)
                M_sab_one = np.repeat(agents["agent_1"].M_sa[..., 1, np.newaxis], B, -1)
                N_sasb = np.repeat(agents["agent_1"].N_sas[..., np.newaxis], B, axis=-1)
            
            R_samples_1 = agents["agent_1"].rng.beta(M_sab_zero, M_sab_one)
            P_samples_1 = agents["agent_1"].rng.gamma(N_sasb) + 1e-10
            if agents["agent_1"].stage_dependent:
                sums = P_samples_1.sum(-2, keepdims=True)
                P_samples_1 = P_samples_1 / sums
            else:
                sums = P_samples_1.sum(-1, keepdims=True)
                P_samples_1 = P_samples_1 / sums
            R_samples_1 = 2.0 * R_samples_1 - 1.0
            
            if agents["agent_1"].stage_dependent:
                backward_induction_sd(agents["agent_1"].Q, agents["agent_1"].V, R_samples_1, P_samples_1,
                                    agents["agent_1"].gamma, agents["agent_1"].v_max[0])
            else:
                backward_induction_in_place(agents["agent_1"].Q, agents["agent_1"].V, R_samples_1, P_samples_1,
                                           agents["agent_1"].horizon, agents["agent_1"].gamma, agents["agent_1"].v_max[0])
        except Exception as e:
            if verbose:
                print(f"Warning: Could not update OPSRL Q functions: {e}")
        
        episode_log["total_reward"] = float(total_reward)
        episode_log["episode_length"] = step_count
        
        episode_results.append({
            "episode": episode,
            "total_reward": total_reward,
            "episode_length": step_count
        })
        
        # Save episode log
        log_file = log_dir / f"opsrl_episode_{episode:04d}.json"
        with open(log_file, 'w') as f:
            json.dump(episode_log, f, indent=2)
        
        if verbose:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward={total_reward:.4f}, Length={step_count}")
    
    # Compute final recommended policies
    if verbose:
        print("\nComputing final recommended policies...")
    
    try:
        from agents.OPSRL.utils import backward_induction_in_place, backward_induction_sd
        
        # Agent 0
        R_hat_0 = agents["agent_0"].M_sa[..., 0] / (agents["agent_0"].M_sa[..., 0] + agents["agent_0"].M_sa[..., 1])
        R_hat_0 = 2.0 * R_hat_0 - 1.0
        P_hat_0 = agents["agent_0"].N_sas / agents["agent_0"].N_sas.sum(-1, keepdims=True)
        if agents["agent_0"].stage_dependent:
            backward_induction_sd(agents["agent_0"].Q_policy, agents["agent_0"].V_policy, R_hat_0, P_hat_0,
                                 agents["agent_0"].gamma, agents["agent_0"].v_max[0])
        else:
            backward_induction_in_place(agents["agent_0"].Q_policy, agents["agent_0"].V_policy, R_hat_0, P_hat_0,
                                       agents["agent_0"].horizon, agents["agent_0"].gamma, agents["agent_0"].v_max[0])
        
        # Agent 1
        R_hat_1 = agents["agent_1"].M_sa[..., 0] / (agents["agent_1"].M_sa[..., 0] + agents["agent_1"].M_sa[..., 1])
        R_hat_1 = 2.0 * R_hat_1 - 1.0
        P_hat_1 = agents["agent_1"].N_sas / agents["agent_1"].N_sas.sum(-1, keepdims=True)
        if agents["agent_1"].stage_dependent:
            backward_induction_sd(agents["agent_1"].Q_policy, agents["agent_1"].V_policy, R_hat_1, P_hat_1,
                                agents["agent_1"].gamma, agents["agent_1"].v_max[0])
        else:
            backward_induction_in_place(agents["agent_1"].Q_policy, agents["agent_1"].V_policy, R_hat_1, P_hat_1,
                                       agents["agent_1"].horizon, agents["agent_1"].gamma, agents["agent_1"].v_max[0])
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute final policies: {e}")
    
    return episode_results


def create_opsrl_agents(env, horizon=150, gamma=0.99, seed=0):
    """
    Create two independent OPSRL agents.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        horizon: Horizon for OPSRL
        gamma: Discount factor
        seed: Random seed
        
    Returns:
        Dict mapping agent_id to OPSRLAgent
    """
    # Create wrapped environments
    wrapped_env_0 = OvercookedOPSRLWrapper(env, "agent_0")
    wrapped_env_1 = OvercookedOPSRLWrapper(env, "agent_1")
    
    # Create OPSRL agents
    agent_0 = OPSRLAgent(
        env=wrapped_env_0,
        gamma=gamma,
        horizon=horizon,
        bernoullized_reward=True,
        scale_prior_reward=1.0,
        thompson_samples=1,
        prior_transition='uniform',
        reward_free=False,
        stage_dependent=False,
        seed=seed
    )
    
    agent_1 = OPSRLAgent(
        env=wrapped_env_1,
        gamma=gamma,
        horizon=horizon,
        bernoullized_reward=True,
        scale_prior_reward=1.0,
        thompson_samples=1,
        prior_transition='uniform',
        reward_free=False,
        stage_dependent=False,
        seed=seed + 1  # Different seed for agent 1
    )
    
    return {
        "agent_0": agent_0,
        "agent_1": agent_1
    }


def plot_results(opsrl_results, output_dir, layout, horizon):
    """Create plots for OPSRL agents."""
    output_dir = Path(output_dir)
    
    # Extract data
    opsrl_rewards = [r["total_reward"] for r in opsrl_results]
    opsrl_lengths = [r["episode_length"] for r in opsrl_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'OPSRL Agents Results\nLayout: {layout}, Horizon: {horizon}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Episode Rewards Over Time
    ax1 = axes[0, 0]
    episodes = range(1, len(opsrl_rewards) + 1)
    ax1.plot(episodes, opsrl_rewards, 'g-o', label='OPSRL', linewidth=2, markersize=6)
    ax1.axhline(y=np.mean(opsrl_rewards), color='g', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(opsrl_rewards):.4f}')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=11)
    ax1.set_title('Episode Rewards Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Lengths Over Time
    ax2 = axes[0, 1]
    ax2.plot(episodes, opsrl_lengths, 'g-o', label='OPSRL', linewidth=2, markersize=6)
    ax2.axhline(y=np.mean(opsrl_lengths), color='g', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(opsrl_lengths):.1f}')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Episode Length (steps)', fontsize=11)
    ax2.set_title('Episode Lengths Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Reward Distribution (Histogram)
    ax3 = axes[1, 0]
    ax3.hist(opsrl_rewards, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(opsrl_rewards), color='g', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(opsrl_rewards):.4f}')
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
    
    OPSRL Agents:
    • Mean Reward: {np.mean(opsrl_rewards):.4f}
    • Std Reward:  {np.std(opsrl_rewards):.4f}
    • Min Reward:  {np.min(opsrl_rewards):.4f}
    • Max Reward:  {np.max(opsrl_rewards):.4f}
    • Mean Length: {np.mean(opsrl_lengths):.1f}
    • Std Length:  {np.std(opsrl_lengths):.1f}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'opsrl_agents_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    # Also save as PDF
    plot_file_pdf = output_dir / 'opsrl_agents_plot.pdf'
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run OPSRL agents on Overcooked")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--horizon", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for logs and plots")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--opsrl-horizon", type=int, default=150, help="OPSRL planning horizon")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"logs/opsrl_agents_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("OPSRL AGENTS")
    print("="*80)
    print(f"Layout: {args.layout}")
    print(f"Horizon: {args.horizon}")
    print(f"Episodes: {args.num_episodes}")
    print(f"OPSRL Horizon: {args.opsrl_horizon}")
    print(f"Gamma: {args.gamma}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Create environment
    env = OvercookedMultiAgentEnv(layout=args.layout, horizon=args.horizon)
    
    # Create OPSRL agents
    print("\nCreating OPSRL agents...")
    opsrl_agents = create_opsrl_agents(
        env,
        horizon=args.opsrl_horizon,
        gamma=args.gamma,
        seed=args.seed
    )
    print("✓ Agents ready")
    
    # Run OPSRL episodes
    opsrl_results = run_episodes_opsrl(
        env, opsrl_agents, args.num_episodes, output_dir, verbose=args.verbose
    )
    
    # Create plots
    print(f"\n{'='*80}")
    print("Generating plots...")
    plot_results(opsrl_results, output_dir, args.layout, args.horizon)
    
    # Extract data for summary
    opsrl_rewards = [r["total_reward"] for r in opsrl_results]
    opsrl_lengths = [r["episode_length"] for r in opsrl_results]
    
    # Save summary
    summary = {
        "config": {
            "layout": args.layout,
            "horizon": args.horizon,
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "gamma": args.gamma,
            "opsrl_horizon": args.opsrl_horizon
        },
        "results": {
            "rewards": opsrl_rewards,
            "lengths": opsrl_lengths,
            "mean_reward": float(np.mean(opsrl_rewards)),
            "std_reward": float(np.std(opsrl_rewards)),
            "mean_length": float(np.mean(opsrl_lengths)),
            "std_length": float(np.std(opsrl_lengths))
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
    print(f"  - Plots: {output_dir / 'opsrl_agents_plot.png'}")
    print(f"{'='*80}\n")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
