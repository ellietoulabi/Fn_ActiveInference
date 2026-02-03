"""
Multi-agent Gymnasium wrapper for Overcooked environment compatible with RLlib/marllib.

This wrapper converts the Overcooked environment to a format that works with
RLlib's multi-agent API, where each agent has its own observation and action space.
"""

import sys
from pathlib import Path

# Add overcooked_ai src to path
# Use the absolute path to the overcooked_ai environment
overcooked_ai_dir = Path("/Users/ellie/dev/thesis/Fn_ActiveInference/environments/overcooked_ai")
if not overcooked_ai_dir.exists():
    # Fallback to relative path if absolute doesn't exist
    project_root = Path(__file__).parent.parent.resolve()
    overcooked_ai_dir = project_root / "environments" / "overcooked_ai"

overcooked_src = overcooked_ai_dir / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))
else:
    raise ImportError(f"Could not find overcooked_ai source at {overcooked_src}")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    # Fallback if Ray is not available
    MultiAgentEnv = gym.Env


class OvercookedMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent Gymnasium wrapper for Overcooked.
    
    This environment returns observations and rewards for each agent separately,
    making it compatible with RLlib's multi-agent training.
    
    Observations are featurized states from the Overcooked MDP.
    Actions are discrete actions (0-5: NORTH, SOUTH, EAST, WEST, STAY, INTERACT).
    """
    
    metadata = {"render_modes": ["human"], "name": "OvercookedMultiAgent-v0"}
    
    def __init__(self, config=None, layout="cramped_room", horizon=400, num_pots=2):
        """
        Initialize the multi-agent Overcooked environment.
        
        Args:
            config: EnvContext from RLlib (contains env_config) or None
            layout: Layout name (e.g., "cramped_room")
            horizon: Maximum steps per episode
            num_pots: Number of pots to consider in featurization
        """
        super().__init__()
        
        # Handle RLlib EnvContext - it passes config as first positional arg
        if config is not None and hasattr(config, 'env_config'):
            # Extract from RLlib EnvContext
            env_config = config.env_config if hasattr(config, 'env_config') else {}
            layout = env_config.get("layout", layout)
            horizon = env_config.get("horizon", horizon)
            num_pots = env_config.get("num_pots", num_pots)
        elif isinstance(config, dict):
            # Direct dict config
            layout = config.get("layout", layout)
            horizon = config.get("horizon", horizon)
            num_pots = config.get("num_pots", num_pots)
        
        self.layout = layout
        self.horizon = horizon
        self.num_pots = num_pots
        
        # Create MDP and base environment
        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout)
        self.base_env = OvercookedEnv.from_mdp(
            self.mdp,
            horizon=horizon,
            info_level=0
        )
        
        # Get observation space from a dummy state
        dummy_state = self.mdp.get_standard_start_state()
        obs_p0, obs_p1 = self.base_env.featurize_state_mdp(dummy_state, num_pots=num_pots)
        
        # Observation space: featurized state vector
        obs_shape = obs_p0.shape
        self.observation_space = spaces.Dict({
            "agent_0": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            ),
            "agent_1": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            )
        })
        
        # Action space: 6 discrete actions per agent
        self.action_space = spaces.Dict({
            "agent_0": spaces.Discrete(len(Action.ALL_ACTIONS)),
            "agent_1": spaces.Discrete(len(Action.ALL_ACTIONS))
        })
        
        # Agent IDs for RLlib
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents.copy()
        
        # Reset environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset base environment
        self.base_env.reset()
        self.mdp = self.base_env.mdp
        
        # Get initial observations
        state = self.base_env.state
        obs_p0, obs_p1 = self.base_env.featurize_state_mdp(state, num_pots=self.num_pots)
        
        # Return observations as dict (RLlib format)
        observations = {
            "agent_0": obs_p0.astype(np.float32),
            "agent_1": obs_p1.astype(np.float32)
        }
        
        # Info dict
        infos = {
            "agent_0": {"state": state},
            "agent_1": {"state": state}
        }
        
        return observations, infos
    
    def step(self, actions):
        """
        Step the environment with actions from both agents.
        
        Args:
            actions: Dict with keys "agent_0" and "agent_1", values are action indices (0-5)
        
        Returns:
            observations: Dict of observations for each agent
            rewards: Dict of rewards for each agent (shared reward in Overcooked)
            terminated: Dict of termination flags
            truncated: Dict of truncation flags
            infos: Dict of info dicts for each agent
        """
        # Convert action indices to Action objects
        action_0 = Action.INDEX_TO_ACTION[actions["agent_0"]]
        action_1 = Action.INDEX_TO_ACTION[actions["agent_1"]]
        joint_action = (action_0, action_1)
        
        # Step base environment
        next_state, reward, done, env_info = self.base_env.step(joint_action)
        
        # Get observations for next state
        obs_p0, obs_p1 = self.base_env.featurize_state_mdp(next_state, num_pots=self.num_pots)
        
        # Format observations
        observations = {
            "agent_0": obs_p0.astype(np.float32),
            "agent_1": obs_p1.astype(np.float32)
        }
        
        # In Overcooked, both agents get the same reward (shared reward)
        rewards = {
            "agent_0": float(reward),
            "agent_1": float(reward)
        }
        
        # Termination and truncation
        terminated = {
            "agent_0": done,
            "agent_1": done,
            "__all__": done
        }
        
        truncated = {
            "agent_0": False,
            "agent_1": False,
            "__all__": False
        }
        
        # Info dicts
        infos = {
            "agent_0": {
                "state": next_state,
                "sparse_reward": env_info.get("sparse_r_by_agent", [0, 0])[0],
                "shaped_reward": env_info.get("shaped_r_by_agent", [0, 0])[0]
            },
            "agent_1": {
                "state": next_state,
                "sparse_reward": env_info.get("sparse_r_by_agent", [0, 0])[1],
                "shaped_reward": env_info.get("shaped_r_by_agent", [0, 0])[1]
            }
        }
        
        # Add episode info if done
        if done and "episode" in env_info:
            episode_info = env_info["episode"]
            for agent_id in self.agents:
                infos[agent_id]["episode"] = {
                    "ep_sparse_r": episode_info.get("ep_sparse_r", 0),
                    "ep_shaped_r": episode_info.get("ep_shaped_r", 0),
                    "ep_length": episode_info.get("ep_length", 0)
                }
        
        return observations, rewards, terminated, truncated, infos
    
    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            return self.base_env.render()
        return None
    
    def close(self):
        """Close the environment."""
        pass
