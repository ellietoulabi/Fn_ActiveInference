"""
Multi-agent Gymnasium wrapper for RedBlueButton environment compatible with RLlib.

This wrapper converts the TwoAgentRedBlueButtonEnv to a format that works with
RLlib's multi-agent API, where each agent has its own observation and action space.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    # Fallback if Ray is not available
    MultiAgentEnv = gym.Env

# Import with proper path handling
try:
    from environments.RedBlueButton.TwoAgentRedBlueButtonEnv import TwoAgentRedBlueButtonEnv
except ImportError:
    # Try alternative import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "TwoAgentRedBlueButtonEnv",
        Path(__file__).parent / "TwoAgentRedBlueButtonEnv.py"
    )
    TwoAgentRedBlueButtonEnv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TwoAgentRedBlueButtonEnv)
    TwoAgentRedBlueButtonEnv = TwoAgentRedBlueButtonEnv.TwoAgentRedBlueButtonEnv


class RedBlueButtonMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent Gymnasium wrapper for RedBlueButton.
    
    This environment returns observations and rewards for each agent separately,
    making it compatible with RLlib's multi-agent training.
    """
    
    metadata = {"render_modes": ["human"], "name": "RedBlueButtonMultiAgent-v0"}
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize the multi-agent RedBlueButton environment.
        
        Args:
            config: EnvContext from RLlib (contains env_config) or None
            **kwargs: Additional arguments passed to TwoAgentRedBlueButtonEnv
        """
        super().__init__()
        
        # Handle RLlib EnvContext - it passes config as first positional arg
        if config is not None and hasattr(config, 'env_config'):
            # Extract from RLlib EnvContext
            env_config = config.env_config if hasattr(config, 'env_config') else {}
            # Merge with kwargs
            kwargs.update(env_config)
        elif isinstance(config, dict):
            # Direct dict config
            kwargs.update(config)
        
        # Create base environment
        self.base_env = TwoAgentRedBlueButtonEnv(**kwargs)
        
        # Get observation space from a dummy reset
        dummy_obs, _ = self.base_env.reset()
        
        # Observation space: flatten dict observation to vector for each agent
        # We'll use the full observation dict for each agent
        obs_shape = self._flatten_obs(dummy_obs).shape
        
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
            "agent_0": spaces.Discrete(6),
            "agent_1": spaces.Discrete(6)
        })
        
        # Agent IDs for RLlib
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents.copy()
        
        # Reset environment
        self.reset()
    
    def _flatten_obs(self, obs_dict):
        """Flatten observation dict to a vector."""
        # obs_dict is already {'agent_0': {...}, 'agent_1': {...}}
        # We'll flatten one agent's observation (they're similar)
        agent_obs = obs_dict.get('agent_0', obs_dict.get('agent_1', {}))
        
        # Extract all values from agent observation dict
        values = []
        for key in sorted(agent_obs.keys()):
            val = agent_obs[key]
            if val is None:
                # Handle None values (e.g., button_just_pressed)
                values.append(0.0)
            elif isinstance(val, (list, tuple, np.ndarray)):
                values.extend(np.array(val).flatten().astype(float))
            elif isinstance(val, str):
                # Handle string values (e.g., "red", "blue", None)
                # Map to numeric: "red"=1, "blue"=2, None/other=0
                if val == "red":
                    values.append(1.0)
                elif val == "blue":
                    values.append(2.0)
                else:
                    values.append(0.0)
            else:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    values.append(0.0)
        return np.array(values, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset base environment
        obs_dict, info = self.base_env.reset(seed=seed, options=options)
        
        # Flatten observations for each agent (both get same observation in this env)
        obs_flat = self._flatten_obs(obs_dict)
        
        # Return observations as dict (RLlib format)
        observations = {
            "agent_0": obs_flat.copy(),
            "agent_1": obs_flat.copy()
        }
        
        # Info dict
        infos = {
            "agent_0": {"state": obs_dict},
            "agent_1": {"state": obs_dict}
        }
        
        return observations, infos
    
    def step(self, actions):
        """
        Step the environment with actions from both agents.
        
        Args:
            actions: Dict with keys "agent_0" and "agent_1", values are action indices (0-5)
        
        Returns:
            observations: Dict of observations for each agent
            rewards: Dict of rewards for each agent (shared reward in RedBlueButton)
            terminated: Dict of termination flags
            truncated: Dict of truncation flags
            infos: Dict of info dicts for each agent
        """
        # Convert action indices to dict format expected by base env
        action_dict = {
            "agent_0": actions["agent_0"],
            "agent_1": actions["agent_1"]
        }
        
        # Step base environment
        next_obs_dict, reward, terminated, truncated, env_info = self.base_env.step(action_dict)
        
        # Handle reward - could be dict or scalar
        if isinstance(reward, dict):
            # If reward is already a dict, extract scalar value
            reward_scalar = reward.get("agent_0", reward.get("agent_1", 0.0))
        else:
            reward_scalar = float(reward)
        
        # Flatten observations
        obs_flat_0 = self._flatten_obs({'agent_0': next_obs_dict.get('agent_0', {})})
        obs_flat_1 = self._flatten_obs({'agent_1': next_obs_dict.get('agent_1', {})})
        
        # Format observations
        observations = {
            "agent_0": obs_flat_0.copy(),
            "agent_1": obs_flat_1.copy()
        }
        
        # In RedBlueButton, both agents get the same reward (shared reward)
        rewards = {
            "agent_0": reward_scalar,
            "agent_1": reward_scalar
        }
        
        # Termination and truncation
        terminated_dict = {
            "agent_0": terminated,
            "agent_1": terminated,
            "__all__": terminated
        }
        
        truncated_dict = {
            "agent_0": truncated,
            "agent_1": truncated,
            "__all__": truncated
        }
        
        # Info dicts
        infos = {
            "agent_0": {
                "state": next_obs_dict,
                "reward": reward,
                "cumulative_reward": self.base_env.cumulative_reward
            },
            "agent_1": {
                "state": next_obs_dict,
                "reward": reward,
                "cumulative_reward": self.base_env.cumulative_reward
            }
        }
        
        # Add episode info if done
        if terminated or truncated:
            for agent_id in self.agents:
                infos[agent_id]["episode"] = {
                    "ep_reward": self.base_env.cumulative_reward,
                    "ep_length": self.base_env.step_count
                }
        
        return observations, rewards, terminated_dict, truncated_dict, infos
    
    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            return self.base_env.render()
        return None
    
    def close(self):
        """Close the environment."""
        pass
