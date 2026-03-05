"""
Single-agent Gymnasium wrapper for Overcooked using a one-player layout (e.g. cramped_room_single).

Uses the same Overcooked MDP as the multi-agent env, but loads a layout with a single
player (e.g. cramped_room_single.layout). Exposes a single agent (agent_0) so that
reset/step return observations and infos keyed by "agent_0" for compatibility with
code that expects env_utils.env_obs_to_model_obs(state, 0, ...) and get_D_config_from_state(state, 0).

Note: The underlying Overcooked-AI library uses a MediumLevelActionManager / JointMotionPlanner
that may assume two players. If you get an error loading cramped_room_single (e.g. missing
cramped_room_single_am.pkl or IndexError in planners), the run script can fall back to
OvercookedMultiAgentEnv with layout "cramped_room" and a dummy second agent.
"""

import sys
from pathlib import Path

overcooked_ai_dir = Path("/Users/ellie/dev/thesis/Fn_ActiveInference/environments/overcooked_ai")
if not overcooked_ai_dir.exists():
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


class OvercookedSingleAgentEnv(gym.Env):
    """
    Single-agent Overcooked environment using a one-player layout (e.g. cramped_room_single).

    Observations and infos are keyed by "agent_0" so that code written for the multi-agent
    wrapper (e.g. env_utils.env_obs_to_model_obs(state, 0, ...)) works unchanged.
    step() takes a single action index (0-5) and returns the same dict structure as
    OvercookedMultiAgentEnv but only with agent_0.
    """

    metadata = {"render_modes": ["human"], "name": "OvercookedSingleAgent-v0"}

    def __init__(self, config=None, layout="cramped_room_single", horizon=400, num_pots=2):
        super().__init__()

        if config is not None and hasattr(config, "env_config"):
            env_config = config.env_config if hasattr(config, "env_config") else {}
            layout = env_config.get("layout", layout)
            horizon = env_config.get("horizon", horizon)
            num_pots = env_config.get("num_pots", num_pots)
        elif isinstance(config, dict):
            layout = config.get("layout", layout)
            horizon = config.get("horizon", horizon)
            num_pots = config.get("num_pots", num_pots)

        self.layout = layout
        self.horizon = horizon
        self.num_pots = num_pots

        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name=layout,
            old_dynamics=True,
        )
        assert self.mdp.num_players == 1, (
            f"OvercookedSingleAgentEnv expects a one-player layout; "
            f"{layout} has num_players={self.mdp.num_players}. Use cramped_room_single."
        )

        self.base_env = OvercookedEnv.from_mdp(
            self.mdp,
            horizon=horizon,
            info_level=0,
        )

        # Use MDP's feature shape without calling featurize_state_mdp (that would create
        # MediumLevelActionManager, which assumes 2 players -> tuple index out of range).
        obs_shape = self.mdp.get_featurize_state_shape(num_pots=num_pots)

        self.observation_space = spaces.Dict({
            "agent_0": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32,
            )
        })
        self.action_space = spaces.Dict({
            "agent_0": spaces.Discrete(len(Action.ALL_ACTIONS))
        })
        self.agents = ["agent_0"]
        self.possible_agents = self.agents.copy()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env.reset()
        self.mdp = self.base_env.mdp

        state = self.base_env.state
        # Dummy obs; run script uses env_utils.env_obs_to_model_obs(state, ...) from infos.
        obs_shape = self.mdp.get_featurize_state_shape(num_pots=self.num_pots)
        observations = {"agent_0": np.zeros(obs_shape, dtype=np.float32)}
        infos = {"agent_0": {"state": state}}

        return observations, infos

    def step(self, actions):
        """
        Step with a single action.

        Args:
            actions: Either an int (action index 0-5) or a dict {"agent_0": action_idx}.

        Returns:
            observations, rewards, terminated, truncated, infos (all with only "agent_0" key).
        """
        if isinstance(actions, dict):
            action_idx = int(actions["agent_0"])
        else:
            action_idx = int(actions)

        action = Action.INDEX_TO_ACTION[action_idx]
        joint_action = (action,)  # single-player tuple

        next_state, reward, done, env_info = self.base_env.step(joint_action)

        # Dummy obs; run script uses env_utils.env_obs_to_model_obs(state, ...) from infos.
        obs_shape = self.mdp.get_featurize_state_shape(num_pots=self.num_pots)
        observations = {"agent_0": np.zeros(obs_shape, dtype=np.float32)}
        rewards = {"agent_0": float(reward)}
        terminated = {"agent_0": done, "__all__": done}
        truncated = {"agent_0": False, "__all__": False}
        infos = {
            "agent_0": {
                "state": next_state,
                "sparse_reward": env_info.get("sparse_r_by_agent", [0])[0],
                "shaped_reward": env_info.get("shaped_r_by_agent", [0])[0],
            }
        }

        if done and "episode" in env_info:
            episode_info = env_info["episode"]
            infos["agent_0"]["episode"] = {
                "ep_sparse_r": episode_info.get("ep_sparse_r", 0),
                "ep_shaped_r": episode_info.get("ep_shaped_r", 0),
                "ep_length": episode_info.get("ep_length", 0),
            }

        return observations, rewards, terminated, truncated, infos

    def render(self, mode="human"):
        if mode == "human":
            return self.base_env.render()
        return None

    def close(self):
        pass
