"""
Minimal MAPPO on Overcooked with AIF-style semantic action space.

No reward shaping. PPO sees:
  - obs: AIF-style discrete-factor observation flattened to a vector
         (self_pos, self_ori, self_held, other_pos, other_held,
          pot_state, soup_delivered, [counter contents...])
         Same factors used by the AIF agent (Independent paradigm).
  - act: Discrete(N_ACTIONS=20) semantic options (destination, mode).
         Each step we replan the chosen semantic option from the agent's
         current state using BFS shortest-path planning, and execute ONLY
         the FIRST primitive action of the resulting path. NO teleporting.
  - reward: env's sparse delivery reward (per agent, shared).

Shortest-path semantic controller rules (mirrors dyn_utils.compile_semantic_policy):
  - Not at target tile  -> first primitive movement along shortest path.
  - At target tile, wrong facing -> primitive rotation toward target.
  - At target pose, mode=interact -> INTERACT.
  - At target pose, mode=stay -> STAY.
  - Path blocked (e.g., by other agent) -> retry without blocking; else STAY.

Uses Ray RLlib PPO (same library stack as
run_scripts_red_blue_doors/multi_agent/run_two_ppo_agents.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.IndependentActiveInferenceWithDynamicPolicies import utils as dyn_utils
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndependentWithSemanticPoliciesActionLevel import (
    env_utils as aif_env_utils,
    model_init as aif_model_init,
)

try:
    import ray
    import torch
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.rllib.utils.annotations import override
    import torch.nn as nn

    RAY_AVAILABLE = True
except ImportError as e:
    RAY_AVAILABLE = False
    IMPORT_ERR = str(e)
    MultiAgentEnv = object


OBS_KEYS = [
    "self_pos_obs",
    "self_orientation_obs",
    "self_held_obs",
    "other_pos_obs",
    "other_orientation_obs",
    "other_held_obs",
    "pot_state_obs",
    "soup_delivered_obs",
] + [f"ctr_{idx}_obs" for idx in aif_model_init.MODELED_COUNTERS]

# other_orientation_obs was previously missing here even though the shared
# env_obs_to_model_obs() function (the same one AIF's Independent agent calls)
# computes and returns it every step -- MAPPO was silently getting less state
# information than the AIF baseline it's compared against. See ai/02-debug.md
# section L. Added for observation parity.
OBS_NORM_DEN = np.array(
    [
        max(1, aif_model_init.N_WALKABLE - 1),
        max(1, aif_model_init.N_DIRECTIONS - 1),
        max(1, aif_model_init.N_HELD_TYPES - 1),
        max(1, aif_model_init.N_WALKABLE - 1),
        max(1, aif_model_init.N_DIRECTIONS - 1),
        max(1, aif_model_init.N_HELD_TYPES - 1),
        max(1, aif_model_init.N_POT_STATES - 1),
        1,
    ] + [max(1, aif_model_init.N_CTR_STATES - 1) for _ in aif_model_init.MODELED_COUNTERS],
    dtype=np.float32,
)


OWN_OBS_DIM = len(OBS_KEYS)


def _obs_dict_to_vec(obs_dict: Dict[str, int]) -> np.ndarray:
    vals = np.asarray([float(obs_dict[k]) for k in OBS_KEYS], dtype=np.float32)
    return vals / OBS_NORM_DEN


# Map utils-primitive indices (used by dyn_utils paths) to env primitives.
#   utils: 0=N, 1=S, 2=W, 3=E, 4=STAY, 5=INTERACT
#   env:   0=N, 1=S, 2=E, 3=W, 4=STAY, 5=INTERACT
UTILS_TO_ENV_ACTION = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5}

# Orientation int (model_init) -> dyn_utils name
ORI_IDX_TO_NAME = {0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST"}
HELD_IDX_TO_NAME = {0: "nothing", 1: "onion", 2: "dish", 3: "soup"}


def _extract_counter_contents(state) -> Dict[str, str]:
    counter_contents = {name: "empty" for name in dyn_utils.COUNTER_TILES.keys()}
    rc_to_name = {rc: name for name, rc in dyn_utils.COUNTER_TILES.items()}
    for pos_xy, obj in state.objects.items():
        if obj is None:
            continue
        obj_name = getattr(obj, "name", None)
        if obj_name not in {"onion", "dish", "soup"}:
            continue
        rc = (int(pos_xy[1]), int(pos_xy[0]))
        counter_name = rc_to_name.get(rc)
        if counter_name is not None:
            counter_contents[counter_name] = str(obj_name).lower()
    return counter_contents


def _extract_pot_status(state) -> Tuple[str, int, bool]:
    pot_rc = dyn_utils.DESTINATION_TO_TILE["pot"]
    pot_xy = (int(pot_rc[1]), int(pot_rc[0]))
    obj = state.objects.get(pot_xy, None)
    if obj is None or getattr(obj, "name", None) != "soup":
        return "empty", 0, False
    ingredients = getattr(obj, "ingredients", []) or []
    onion_count = int(len(ingredients))
    is_idle = bool(getattr(obj, "is_idle", False))
    is_cooking = bool(getattr(obj, "is_cooking", False))
    is_ready = bool(getattr(obj, "is_ready", False))
    if is_ready:
        pot_state = "ready"
    elif is_cooking:
        pot_state = "cooking"
    elif onion_count <= 0:
        pot_state = "empty"
    elif onion_count == 1:
        pot_state = "one_onion"
    elif onion_count == 2:
        pot_state = "two_onions"
    else:
        pot_state = "three_onions"
    if is_idle and onion_count == 0:
        pot_state = "empty"
    return pot_state, onion_count, is_ready


def _build_policy_state_for_planner(state, agent_idx: int, reward_info: Dict) -> Dict:
    """Build the dyn_utils policy-state dict for shortest-path planning."""
    other_idx = 1 - int(agent_idx)
    obs_self = aif_env_utils.env_obs_to_model_obs(state, agent_idx, reward_info=reward_info)
    obs_other = aif_env_utils.env_obs_to_model_obs(state, other_idx, reward_info=reward_info)
    pot_state, pot_onions, ready = _extract_pot_status(state)
    counters = _extract_counter_contents(state)
    return dyn_utils.build_policy_state(
        self_pos=int(obs_self["self_pos_obs"]),
        self_orient=ORI_IDX_TO_NAME[int(obs_self["self_orientation_obs"])],
        self_held=HELD_IDX_TO_NAME[int(obs_self["self_held_obs"])],
        other_pos=int(obs_other["self_pos_obs"]),
        other_orient=ORI_IDX_TO_NAME[int(obs_other["self_orientation_obs"])],
        other_held=HELD_IDX_TO_NAME[int(obs_other["self_held_obs"])],
        pot_state=pot_state,
        pot_onions=pot_onions,
        soup_ready=bool(ready),
        counter_contents=counters,
    )


class AIFObsOvercookedMAEnv(MultiAgentEnv):
    """
    RLlib MultiAgentEnv: AIF-style discrete-factor obs + AIF-style semantic action space.

    Action space: Discrete(N_ACTIONS=20). Each index is a (destination, mode)
    pair in the order of dyn_utils.DESTINATIONS x dyn_utils.MODES. Each step:
      1. Replan from current state via dyn_utils.compile_semantic_policy (BFS).
      2. Execute ONLY the first primitive of the compiled path.
      3. Step the real Overcooked env once. No teleporting.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is not None and hasattr(config, "env_config"):
            kwargs.update(getattr(config, "env_config", {}))
        elif isinstance(config, dict):
            kwargs.update(config)

        layout = kwargs.get("layout", "cramped_room")
        horizon = int(kwargs.get("horizon", 400))
        self.fixed_reset_seed = kwargs.get("fixed_reset_seed")
        self.base = OvercookedMultiAgentEnv(config={"layout": layout, "horizon": horizon})
        self.reward_info = {"sparse_reward_by_agent": [0, 0]}
        self.state = None

        self.agents = list(self.base.agents)
        self.possible_agents = list(self.base.possible_agents)

        # Observation is [own_obs (OWN_OBS_DIM), other_obs (OWN_OBS_DIM)] --
        # "own" is always this agent's own egocentric vector (identical in kind
        # to what every AIF paradigm's ego agent sees), "other" is the partner's
        # own egocentric vector, in the same canonical slot regardless of which
        # physical agent this is (so a shared-parameter policy stays symmetric).
        # This lets the centralized critic (CentralizedCriticPPOTorchRLModule,
        # below) see the full joint state at training time while the actor
        # architecturally only ever reads the first OWN_OBS_DIM entries --
        # standard CTDE: privileged critic, decentralized execution. See
        # ai/02-debug.md section L ("this is IPPO, not MAPPO -- no centralized
        # critic exists anywhere").
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(2 * OWN_OBS_DIM,), dtype=np.float32)
        n_semantic = int(len(dyn_utils.DESTINATIONS) * len(dyn_utils.MODES))
        act_space = spaces.Discrete(n_semantic)
        self.n_semantic = n_semantic
        self.single_observation_space = obs_space
        self.single_action_space = act_space
        self.observation_space = spaces.Dict({aid: obs_space for aid in self.agents})
        self.action_space = spaces.Dict({aid: act_space for aid in self.agents})

    def _build_obs(self, state) -> Dict[str, np.ndarray]:
        own_vecs: Dict[str, np.ndarray] = {}
        for i, aid in enumerate(self.agents):
            obs_dict = aif_env_utils.env_obs_to_model_obs(
                state, i, reward_info=self.reward_info
            )
            own_vecs[aid] = _obs_dict_to_vec(obs_dict)
        out: Dict[str, np.ndarray] = {}
        for i, aid in enumerate(self.agents):
            other_aid = self.agents[1 - i]
            out[aid] = np.concatenate([own_vecs[aid], own_vecs[other_aid]]).astype(np.float32)
        return out

    def _build_dynamic_action_options(
        self, agent_idx: int
    ) -> Tuple[List[List[int]], List[Dict]]:
        """
        Compile all 20 semantic (destination, mode) options from the agent's
        current state into env-primitive action sequences (shortest-path BFS).
        """
        policy_state = _build_policy_state_for_planner(self.state, agent_idx, self.reward_info)
        policies_utils, metadata = dyn_utils.generate_policies_from_state(
            state=policy_state,
            destinations=None,
            modes=None,
            block_other_agent=True,
            deduplicate=False,
            pad=False,
            return_metadata=True,
        )
        policies_env: List[List[int]] = []
        for p in policies_utils:
            translated = [int(UTILS_TO_ENV_ACTION.get(int(a), aif_model_init.STAY)) for a in p]
            policies_env.append(translated if translated else [int(aif_model_init.STAY)])
        return policies_env, metadata

    def _semantic_to_primitive(
        self, agent_idx: int, option_idx: int
    ) -> Tuple[int, Dict]:
        """
        Decode the semantic option for this agent into the FIRST primitive of
        its shortest-path plan. No teleporting.
        """
        policies_env, metadata = self._build_dynamic_action_options(agent_idx)
        if not policies_env:
            return int(aif_model_init.STAY), {
                "option_idx": int(option_idx),
                "destination": "?",
                "mode": "?",
                "primitive": int(aif_model_init.STAY),
                "valid": False,
                "num_options": 0,
            }
        idx = int(option_idx)
        if idx < 0 or idx >= len(policies_env):
            idx = idx % len(policies_env)
        chosen_policy = policies_env[idx]
        chosen_meta = metadata[idx] if idx < len(metadata) else {}
        env_action = int(chosen_policy[0]) if chosen_policy else int(aif_model_init.STAY)
        return env_action, {
            "option_idx": int(idx),
            "destination": chosen_meta.get("destination", "?"),
            "mode": chosen_meta.get("mode", "?"),
            "primitive": env_action,
            "valid": bool(chosen_meta.get("valid", True)),
            "num_options": len(policies_env),
        }

    def reset(self, *, seed=None, options=None):
        if self.fixed_reset_seed is not None:
            seed = int(self.fixed_reset_seed)
        _, infos = self.base.reset(seed=seed, options=options)
        self.state = infos["agent_0"]["state"]
        self.reward_info = {"sparse_reward_by_agent": [0, 0]}
        obs = self._build_obs(self.state)
        out_infos = {aid: {"state": self.state} for aid in self.agents}
        return obs, out_infos

    def step(self, actions):
        primitive_actions: Dict[str, int] = {}
        semantic_meta: Dict[str, Dict] = {}
        for i, aid in enumerate(self.agents):
            prim, meta = self._semantic_to_primitive(i, int(actions[aid]))
            primitive_actions[aid] = prim
            semantic_meta[aid] = meta

        _, rewards, term, trunc, infos = self.base.step(primitive_actions)
        self.state = infos["agent_0"]["state"]
        self.reward_info = {
            "sparse_reward_by_agent": [
                infos["agent_0"].get("sparse_reward", 0),
                infos["agent_1"].get("sparse_reward", 0),
            ]
        }
        obs = self._build_obs(self.state)
        for aid in self.agents:
            infos[aid]["semantic"] = semantic_meta[aid]
        return obs, rewards, term, trunc, infos


if RAY_AVAILABLE:
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
    from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI

    def _mlp(in_dim: int, hidden: int, out_dim: int) -> "nn.Module":
        """Small orthogonal-initialized MLP encoder+head, matching the
        orthogonal-init / no-Catalog-magic best practice applied elsewhere in
        build_config() (ai/02-debug.md section L)."""
        layers = [
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        ]
        head = nn.Linear(hidden, out_dim)
        for layer in layers + [head]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
        return nn.Sequential(*layers, head)

    class CentralizedCriticPPOTorchRLModule(TorchRLModule, ValueFunctionAPI):
        """
        PPO RLModule implementing a genuine CTDE centralized critic.

        The actor sees ONLY the first `own_obs_dim` entries of the observation
        (this agent's own local, egocentric state -- identical in kind to what
        every AIF paradigm's ego agent observes). The critic sees the FULL
        observation (own + partner's own local state, i.e. the joint/global
        state), exactly the defining architectural feature that distinguishes
        MAPPO from independent PPO -- see ai/02-debug.md section L, which
        found this was entirely absent from the prior implementation (a shared
        actor-critic encoder reading only local obs for both branches, making
        it IPPO with parameter sharing, not MAPPO).

        Execution stays fully decentralized: at inference/rollout time nothing
        changes about what the *policy* conditions on (still local-only) --
        only the value function used during training gets the privileged joint
        view. This is standard CTDE, not a fairness violation against the
        ego-only AIF baselines: those are decentralized at both training and
        execution (they have no training phase at all), whereas RL baselines
        generally use CTDE precisely because "centralized critic, decentralized
        actor" is understood to not require execution-time communication.
        """

        def setup(self):
            own_dim = int(self.model_config.get("own_obs_dim", OWN_OBS_DIM))
            hidden = int(self.model_config.get("hidden_dim", 128))
            n_actions = int(self.action_space.n)
            self._own_dim = own_dim

            self.actor_encoder = _mlp(own_dim, hidden, hidden)
            self.pi = nn.Linear(hidden, n_actions)
            nn.init.orthogonal_(self.pi.weight, gain=0.01)
            nn.init.zeros_(self.pi.bias)

            joint_dim = int(self.observation_space.shape[0])
            self.critic_encoder = _mlp(joint_dim, hidden, hidden)
            self.vf = nn.Linear(hidden, 1)
            nn.init.orthogonal_(self.vf.weight)
            nn.init.zeros_(self.vf.bias)

        def get_initial_state(self) -> dict:
            return {}

        def _actor_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            obs = batch[Columns.OBS]
            own = obs[..., : self._own_dim]
            logits = self.pi(self.actor_encoder(own))
            return {Columns.ACTION_DIST_INPUTS: logits}

        @override(RLModule)
        def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            return self._actor_forward(batch)

        @override(RLModule)
        def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            return self._actor_forward(batch)

        @override(ValueFunctionAPI)
        def compute_values(self, batch: Dict[str, Any], embeddings=None):
            # Always recompute from the raw (full, joint) observation rather
            # than reusing any actor-side embedding -- correctness over the
            # shared-encoder speed shortcut, since actor and critic here
            # deliberately do NOT share an encoder (different input widths).
            obs = batch[Columns.OBS]
            vf_out = self.vf(self.critic_encoder(obs))
            return vf_out.squeeze(-1)


AGENT_IDS = ("agent_0", "agent_1")
PRIM_NAME = {0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST", 4: "STAY", 5: "INTERACT"}
ORI_NAME = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}


def _player_snapshot(state) -> list:
    out = []
    for i, p in enumerate(state.players[:2]):
        x, y = p.position
        ori = ORI_NAME.get(getattr(p, "orientation", (0, -1)), "?")
        held = "nothing"
        if p.has_object() and p.held_object is not None:
            held = str(getattr(p.held_object, "name", "obj"))
        out.append({"id": i, "x": int(x), "y": int(y), "ori": ori, "held": held})
    return out


def _pot_snapshot(state) -> str:
    parts = []
    for pos, obj in state.objects.items():
        if obj is None or getattr(obj, "name", None) != "soup":
            continue
        ing = getattr(obj, "ingredients", []) or []
        is_cooking = bool(getattr(obj, "is_cooking", False))
        is_ready = bool(getattr(obj, "is_ready", False))
        tag = "ready" if is_ready else ("cooking" if is_cooking else f"{len(ing)}onion")
        parts.append(f"{tuple(pos)}:{tag}")
    return "; ".join(parts) if parts else "(no pot soup)"


def _objects_snapshot(state) -> str:
    parts = []
    for pos, obj in state.objects.items():
        if obj is None:
            continue
        name = getattr(obj, "name", "obj")
        if name == "soup":
            continue
        parts.append(f"{tuple(pos)}:{name}")
    return "; ".join(parts) if parts else "(no counter items)"


def _format_state_line(state) -> str:
    players = _player_snapshot(state)
    pstr = " | ".join(
        f"A{p['id']}(xy=({p['x']},{p['y']}),ori={p['ori']},held={p['held']})"
        for p in players
    )
    return f"{pstr} | pot=[{_pot_snapshot(state)}] | objs=[{_objects_snapshot(state)}]"


def build_config(args):
    env_config = {"layout": args.layout, "horizon": args.horizon}
    episode_seed = getattr(args, "episode_seed", None)
    if episode_seed is not None:
        env_config["fixed_reset_seed"] = int(episode_seed)
    sample = AIFObsOvercookedMAEnv(env_config)
    obs_space = sample.single_observation_space
    act_space = sample.single_action_space

    if args.shared_policy:
        policies = {"shared": (None, obs_space, act_space, {})}
        policy_mapping_fn = lambda aid, episode, **kw: "shared"
    else:
        policies = {
            "agent_0": (None, obs_space, act_space, {}),
            "agent_1": (None, obs_space, act_space, {}),
        }
        policy_mapping_fn = lambda aid, episode, **kw: aid

    # Genuine CTDE centralized critic (see CentralizedCriticPPOTorchRLModule
    # above and ai/02-debug.md section L): actor conditions on local obs only
    # (own_obs_dim entries), critic conditions on the full joint observation
    # (own + partner's own state). This bypasses RLlib's default Catalog-built
    # shared-obs-space module entirely, so it also directly implements the
    # orthogonal-init / separate-encoders best practices from Yu et al. 2021
    # natively (no DefaultModelConfig needed -- those knobs only apply to the
    # Catalog-built default module this config no longer uses).
    rl_module_spec = RLModuleSpec(
        module_class=CentralizedCriticPPOTorchRLModule,
        model_config={"own_obs_dim": OWN_OBS_DIM, "hidden_dim": 128},
    )

    return (
        PPOConfig()
        .environment(env=AIFObsOvercookedMAEnv, env_config=env_config)
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_eps,
            entropy_coeff=args.ent_coef,
            vf_loss_coeff=args.vf_coef,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.epochs,
            grad_clip=10.0,
            grad_clip_by="global_norm",
        )
        .resources(num_gpus=int(args.gpus))
        .env_runners(
            num_env_runners=int(args.num_workers),
            num_envs_per_env_runner=int(args.envs_per_worker),
            num_cpus_per_env_runner=1,
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .rl_module(rl_module_spec=rl_module_spec)
        .debugging(seed=args.seed)
    )


def _get_total_env_steps(result) -> int:
    """Extract cumulative env steps sampled so far across RLlib API versions."""
    env_runners = result.get("env_runners", {}) or {}
    for key in (
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "num_env_steps_taken_lifetime",
    ):
        if key in env_runners:
            try:
                return int(env_runners[key])
            except Exception:
                pass
    for key in (
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "timesteps_total",
    ):
        if key in result:
            try:
                return int(result[key])
            except Exception:
                pass
    return 0


def _summarize_iter(result) -> Dict[str, object]:
    """Pull useful metrics out of an RLlib train() result for logging.

    RLlib's own `episode_return_mean`/`module_episode_returns_mean` (and,
    empirically confirmed, `episode_return_min`/`episode_return_max` the same
    way) sum every agent's reward for a multi-agent episode before averaging.
    Overcooked's reward is shared and identical for both agents every step, so
    this reports exactly 2x the true single-agent-equivalent team return --
    the same class of bug already found and fixed in the AIF run scripts'
    post-hoc `combined_mean` metric (ai/02-debug.md, "Stage-3 combined return
    was ~2x the real team return"), just baked into RLlib's own internal
    aggregation instead of user code this time. Verified directly: a minimal
    synthetic 2-agent env with reward=5.0/step/agent over a 4-step episode
    reports `episode_return_mean=40.0` while `agent_episode_returns_mean`
    correctly shows `{"agent_0": 20.0, "agent_1": 20.0}` -- the true value.
    This only affects this human-readable progress print, not the actual PPO
    loss/advantage computation (which operates on real per-step, per-agent
    rewards, not this summary metric), so it never corrupted training itself
    -- only made every printed training-progress number look 2x too high.
    """
    env_runners = result.get("env_runners", {}) or {}
    info: Dict[str, object] = {}
    info["episodes_completed"] = env_runners.get(
        "num_episodes_lifetime",
        env_runners.get("num_episodes", env_runners.get("episodes_total", 0)),
    )
    # Prefer the per-agent breakdown (correct, not doubled) over the summed
    # episode_return_mean/module_episode_returns_mean fields.
    agent_returns = env_runners.get("agent_episode_returns_mean")
    if agent_returns:
        info["return_mean"] = next(iter(agent_returns.values()), None)
    else:
        info["return_mean"] = None
    # episode_return_min/max have no per-agent breakdown available; correct
    # them by /2 (exact for this env, since both agents always receive the
    # identical shared reward every step -- not an approximation).
    raw_min = env_runners.get("episode_return_min")
    raw_max = env_runners.get("episode_return_max")
    info["return_min"] = raw_min / 2.0 if raw_min is not None else None
    info["return_max"] = raw_max / 2.0 if raw_max is not None else None
    info["ep_len_mean"] = env_runners.get("episode_len_mean")
    # Reward summed across all transitions sampled this iter (works even if no
    # episode completed yet). Also doubled by the same mechanism; halved for
    # consistency with return_mean above.
    for k in (
        "module_episode_returns_mean",
        "episode_reward_mean",
        "policy_reward_mean",
    ):
        if k in env_runners and info["return_mean"] is None:
            raw = env_runners.get(k)
            try:
                info["return_mean"] = next(iter(raw.values())) / 2.0 if isinstance(raw, dict) else raw / 2.0
            except (TypeError, StopIteration):
                info["return_mean"] = raw
            break
    # Sum of all per-step rewards sampled this iteration (most resilient signal).
    iter_reward = None
    for k in ("env_to_module_reward_sum", "rewards", "reward_total"):
        if k in env_runners:
            try:
                iter_reward = float(env_runners[k])
                break
            except Exception:
                pass
    info["iter_reward_sum"] = iter_reward
    return info


def train(args) -> str:
    cfg = build_config(args)
    out_dir = Path(args.checkpoint_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_label = getattr(args, "run_label", None)
    prefix = f"[{run_label}] " if run_label else ""
    algo = cfg.build_algo()
    max_steps = int(args.max_train_steps) if args.max_train_steps else 0
    save_every = int(getattr(args, "checkpoint_every", 0) or 0)
    last_saved_at = 0
    try:
        i = 0
        total_steps = 0
        while True:
            i += 1
            if int(args.iterations) > 0 and i > int(args.iterations):
                break
            result = algo.train()
            total_steps = _get_total_env_steps(result)
            summary = _summarize_iter(result)
            if i % max(1, int(args.log_every)) == 0:
                print(
                    f"{prefix}[iter {i:04d}] steps={total_steps} "
                    f"episodes_done={summary['episodes_completed']} "
                    f"return_mean={summary['return_mean']} "
                    f"min={summary['return_min']} max={summary['return_max']} "
                    f"ep_len_mean={summary['ep_len_mean']} "
                    f"iter_reward_sum={summary['iter_reward_sum']}"
                )
            if max_steps > 0 and total_steps >= max_steps:
                print(
                    f"{prefix}Reached max-train-steps cap "
                    f"({total_steps} >= {max_steps}); stopping training."
                )
                break
            if (
                save_every > 0
                and total_steps >= save_every
                and total_steps - last_saved_at >= save_every
            ):
                algo.save(str(out_dir))
                last_saved_at = total_steps
                print(f"{prefix}Intermediate checkpoint at {total_steps} env steps -> {out_dir}")
        save_obj = algo.save(str(out_dir))
        if hasattr(save_obj, "checkpoint") and hasattr(save_obj.checkpoint, "path"):
            ckpt = str(save_obj.checkpoint.path)
        else:
            ckpt = str(save_obj)
        print(f"{prefix}Saved checkpoint: {ckpt}")
        return ckpt
    finally:
        algo.stop()


def _select_action(module, obs_vec, stochastic: bool, generator=None):
    """
    Args:
        generator: optional torch.Generator used for reproducible stochastic
            sampling. Without this, evaluation-time action selection draws
            from PyTorch's global, unseeded RNG, so re-running the identical
            episode/agent seeds produces a different trajectory every time
            (see ai/02-debug.md section L). When provided, sampling is drawn
            from this generator instead and its state advances deterministically
            across calls, so passing the same seeded generator into every call
            for a given agent across an episode reproduces that agent's full
            action sequence exactly on re-run.
    """
    obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        fwd = module.forward_inference({"obs": obs_t})
    logits = None
    if "action_dist_inputs" in fwd:
        logits = fwd["action_dist_inputs"]
        if stochastic:
            if generator is not None:
                # torch.distributions.Categorical.sample() has no generator hook
                # (it calls probs.multinomial() internally with no way to pass
                # one through), so draw manually via torch.multinomial instead,
                # which does accept a generator.
                probs = torch.softmax(logits, dim=-1)
                act_t = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
            else:
                dist_cls = module.get_inference_action_dist_cls()
                dist = dist_cls.from_logits(logits)
                act_t = dist.sample() if hasattr(dist, "sample") else torch.argmax(logits, dim=-1)
        else:
            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(logits)
            if hasattr(dist, "deterministic_sample"):
                act_t = dist.deterministic_sample()
            else:
                act_t = torch.argmax(logits, dim=-1)
    elif "actions" in fwd:
        act_t = fwd["actions"]
    else:
        act_t = torch.tensor([4], dtype=torch.int64)
    if isinstance(act_t, torch.Tensor):
        act_int = int(act_t.reshape(-1)[0].item())
    else:
        act_int = int(act_t)
    return act_int, logits


def evaluate(args, checkpoint_path: str) -> None:
    algo = Algorithm.from_checkpoint(str(checkpoint_path))
    env = AIFObsOvercookedMAEnv({"layout": args.layout, "horizon": args.horizon})
    obs, infos = env.reset(seed=args.seed)
    pre_state = infos["agent_0"]["state"]
    totals = {aid: 0.0 for aid in AGENT_IDS}
    # Reproducible stochastic action sampling: without this, evaluation draws
    # from PyTorch's global, unseeded RNG, so identical --seed reruns produce
    # different trajectories every time (ai/02-debug.md section L). Each agent
    # gets its own generator, offset from --seed, matching AIF's own established
    # pattern of independent-but-derived per-agent RNG streams.
    agent_generators = {
        aid: torch.Generator().manual_seed(int(args.seed) + 1000 * i)
        for i, aid in enumerate(AGENT_IDS)
    }
    try:
        for step_idx in range(1, int(args.eval_steps) + 1):
            actions = {}
            top_lines = {aid: [] for aid in AGENT_IDS}
            for i_agent, aid in enumerate(AGENT_IDS):
                pid = "shared" if args.shared_policy else aid
                module = algo.get_module(pid)
                act_int, logits = _select_action(
                    module, obs[aid], args.stochastic, generator=agent_generators[aid]
                )
                actions[aid] = act_int
                if logits is not None and args.step_log:
                    probs = torch.softmax(logits, dim=-1).reshape(-1)
                    k = int(min(3, probs.shape[0]))
                    top_vals, top_idx = torch.topk(probs, k=k)
                    _, metadata = env._build_dynamic_action_options(i_agent)
                    for r in range(k):
                        idx = int(top_idx[r].item())
                        p = float(top_vals[r].item())
                        if metadata and idx < len(metadata):
                            m = metadata[idx]
                            dst = m.get("destination", "?")
                            mode = m.get("mode", "?")
                            top_lines[aid].append(
                                f"#{r + 1}: idx={idx:02d} p={p:.3f} -> {dst}:{mode}"
                            )
                        else:
                            top_lines[aid].append(f"#{r + 1}: idx={idx:02d} p={p:.3f}")

            next_obs, rewards, term, trunc, next_infos = env.step(actions)
            post_state = next_infos["agent_0"]["state"]
            for aid in AGENT_IDS:
                totals[aid] += float(rewards[aid])

            if args.step_log and ((step_idx - 1) % max(1, int(args.step_log_every)) == 0):
                print("")
                print("=" * 80)
                print(f"Step {step_idx:05d}")
                print("-" * 80)
                print(f"PRE  state: {_format_state_line(pre_state)}")
                for i, aid in enumerate(AGENT_IDS):
                    a = actions[aid]
                    r = float(rewards[aid])
                    sparse = next_infos[aid].get("sparse_reward", 0)
                    shaped = next_infos[aid].get("shaped_reward", 0)
                    sem = next_infos[aid].get("semantic", {})
                    dst = sem.get("destination", "?")
                    mode = sem.get("mode", "?")
                    prim = sem.get("primitive", -1)
                    obs_dict = aif_env_utils.env_obs_to_model_obs(
                        pre_state, i, reward_info=env.reward_info
                    )
                    obs_pairs = ", ".join(
                        f"{k.replace('_obs','')}={int(obs_dict[k])}" for k in OBS_KEYS
                    )
                    print(
                        f"{aid}: semantic={dst}:{mode} (idx={a:02d}) "
                        f"-> primitive={PRIM_NAME.get(prim, prim):<8} "
                        f"r={r:+.2f} sparse={sparse} shaped={shaped}"
                    )
                    print(f"  AIF obs: {obs_pairs}")
                    print(f"  obs vec: {np.round(obs[aid], 3)}")
                    for line in top_lines[aid]:
                        print(f"  {line}")
                print(f"POST state: {_format_state_line(post_state)}")
                pre_players = _player_snapshot(pre_state)
                post_players = _player_snapshot(post_state)
                for i in range(2):
                    pp, qp = pre_players[i], post_players[i]
                    moved = (pp["x"], pp["y"]) != (qp["x"], qp["y"])
                    rotated = pp["ori"] != qp["ori"]
                    held_changed = pp["held"] != qp["held"]
                    print(
                        f"  A{i} diff: moved={moved} "
                        f"rotated={rotated} ({pp['ori']}->{qp['ori']}) "
                        f"held_changed={held_changed} ({pp['held']}->{qp['held']})"
                    )
                print(
                    f"return so far: a0={totals['agent_0']:+.2f} "
                    f"a1={totals['agent_1']:+.2f}"
                )

            obs = next_obs
            pre_state = post_state
            if term.get("__all__") or trunc.get("__all__"):
                print(f"Episode finished at step {step_idx}")
                break
        print(
            f"Episode totals: a0={totals['agent_0']:+.2f} "
            f"a1={totals['agent_1']:+.2f}"
        )
    finally:
        algo.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Minimal MAPPO on Overcooked (Discrete(6) actions, sparse reward)"
    )
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--iterations", type=int, default=0,
                        help="Max PPO train() iterations (0 = unlimited; use --max-train-steps instead).")
    parser.add_argument("--max-train-steps", type=int, default=2000,
                        help="Stop after this many total env steps sampled (0 = no cap).")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument(
        "--ent-coef", type=float, default=0.05,
        help=(
            "Entropy bonus weight. A 2026-08-14 attempt to fix cramped_room's mutual "
            "collision deadlock (block_other_agent=True can freeze both agents "
            "permanently on this 6-tile layout) by raising this to 0.10 was tried and "
            "REVERTED: a real 500k-step training run at 0.10 produced a policy that "
            "stayed at ~99% of max entropy (statistically uniform-random) for the "
            "entire run and never converged to any task-directed behavior at all -- "
            "worse than the original problem, not better. Reward here is sparse enough "
            "(a soup needs ~8+ correctly-ordered macro picks out of 20 options, 0 "
            "reward otherwise) that a too-high entropy bonus can dominate the policy "
            "gradient indefinitely with no exploitation ever emerging. Left at the "
            "original 0.05 pending a real fix (candidates: decaying entropy schedule, "
            "or a stall-detector at the macro-planning level that doesn't touch overall "
            "policy stochasticity). See ai/02-debug.md, MAPPO baseline section."
        ),
    )
    parser.add_argument("--train-batch-size", type=int, default=1000)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)

    parser.add_argument("--shared-policy", action="store_true", default=True)
    parser.add_argument("--separate-policies", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/mappo_overcooked_simple",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-train", action="store_true")

    parser.add_argument("--run-episode", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--stochastic", action="store_true", default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--step-log", action="store_true")
    parser.add_argument("--step-log-every", type=int, default=1)
    args = parser.parse_args()

    if args.separate_policies:
        args.shared_policy = False
    if args.deterministic:
        args.stochastic = False

    if not RAY_AVAILABLE:
        raise RuntimeError(f"Ray RLlib not available: {IMPORT_ERR}")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=max(2, int(args.num_workers) + 1))

    try:
        checkpoint_path = args.checkpoint
        if not args.no_train:
            checkpoint_path = train(args)
        elif not checkpoint_path:
            raise ValueError("--checkpoint is required when using --no-train")
        elif not Path(checkpoint_path).is_absolute():
            checkpoint_path = str((PROJECT_ROOT / checkpoint_path).resolve())

        if args.run_episode and checkpoint_path:
            evaluate(args, checkpoint_path)
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
