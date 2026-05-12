"""
MAPPO-style training on RLlib PPO with AIF-aligned semantic interfaces.

Uses the same PPO library stack as:
`run_scripts_red_blue_doors/multi_agent/run_two_ppo_agents.py`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.IndependentActiveInferenceWithDynamicPolicies import utils as dyn_utils
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndependentWithSemanticPoliciesActionLevel import (
    env_utils,
    model_init,
)
from utils.visualization.overcooked_terminal_map import render_overcooked_grid

try:
    import ray
    import torch
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.multi_agent_env import MultiAgentEnv

    RAY_AVAILABLE = True
except ImportError as e:
    RAY_AVAILABLE = False
    IMPORT_ERR = str(e)
    MultiAgentEnv = object


AGENT_IDS = ("agent_0", "agent_1")
OBS_KEYS = [
    "self_pos_obs",
    "self_orientation_obs",
    "self_held_obs",
    "other_pos_obs",
    "other_held_obs",
    "pot_state_obs",
    "soup_delivered_obs",
] + [f"ctr_{idx}_obs" for idx in model_init.MODELED_COUNTERS]

OBS_NORM_DEN = np.array(
    [
        max(1, model_init.N_WALKABLE - 1),
        max(1, model_init.N_DIRECTIONS - 1),
        max(1, model_init.N_HELD_TYPES - 1),
        max(1, model_init.N_WALKABLE - 1),
        max(1, model_init.N_HELD_TYPES - 1),
        max(1, model_init.N_POT_STATES - 1),
        1,
    ] + [max(1, model_init.N_CTR_STATES - 1) for _ in model_init.MODELED_COUNTERS],
    dtype=np.float32,
)

# utils primitives: 0=N,1=S,2=W,3=E,4=STAY,5=INTERACT
# env/model primitives: 0=N,1=S,2=E,3=W,4=STAY,5=INTERACT
UTILS_TO_ENV_ACTION = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5}
ORI_NAME = {0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST"}
ORI_SHORT = {0: "N", 1: "S", 2: "E", 3: "W"}
HELD_NAME = {0: "nothing", 1: "onion", 2: "dish", 3: "soup"}


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


def _build_policy_state(state, agent_idx: int, reward_info: Dict) -> Dict:
    other_idx = 1 - int(agent_idx)
    obs_self = env_utils.env_obs_to_model_obs(state, agent_idx, reward_info=reward_info)
    obs_other = env_utils.env_obs_to_model_obs(state, other_idx, reward_info=reward_info)
    pot_state, pot_onions, ready = _extract_pot_status(state)
    counters = _extract_counter_contents(state)
    return dyn_utils.build_policy_state(
        self_pos=int(obs_self["self_pos_obs"]),
        self_orient=ORI_NAME[int(obs_self["self_orientation_obs"])],
        self_held=HELD_NAME[int(obs_self["self_held_obs"])],
        other_pos=int(obs_other["self_pos_obs"]),
        other_orient="NORTH",
        other_held=HELD_NAME[int(obs_other["self_held_obs"])],
        pot_state=pot_state,
        pot_onions=pot_onions,
        soup_ready=bool(ready),
        counter_contents=counters,
    )


def _obs_to_vec(obs_dict: Dict[str, int]) -> np.ndarray:
    vals = np.asarray([float(obs_dict[k]) for k in OBS_KEYS], dtype=np.float32)
    return vals / OBS_NORM_DEN


def _semantic_label(action_idx: int, semantic_meta: Dict) -> str:
    dst = semantic_meta.get("destination", "?")
    mode = semantic_meta.get("mode", "?")
    prim = semantic_meta.get("primitive", "?")
    return f"semantic #{action_idx:02d}: go `{dst}` and `{mode}` (executes primitive {prim})"


def _format_env_state_lines(state) -> List[str]:
    if state is None:
        return ["Environment snapshot: unavailable"]

    agent_parts: List[str] = []
    for i, p in enumerate(state.players[:2]):
        x, y = p.position
        grid = model_init.xy_to_index(x, y)
        w = model_init.grid_idx_to_walkable_idx(grid)
        held = "nothing"
        if p.has_object() and p.held_object:
            held = str(getattr(p.held_object, "name", "obj"))
        ori_idx = model_init.direction_to_index(p.orientation)
        ori = ORI_SHORT.get(int(ori_idx), str(ori_idx))
        wtxt = str(w) if w is not None else "NA"
        agent_parts.append(f"A{i}(w={wtxt},xy=({x},{y}),ori={ori},held={held})")

    pot_state, pot_onions, ready = _extract_pot_status(state)
    pot_label = "ready" if ready else f"{pot_state}/{pot_onions}onion"
    counters = _extract_counter_contents(state)
    counter_txt = ", ".join([f"{name}:{counters[name]}" for name in sorted(counters.keys())])
    map_lines = [f"Map: {row}" for row in render_overcooked_grid(state, model_init)]
    return [
        f"Environment: {' | '.join(agent_parts)}",
        f"Pot status: {pot_label}",
        f"Counter status: {counter_txt}",
    ] + map_lines


def _print_step_log(
    *,
    step_idx: int,
    episode_idx: int,
    actions: Dict[str, int],
    rewards: Dict[str, float],
    infos: Dict[str, Dict],
    top_policies: Dict[str, List[str]] | None,
    done: bool,
) -> None:
    s0 = infos["agent_0"].get("semantic", {})
    s1 = infos["agent_1"].get("semantic", {})
    a0_txt = _semantic_label(int(actions["agent_0"]), s0)
    a1_txt = _semantic_label(int(actions["agent_1"]), s1)
    env_lines = _format_env_state_lines(infos.get("agent_0", {}).get("state", None))
    done_txt = "yes" if done else "no"
    print("")
    print("=" * 72)
    print(f"Step {step_idx:07d} | episode {episode_idx:05d}")
    print("-" * 72)
    print(f"Agent 0 chose {a0_txt}. Reward: {float(rewards['agent_0']):+6.2f}")
    print(f"Agent 1 chose {a1_txt}. Reward: {float(rewards['agent_1']):+6.2f}")
    if top_policies is not None:
        for aid in AGENT_IDS:
            lines = top_policies.get(aid, [])
            if not lines:
                continue
            print(f"Top 3 options for {aid}:")
            for line in lines:
                print(f"  {line}")
    for line in env_lines:
        print(line)
    print(f"Episode finished on this step: {done_txt}")
    print("=" * 72)


class SemanticAIFObsOvercookedRLlibEnv(MultiAgentEnv):
    """RLlib MultiAgentEnv with AIF-matched observations and semantic actions."""

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is not None and hasattr(config, "env_config"):
            kwargs.update(getattr(config, "env_config", {}))
        elif isinstance(config, dict):
            kwargs.update(config)

        layout = kwargs.get("layout", "cramped_room")
        horizon = int(kwargs.get("horizon", 400))
        self.base_env = OvercookedMultiAgentEnv(config={"layout": layout, "horizon": horizon})
        self.state = None
        self.reward_info = {"sparse_reward_by_agent": [0, 0]}

        self.agents = list(AGENT_IDS)
        self.possible_agents = list(AGENT_IDS)
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(len(OBS_KEYS),), dtype=np.float32)
        act_space = spaces.Discrete(int(model_init.N_ACTIONS))
        self.single_observation_space = obs_space
        self.single_action_space = act_space
        self.observation_space = spaces.Dict({aid: obs_space for aid in AGENT_IDS})
        self.action_space = spaces.Dict({aid: act_space for aid in AGENT_IDS})

    def _build_obs(self) -> Dict[str, np.ndarray]:
        out = {}
        for i, aid in enumerate(AGENT_IDS):
            obs_dict = env_utils.env_obs_to_model_obs(self.state, i, reward_info=self.reward_info)
            out[aid] = _obs_to_vec(obs_dict)
        return out

    def _build_dynamic_action_options(self, agent_idx: int) -> Tuple[List[List[int]], List[Dict]]:
        """
        Build the same dynamic per-step policy list AIF uses for infer_policies().

        Returns:
            policies_env: list of primitive policies translated to env primitive indexing
            metadata: list of policy metadata in the same order as policies_env
        """
        policy_state = _build_policy_state(self.state, agent_idx, self.reward_info)
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
            translated = [int(UTILS_TO_ENV_ACTION.get(int(a), model_init.STAY)) for a in p]
            policies_env.append(translated if translated else [int(model_init.STAY)])
        return policies_env, metadata

    def _semantic_to_primitive(self, agent_idx: int, option_idx: int) -> Tuple[int, Dict]:
        policies_env, metadata = self._build_dynamic_action_options(agent_idx)
        if not policies_env:
            return int(model_init.STAY), {
                "option_idx": int(option_idx),
                "destination": "?",
                "mode": "?",
                "primitive": int(model_init.STAY),
                "valid": False,
                "num_options": 0,
            }

        idx = int(option_idx)
        if idx < 0 or idx >= len(policies_env):
            idx = idx % len(policies_env)

        chosen_policy = policies_env[idx]
        chosen_meta = metadata[idx] if idx < len(metadata) else {}
        env_action = int(chosen_policy[0]) if chosen_policy else int(model_init.STAY)
        return env_action, {
            "option_idx": int(idx),
            "semantic_idx": int(idx),
            "destination": chosen_meta.get("destination", "?"),
            "mode": chosen_meta.get("mode", "?"),
            "primitive": env_action,
            "valid": bool(chosen_meta.get("valid", True)),
            "num_options": len(policies_env),
        }

    def reset(self, *, seed=None, options=None):
        del options
        _, infos = self.base_env.reset(seed=seed)
        self.state = infos["agent_0"]["state"]
        self.reward_info = {"sparse_reward_by_agent": [0, 0]}
        obs = self._build_obs()
        infos = {aid: {"state": self.state} for aid in AGENT_IDS}
        return obs, infos

    def step(self, actions):
        primitive_actions: Dict[str, int] = {}
        semantic_meta: Dict[str, Dict] = {}
        for i, aid in enumerate(AGENT_IDS):
            prim, meta = self._semantic_to_primitive(i, int(actions[aid]))
            primitive_actions[aid] = prim
            semantic_meta[aid] = meta

        _, rewards, terminated, truncated, infos = self.base_env.step(primitive_actions)
        self.state = infos["agent_0"]["state"]
        self.reward_info = {
            "sparse_reward_by_agent": [
                infos["agent_0"].get("sparse_reward", 0),
                infos["agent_1"].get("sparse_reward", 0),
            ]
        }
        obs = self._build_obs()
        for aid in AGENT_IDS:
            infos[aid]["semantic"] = semantic_meta[aid]
        return obs, rewards, terminated, truncated, infos


def build_rllib_config(args):
    env_config = {"layout": args.layout, "horizon": int(args.horizon)}
    env_instance = SemanticAIFObsOvercookedRLlibEnv(env_config)

    if args.shared_policy:
        policies = {
            "shared_policy": (
                None,
                env_instance.single_observation_space,
                env_instance.single_action_space,
                {},
            )
        }
        policy_mapping_fn = lambda agent_id, episode, **kwargs: "shared_policy"
    else:
        policies = {
            "agent_0": (
                None,
                env_instance.single_observation_space,
                env_instance.single_action_space,
                {},
            ),
            "agent_1": (
                None,
                env_instance.single_observation_space,
                env_instance.single_action_space,
                {},
            ),
        }
        policy_mapping_fn = lambda agent_id, episode, **kwargs: agent_id

    cfg = (
        PPOConfig()
        .environment(env=SemanticAIFObsOvercookedRLlibEnv, env_config=env_config)
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
        )
        .resources(num_gpus=int(args.gpus))
        .env_runners(
            num_env_runners=int(args.num_workers),
            num_envs_per_env_runner=int(args.num_envs_per_worker),
            num_cpus_per_env_runner=1,
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .debugging(seed=args.seed)
    )
    return cfg


def train_with_rllib(args) -> str:
    cfg = build_rllib_config(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (PROJECT_ROOT / checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    algo = cfg.build_algo()
    try:
        for i in range(1, int(args.iterations) + 1):
            result = algo.train()
            if i % max(1, int(args.log_every)) == 0:
                rew = result.get("episode_reward_mean", None)
                print(f"[iter {i:04d}] episode_reward_mean={rew}")
        save_obj = algo.save(str(checkpoint_dir))
        if hasattr(save_obj, "checkpoint") and hasattr(save_obj.checkpoint, "path"):
            ckpt = str(save_obj.checkpoint.path)
        else:
            ckpt = str(save_obj)
        print(f"Saved checkpoint: {ckpt}")
        return ckpt
    finally:
        algo.stop()


def run_logged_episode(args, checkpoint_path: str) -> None:
    algo = Algorithm.from_checkpoint(str(checkpoint_path))
    env = SemanticAIFObsOvercookedRLlibEnv({"layout": args.layout, "horizon": args.horizon})
    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_idx = 0
    try:
        for step_idx in range(1, int(args.episode_steps) + 1):
            actions = {}
            top_policies: Dict[str, List[str]] = {}
            for aid in AGENT_IDS:
                policy_id = "shared_policy" if args.shared_policy else aid
                try:
                    module = algo.get_module(policy_id)
                    obs_tensor = torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        out = module.forward_inference({"obs": obs_tensor})
                    logits = None
                    if "action_dist_inputs" in out:
                        logits = out["action_dist_inputs"]
                        dist_cls = module.get_inference_action_dist_cls()
                        dist = dist_cls.from_logits(logits)
                        if hasattr(dist, "deterministic_sample"):
                            act = dist.deterministic_sample()
                        else:
                            act = torch.argmax(logits, dim=-1)
                    elif "actions" in out:
                        act = out["actions"]
                    elif "action" in out:
                        act = out["action"]
                    else:
                        act = torch.tensor([model_init.STAY], dtype=torch.int64)
                    if isinstance(act, torch.Tensor):
                        act = int(act.reshape(-1)[0].item())
                    else:
                        act = int(act)
                    actions[aid] = act

                    # Human-readable top-3 policy options from PPO distribution.
                    if logits is not None:
                        probs = torch.softmax(logits, dim=-1).reshape(-1)
                        k = int(min(3, probs.shape[0]))
                        top_vals, top_idx = torch.topk(probs, k=k)
                        agent_idx = 0 if aid == "agent_0" else 1
                        _pols, metadata = env._build_dynamic_action_options(agent_idx)
                        lines = []
                        for rank in range(k):
                            idx = int(top_idx[rank].item())
                            p = float(top_vals[rank].item())
                            if metadata:
                                m = metadata[idx % len(metadata)]
                                dst = m.get("destination", "?")
                                mode = m.get("mode", "?")
                                lines.append(f"#{rank + 1}: idx={idx:02d} p={p:.3f} -> {dst}:{mode}")
                            else:
                                lines.append(f"#{rank + 1}: idx={idx:02d} p={p:.3f}")
                        top_policies[aid] = lines
                except Exception:
                    actions[aid] = int(model_init.STAY)
                    top_policies[aid] = ["#1: <unavailable>", "#2: <unavailable>", "#3: <unavailable>"]

            obs, rewards, terminated, truncated, infos = env.step(actions)
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            episode_return += float(rewards["agent_0"])
            if args.step_log and ((step_idx - 1) % max(1, int(args.step_log_every)) == 0):
                _print_step_log(
                    step_idx=step_idx,
                    episode_idx=episode_idx,
                    actions=actions,
                    rewards=rewards,
                    infos=infos,
                    top_policies=top_policies,
                    done=done,
                )
            if done:
                break
        print(f"Episode return: {episode_return:+.2f}")
    finally:
        algo.stop()


def main():
    parser = argparse.ArgumentParser(description="RLlib PPO MAPPO-style runner with AIF semantic obs/actions")
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=2000)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--shared-policy", action="store_true", default=True)
    parser.add_argument("--separate-policies", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mappo_semantic_overcooked_rllib")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-train", action="store_true")

    parser.add_argument("--run-episode", action="store_true")
    parser.add_argument("--episode-steps", type=int, default=5000)
    parser.add_argument("--step-log", action="store_true")
    parser.add_argument("--step-log-every", type=int, default=1)
    args = parser.parse_args()

    if args.separate_policies:
        args.shared_policy = False

    if not RAY_AVAILABLE:
        raise RuntimeError(f"Ray RLlib not available: {IMPORT_ERR}")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=max(2, int(args.num_workers) + 1))

    checkpoint_path = args.checkpoint
    try:
        if not args.no_train:
            checkpoint_path = train_with_rllib(args)
        elif not checkpoint_path:
            raise ValueError("--checkpoint is required when using --no-train")
        elif not Path(checkpoint_path).is_absolute():
            checkpoint_path = str((PROJECT_ROOT / checkpoint_path).resolve())

        if args.run_episode:
            run_logged_episode(args, checkpoint_path=checkpoint_path)
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
