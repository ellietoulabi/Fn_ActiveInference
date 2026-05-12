"""
Minimal MAPPO on Overcooked with primitive Discrete(6) actions.

No semantic policies. No reward shaping. PPO sees:
  - obs: AIF-style discrete-factor observation flattened to a vector
         (self_pos, self_ori, self_held, other_pos, other_held,
          pot_state, soup_delivered, [counter contents...])
         Same factors used by the AIF agent (Independent paradigm).
  - act: Discrete(6) primitive actions (N, S, E, W, STAY, INTERACT)
  - reward: env's sparse delivery reward (per agent, shared)

Uses Ray RLlib PPO (same library stack as
run_scripts_red_blue_doors/multi_agent/run_two_ppo_agents.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    from ray.rllib.env.multi_agent_env import MultiAgentEnv

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
    "other_held_obs",
    "pot_state_obs",
    "soup_delivered_obs",
] + [f"ctr_{idx}_obs" for idx in aif_model_init.MODELED_COUNTERS]

OBS_NORM_DEN = np.array(
    [
        max(1, aif_model_init.N_WALKABLE - 1),
        max(1, aif_model_init.N_DIRECTIONS - 1),
        max(1, aif_model_init.N_HELD_TYPES - 1),
        max(1, aif_model_init.N_WALKABLE - 1),
        max(1, aif_model_init.N_HELD_TYPES - 1),
        max(1, aif_model_init.N_POT_STATES - 1),
        1,
    ] + [max(1, aif_model_init.N_CTR_STATES - 1) for _ in aif_model_init.MODELED_COUNTERS],
    dtype=np.float32,
)


def _obs_dict_to_vec(obs_dict: Dict[str, int]) -> np.ndarray:
    vals = np.asarray([float(obs_dict[k]) for k in OBS_KEYS], dtype=np.float32)
    return vals / OBS_NORM_DEN


class AIFObsOvercookedMAEnv(MultiAgentEnv):
    """
    RLlib MultiAgentEnv that wraps OvercookedMultiAgentEnv but exposes the same
    discrete-factor observation the AIF Independent agent receives.

    - Action space: Discrete(6) primitives (no semantic policies).
    - Reward: sparse delivery reward from the base env (no shaping).
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is not None and hasattr(config, "env_config"):
            kwargs.update(getattr(config, "env_config", {}))
        elif isinstance(config, dict):
            kwargs.update(config)

        layout = kwargs.get("layout", "cramped_room")
        horizon = int(kwargs.get("horizon", 400))
        self.base = OvercookedMultiAgentEnv(config={"layout": layout, "horizon": horizon})
        self.reward_info = {"sparse_reward_by_agent": [0, 0]}

        self.agents = list(self.base.agents)
        self.possible_agents = list(self.base.possible_agents)

        obs_space = spaces.Box(low=0.0, high=1.0, shape=(len(OBS_KEYS),), dtype=np.float32)
        act_space = spaces.Discrete(6)
        self.single_observation_space = obs_space
        self.single_action_space = act_space
        self.observation_space = spaces.Dict({aid: obs_space for aid in self.agents})
        self.action_space = spaces.Dict({aid: act_space for aid in self.agents})

    def _build_obs(self, state) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for i, aid in enumerate(self.agents):
            obs_dict = aif_env_utils.env_obs_to_model_obs(
                state, i, reward_info=self.reward_info
            )
            out[aid] = _obs_dict_to_vec(obs_dict)
        return out

    def reset(self, *, seed=None, options=None):
        _, infos = self.base.reset(seed=seed, options=options)
        state = infos["agent_0"]["state"]
        self.reward_info = {"sparse_reward_by_agent": [0, 0]}
        obs = self._build_obs(state)
        out_infos = {aid: {"state": state} for aid in self.agents}
        return obs, out_infos

    def step(self, actions):
        _, rewards, term, trunc, infos = self.base.step(actions)
        state = infos["agent_0"]["state"]
        self.reward_info = {
            "sparse_reward_by_agent": [
                infos["agent_0"].get("sparse_reward", 0),
                infos["agent_1"].get("sparse_reward", 0),
            ]
        }
        obs = self._build_obs(state)
        return obs, rewards, term, trunc, infos


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
        )
        .resources(num_gpus=int(args.gpus))
        .env_runners(
            num_env_runners=int(args.num_workers),
            num_envs_per_env_runner=int(args.envs_per_worker),
            num_cpus_per_env_runner=1,
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
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


def train(args) -> str:
    cfg = build_config(args)
    out_dir = Path(args.checkpoint_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    algo = cfg.build_algo()
    max_steps = int(args.max_train_steps) if args.max_train_steps else 0
    try:
        i = 0
        total_steps = 0
        while True:
            i += 1
            if int(args.iterations) > 0 and i > int(args.iterations):
                break
            result = algo.train()
            total_steps = _get_total_env_steps(result)
            env_runners = result.get("env_runners", {}) or {}
            ret_mean = env_runners.get("episode_return_mean")
            ret_min = env_runners.get("episode_return_min")
            ret_max = env_runners.get("episode_return_max")
            if i % max(1, int(args.log_every)) == 0:
                print(
                    f"[iter {i:04d}] steps={total_steps} return_mean={ret_mean} "
                    f"min={ret_min} max={ret_max}"
                )
            if max_steps > 0 and total_steps >= max_steps:
                print(
                    f"Reached max-train-steps cap "
                    f"({total_steps} >= {max_steps}); stopping training."
                )
                break
        save_obj = algo.save(str(out_dir))
        if hasattr(save_obj, "checkpoint") and hasattr(save_obj.checkpoint, "path"):
            ckpt = str(save_obj.checkpoint.path)
        else:
            ckpt = str(save_obj)
        print(f"Saved checkpoint: {ckpt}")
        return ckpt
    finally:
        algo.stop()


def _select_action(module, obs_vec, stochastic: bool):
    obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        fwd = module.forward_inference({"obs": obs_t})
    logits = None
    if "action_dist_inputs" in fwd:
        logits = fwd["action_dist_inputs"]
        dist_cls = module.get_inference_action_dist_cls()
        dist = dist_cls.from_logits(logits)
        if stochastic and hasattr(dist, "sample"):
            act_t = dist.sample()
        elif hasattr(dist, "deterministic_sample"):
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
    try:
        for step_idx in range(1, int(args.eval_steps) + 1):
            actions = {}
            top_lines = {aid: [] for aid in AGENT_IDS}
            for aid in AGENT_IDS:
                pid = "shared" if args.shared_policy else aid
                module = algo.get_module(pid)
                act_int, logits = _select_action(module, obs[aid], args.stochastic)
                actions[aid] = act_int
                if logits is not None and args.step_log:
                    probs = torch.softmax(logits, dim=-1).reshape(-1)
                    k = int(min(3, probs.shape[0]))
                    top_vals, top_idx = torch.topk(probs, k=k)
                    for r in range(k):
                        idx = int(top_idx[r].item())
                        p = float(top_vals[r].item())
                        top_lines[aid].append(
                            f"#{r + 1}: {PRIM_NAME.get(idx, idx):<8} p={p:.3f}"
                        )

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
                    obs_dict = aif_env_utils.env_obs_to_model_obs(
                        pre_state, i, reward_info=env.reward_info
                    )
                    obs_pairs = ", ".join(
                        f"{k.replace('_obs','')}={int(obs_dict[k])}" for k in OBS_KEYS
                    )
                    print(
                        f"{aid}: action={PRIM_NAME.get(a, a):<8} "
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
    parser.add_argument("--ent-coef", type=float, default=0.05)
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
