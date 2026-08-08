#!/usr/bin/env python3
"""
Sample N random Overcooked states (cramped_room) and print the dynamic semantic
policy library for both agents at each state — same pipeline as
run_individually_collective_policy_semantic_action_level.py (build_policy_state,
update_policies, translate, wrap).

Usage:
  python show_random_states_policies_semantic_action_level.py
  python show_random_states_policies_semantic_action_level.py --n 5 --master-seed 42
  python show_random_states_policies_semantic_action_level.py --warmup-steps 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

overcooked_src = PROJECT_ROOT / "environments" / "overcooked_ai" / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))

import run_individually_collective_policy_semantic_action_level as ric

from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollectiveWithSemanticPoliciesActionLevel.model_init import (
    PRIMITIVE_POLICY_STEP,
)


def _sample_state(env, rng: np.random.Generator, warmup_steps: int, horizon: int):
    """reset with random seed, optional random joint primitives, return state."""
    ep_seed = int(rng.integers(0, 2**31 - 1))
    env.reset(seed=ep_seed)
    state = None
    _obs, infos = None, None
    for _ in range(max(0, int(warmup_steps))):
        a0 = int(rng.integers(0, ric.N_PRIMITIVE_ACTIONS))
        a1 = int(rng.integers(0, ric.N_PRIMITIVE_ACTIONS))
        _obs, _r, terminated, truncated, infos = env.step({"agent_0": a0, "agent_1": a1})
        state = infos["agent_0"]["state"]
        if terminated.get("__all__") or truncated.get("__all__"):
            _obs, infos = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            state = infos["agent_0"]["state"]
    if state is None:
        _obs, infos = env.reset(seed=ep_seed)
        state = infos["agent_0"]["state"]
    return state, ep_seed


def _print_policies_for_agent(tag: str, agent, model_init):
    n = len(agent.policies or [])
    meta = agent.get_policy_metadata() or []
    print("    {}: {} policies".format(tag, n), flush=True)
    for j in range(n):
        pol = agent.policies[j]
        prim = ric._fmt_policy(pol)
        if j < len(meta) and meta[j]:
            m = meta[j]
            sem = "destination={!r} mode={!r}".format(m.get("destination"), m.get("mode"))
        else:
            sem = "(no metadata)"
        print("      [{:3d}] {}  |  {}".format(j, sem, prim), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of random states to show.")
    parser.add_argument("--master-seed", type=int, default=None, help="RNG seed for sampling.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Random joint primitives after each reset.")
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--noig", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.master_seed)

    from agents.ActiveInferenceFixedPolicies.agent import Agent
    from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollectiveWithSemanticPoliciesActionLevel import (
        A_fn,
        B_fn,
        C_fn,
        D_fn,
        model_init as mon_model_init,
        env_utils,
    )
    from environments.overcooked_ma_gym import OvercookedMultiAgentEnv

    model_init_agent = mon_model_init
    if int(getattr(model_init_agent, "N_PRIMITIVE_ACTIONS", ric.N_PRIMITIVE_ACTIONS)) != ric.N_PRIMITIVE_ACTIONS:
        print("N_PRIMITIVE_ACTIONS mismatch; abort.")
        sys.exit(1)

    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    base_env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}
    policy_len = ric.PAIR_POLICY_HORIZON
    max_steps = 50
    horizon = max_steps + 10

    def make_agent(ego: int, seed: int):
        np.random.seed(seed)
        return Agent(
            A_fn=A_fn,
            B_fn=B_fn,
            C_fn=C_fn,
            D_fn=D_fn,
            state_factors=state_factors,
            state_sizes=state_sizes,
            observation_labels=observation_labels,
            env_params={**base_env_params, "ego_agent_index": int(ego)},
            observation_state_dependencies=model_init_agent.observation_state_dependencies,
            actions=list(range(ric.N_PRIMITIVE_ACTIONS)),
            gamma=float(args.gamma),
            alpha=float(args.alpha),
            policy_len=policy_len,
            inference_horizon=policy_len,
            action_selection="stochastic",
            sampling_mode="full",
            inference_algorithm="VANILLA",
            num_iter=16,
            dF_tol=0.01,
            use_action_for_state_inference=True,
        )

    env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": horizon})
    agent_0 = make_agent(0, int(rng.integers(0, 10_000)))
    agent_1 = make_agent(1, int(rng.integers(0, 10_000)))
    if args.noig:
        agent_0.use_states_info_gain = False
        agent_1.use_states_info_gain = False

    prev_zero = {"sparse_reward_by_agent": [0, 0]}

    for k in range(int(args.n)):
        state, ep_seed = _sample_state(env, rng, args.warmup_steps, horizon)

        print("\n" + "=" * 72, flush=True)
        print("  Sample {:d}/{:d}  (reset/warmup seed component ~ {})".format(k + 1, args.n, ep_seed), flush=True)
        print("  " + ric._state_summary(state, model_init_agent, max_agents=2), flush=True)
        print("    Map:", flush=True)
        for row in ric.render_overcooked_grid(state, model_init_agent):
            print("      " + row, flush=True)

        for i, ag in enumerate((agent_0, agent_1)):
            cfg = env_utils.get_D_config_from_state(state, i)
            ag.reset(config=cfg)

        ps0 = ric.build_policy_state_for_agent(state, 0, env_utils, prev_reward_info=prev_zero)
        ps1 = ric.build_policy_state_for_agent(state, 1, env_utils, prev_reward_info=prev_zero)

        agent_0.update_policies(ps0)
        agent_1.update_policies(ps1)
        ric._translate_agent_policies_utils_to_env(agent_0)
        ric._translate_agent_policies_utils_to_env(agent_1)
        ric._wrap_policies_for_primitive_B_rollout(agent_0)
        ric._wrap_policies_for_primitive_B_rollout(agent_1)

        _print_policies_for_agent("Agent 0 (ego=0)", agent_0, model_init_agent)
        _print_policies_for_agent("Agent 1 (ego=1)", agent_1, model_init_agent)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()