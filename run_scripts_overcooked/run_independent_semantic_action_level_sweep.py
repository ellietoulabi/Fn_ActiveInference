"""
Run the same semantic-action-level Overcooked scenario multiple times with different
RNG seeds. Does not modify run_individually_collective_policy_semantic_action_level.py.

Default sweep: gamma=4.0, alpha=8.0 (overridable with --gamma / --alpha).

Use --log-steps for the same style of per-step console output as the main runner
(run_individually_collective_policy_semantic_action_level.py verbose mode).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))  # make sibling modules importable
overcooked_src = PROJECT_ROOT / "environments" / "overcooked_ai" / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))

import run_independent_semantic_action_level as ind

from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndependentWithSemanticPoliciesActionLevel.model_init import (
    PRIMITIVE_POLICY_STEP,
)


def _run_sweep(
    *,
    n_runs: int,
    episode_seeds: list[int],
    agent0_seeds: list[int],
    agent1_seeds: list[int],
    gamma: float,
    alpha: float,
    no_ig: bool,
    verbose: bool,
    log_steps: bool,
) -> None:
    if len(episode_seeds) != n_runs or len(agent0_seeds) != n_runs or len(agent1_seeds) != n_runs:
        raise ValueError("episode_seeds, agent0_seeds, agent1_seeds must each have length n_runs")

    try:
        from agents.IndependentActiveInferenceWithDynamicPolicies.agent import Agent
        from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndependentWithSemanticPoliciesActionLevel import (
            A_fn,
            B_fn,
            C_fn,
            D_fn,
            model_init as mon_model_init,
            env_utils,
        )
        from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
    except Exception as e:
        print("[SKIP] Could not load agent or environment: {}".format(e))
        return

    model_init_agent = mon_model_init
    if int(getattr(model_init_agent, "N_PRIMITIVE_ACTIONS", ind.N_PRIMITIVE_ACTIONS)) != ind.N_PRIMITIVE_ACTIONS:
        print(
            "[SKIP] model_init N_PRIMITIVE_ACTIONS ({}) != ind.N_PRIMITIVE_ACTIONS ({}).".format(
                getattr(model_init_agent, "N_PRIMITIVE_ACTIONS", None),
                ind.N_PRIMITIVE_ACTIONS,
            )
        )
        return

    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    base_env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}
    policy_len = ind.PAIR_POLICY_HORIZON
    max_steps_per_scenario = 2000
    horizon = max_steps_per_scenario + 10
    env_layout = "cramped_room"

    def create_agent(seed=None, ego_agent_index: int = 0):
        if seed is not None:
            np.random.seed(seed)
        env_params = {**base_env_params, "ego_agent_index": int(ego_agent_index)}
        agent = Agent(
            A_fn=A_fn,
            B_fn=B_fn,
            C_fn=C_fn,
            D_fn=D_fn,
            state_factors=state_factors,
            state_sizes=state_sizes,
            observation_labels=observation_labels,
            env_params=env_params,
            observation_state_dependencies=model_init_agent.observation_state_dependencies,
            actions=list(range(ind.N_PRIMITIVE_ACTIONS)),
            gamma=float(gamma),
            alpha=float(alpha),
            policy_len=policy_len,
            inference_horizon=policy_len,
            action_selection="stochastic",
            sampling_mode="full",
            inference_algorithm="VANILLA",
            num_iter=16,
            dF_tol=0.01,
            use_action_for_state_inference=True,
        )
        if no_ig:
            agent.use_states_info_gain = False
        return agent

    def _execute_first_primitive_step(env_obj, a0_prim: int, a1_prim: int):
        observations, rewards, terminated, truncated, infos = env_obj.step(
            {"agent_0": int(a0_prim), "agent_1": int(a1_prim)}
        )
        next_state = infos["agent_0"]["state"]
        reward_info = {
            "sparse_reward_by_agent": [
                infos["agent_0"].get("sparse_reward", 0),
                infos["agent_1"].get("sparse_reward", 0),
            ]
        }
        return observations, next_state, reward_info, rewards, terminated, truncated, infos

    def run_one_episode(
        env_obj,
        agent_0,
        agent_1,
        episode_name: str,
        episode_seed: int | None,
        *,
        log_steps: bool,
        run_tag: str,
    ):
        _obs, infos = env_obj.reset(seed=episode_seed)
        state = infos["agent_0"]["state"]

        config_0 = env_utils.get_D_config_from_state(state, 0)
        config_1 = env_utils.get_D_config_from_state(state, 1)
        agent_0.reset(config=config_0)
        agent_1.reset(config=config_1)

        prev_reward_info = {"sparse_reward_by_agent": [0, 0]}
        total_reward_0 = 0.0
        total_reward_1 = 0.0

        if verbose or log_steps:
            print("\n" + "=" * 72, flush=True)
            print("  {}".format(episode_name), flush=True)
            print("=" * 72, flush=True)

        for step in range(1, max_steps_per_scenario + 1):
            obs_0 = env_utils.env_obs_to_model_obs(state, 0, reward_info=prev_reward_info)
            obs_1 = env_utils.env_obs_to_model_obs(state, 1, reward_info=prev_reward_info)

            if log_steps:
                state_str = ind._state_summary(state, model_init_agent, max_agents=2)
                print("\n  --- [{}] Step {} ---".format(run_tag, step), flush=True)
                print("    Env state:  {}".format(state_str), flush=True)
                print("    Map (before action):", flush=True)
                for row in ind.render_overcooked_grid(state, model_init_agent):
                    print("      " + row, flush=True)
                for line in ind._agent_summary_lines(state, model_init_agent, max_agents=2):
                    print(line, flush=True)
                print(
                    "    Obs A0: self_pos={} self_ori={} self_held={} other_pos={} other_held={} pot={} delivered={}".format(
                        obs_0["self_pos_obs"],
                        obs_0["self_orientation_obs"],
                        obs_0["self_held_obs"],
                        obs_0["other_pos_obs"],
                        obs_0["other_held_obs"],
                        obs_0["pot_state_obs"],
                        obs_0["soup_delivered_obs"],
                    ),
                    flush=True,
                )
                print(
                    "    Obs A1: self_pos={} self_ori={} self_held={} other_pos={} other_held={} pot={} delivered={}".format(
                        obs_1["self_pos_obs"],
                        obs_1["self_orientation_obs"],
                        obs_1["self_held_obs"],
                        obs_1["other_pos_obs"],
                        obs_1["other_held_obs"],
                        obs_1["pot_state_obs"],
                        obs_1["soup_delivered_obs"],
                    ),
                    flush=True,
                )

            agent_0.infer_states(obs_0)
            agent_1.infer_states(obs_1)

            policy_state_0 = ind.build_policy_state_for_agent(
                state, agent_idx=0, env_utils=env_utils, prev_reward_info=prev_reward_info
            )
            policy_state_1 = ind.build_policy_state_for_agent(
                state, agent_idx=1, env_utils=env_utils, prev_reward_info=prev_reward_info
            )

            agent_0.update_policies(policy_state_0)
            agent_1.update_policies(policy_state_1)

            ind._translate_agent_policies_utils_to_env(agent_0)
            ind._translate_agent_policies_utils_to_env(agent_1)

            ind._wrap_policies_for_primitive_B_rollout(agent_0)
            ind._wrap_policies_for_primitive_B_rollout(agent_1)

            agent_0.infer_policies()
            agent_1.infer_policies()

            pol_idx_0 = ind._sample_or_argmax_policy_index(agent_0)
            pol_idx_1 = ind._sample_or_argmax_policy_index(agent_1)

            pol_0 = agent_0.policies[pol_idx_0]
            pol_1 = agent_1.policies[pol_idx_1]

            meta_0 = agent_0.get_policy_metadata()[pol_idx_0] if agent_0.get_policy_metadata() else None
            meta_1 = agent_1.get_policy_metadata()[pol_idx_1] if agent_1.get_policy_metadata() else None

            a0_prim = (
                ind._env_primitive_from_policy_step(pol_0[0])
                if len(pol_0) > 0
                else int(model_init_agent.STAY)
            )
            a1_prim = (
                ind._env_primitive_from_policy_step(pol_1[0])
                if len(pol_1) > 0
                else int(model_init_agent.STAY)
            )

            agent_0.action = (PRIMITIVE_POLICY_STEP, int(a0_prim), int(a1_prim))
            agent_0.step_time()
            agent_1.action = (PRIMITIVE_POLICY_STEP, int(a1_prim), int(a0_prim))
            agent_1.step_time()

            _observations, state, prev_reward_info, rewards, terminated, truncated, infos = (
                _execute_first_primitive_step(env_obj, a0_prim, a1_prim)
            )

            r0 = float(rewards["agent_0"])
            r1 = float(rewards["agent_1"])
            total_reward_0 += r0
            total_reward_1 += r1

            if log_steps:
                print(
                    "    Generated policies: A0={}  A1={}".format(
                        len(agent_0.policies or []),
                        len(agent_1.policies or []),
                    ),
                    flush=True,
                )

                qs_0 = agent_0.get_state_beliefs()
                print(ind._belief_table(np, qs_0, model_init_agent, title="Beliefs A0"), flush=True)

                qs_1 = agent_1.get_state_beliefs()
                print(ind._belief_table(np, qs_1, model_init_agent, title="Beliefs A1"), flush=True)

                print("    Policy beliefs A0:", flush=True)
                q_pi_0 = np.asarray(agent_0.get_policy_posterior(), dtype=float)
                H_pi_0 = float(-np.sum(q_pi_0 * np.log(q_pi_0 + 1e-16)))
                top_0 = agent_0.get_top_policies(top_k=5)
                print("      entropy {:.3f}:".format(H_pi_0), flush=True)
                if top_0:
                    best_pol, best_prob, best_idx = top_0[0]
                    print(
                        "      BEST: idx={} p={:.3f}  {}".format(
                            int(best_idx),
                            float(best_prob),
                            ind._fmt_policy(best_pol),
                        ),
                        flush=True,
                    )
                for rank, (pol, prob, _pidx) in enumerate(top_0, 1):
                    pol_str = ind._fmt_policy(pol)
                    bar = "█" * int(float(prob) * 20)
                    print("        #{:d} [{:>24}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)), flush=True)

                print("    Policy beliefs A1:", flush=True)
                q_pi_1 = np.asarray(agent_1.get_policy_posterior(), dtype=float)
                H_pi_1 = float(-np.sum(q_pi_1 * np.log(q_pi_1 + 1e-16)))
                top_1 = agent_1.get_top_policies(top_k=5)
                print("      entropy {:.3f}:".format(H_pi_1), flush=True)
                if top_1:
                    best_pol, best_prob, best_idx = top_1[0]
                    print(
                        "      BEST: idx={} p={:.3f}  {}".format(
                            int(best_idx),
                            float(best_prob),
                            ind._fmt_policy(best_pol),
                        ),
                        flush=True,
                    )
                for rank, (pol, prob, _pidx) in enumerate(top_1, 1):
                    pol_str = ind._fmt_policy(pol)
                    bar = "█" * int(float(prob) * 20)
                    print("        #{:d} [{:>24}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)), flush=True)

                if meta_0 is not None:
                    print(
                        "    A0 selected policy idx={}  semantic=({}, {})".format(
                            int(pol_idx_0),
                            meta_0["destination"],
                            meta_0["mode"],
                        ),
                        flush=True,
                    )
                else:
                    print("    A0 selected policy idx={}".format(int(pol_idx_0)), flush=True)

                if meta_1 is not None:
                    print(
                        "    A1 selected policy idx={}  semantic=({}, {})".format(
                            int(pol_idx_1),
                            meta_1["destination"],
                            meta_1["mode"],
                        ),
                        flush=True,
                    )
                else:
                    print("    A1 selected policy idx={}".format(int(pol_idx_1)), flush=True)

                print("    Primitive plan A0: {}".format(ind._fmt_policy(pol_0)), flush=True)
                print("    Primitive plan A1: {}".format(ind._fmt_policy(pol_1)), flush=True)

                print(
                    "    Executed primitive actions: A0={}  A1={}".format(
                        ind.PRIMITIVE_ACTION_NAMES.get(a0_prim, str(a0_prim)),
                        ind.PRIMITIVE_ACTION_NAMES.get(a1_prim, str(a1_prim)),
                    ),
                    flush=True,
                )

                print("    Reward A0: {}  (cumulative: {})".format(r0, total_reward_0), flush=True)
                print("    Reward A1: {}  (cumulative: {})".format(r1, total_reward_1), flush=True)

            if terminated.get("__all__") or truncated.get("__all__"):
                if log_steps:
                    print("    Episode ended.", flush=True)
                break

        if verbose or log_steps:
            print("\n  Scenario total reward A0: {}".format(total_reward_0), flush=True)
            print("  Scenario total reward A1: {}".format(total_reward_1), flush=True)

        return total_reward_0, total_reward_1

    print(
        "[Sweep] n_runs={} gamma={} alpha={} no_ig={} max_steps={} log_steps={}".format(
            n_runs, gamma, alpha, no_ig, max_steps_per_scenario, log_steps
        ),
        flush=True,
    )

    results = []
    for i in range(n_runs):
        ep_seed = int(episode_seeds[i])
        s0 = int(agent0_seeds[i])
        s1 = int(agent1_seeds[i])
        env = OvercookedMultiAgentEnv(config={"layout": env_layout, "horizon": horizon})
        agent_0 = create_agent(seed=s0, ego_agent_index=0)
        agent_1 = create_agent(seed=s1, ego_agent_index=1)

        name = "run {}/{}  episode_seed={}  agent_seeds=({}, {})".format(i + 1, n_runs, ep_seed, s0, s1)
        run_tag = "run{}/{}".format(i + 1, n_runs)
        r0, r1 = run_one_episode(
            env,
            agent_0,
            agent_1,
            name,
            ep_seed,
            log_steps=log_steps,
            run_tag=run_tag,
        )
        results.append((ep_seed, s0, s1, r0, r1))
        print(
            "  run {:d}: episode_seed={} agent_seeds=({}, {})  total_reward A0={:.3f} A1={:.3f}  sum={:.3f}".format(
                i + 1, ep_seed, s0, s1, r0, r1, r0 + r1
            ),
            flush=True,
        )

    sum_r0 = sum(t[3] for t in results)
    sum_r1 = sum(t[4] for t in results)
    print(
        "\n[Sweep] done. Mean total_reward per run: A0={:.4f} A1={:.4f}  combined_mean={:.4f}".format(
            sum_r0 / n_runs,
            sum_r1 / n_runs,
            (sum_r0 + sum_r1) / n_runs,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Seed sweep for semantic action-level runner (default gamma=4, alpha=8).")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of episodes (default: 5).")
    parser.add_argument(
        "--episode-seeds",
        type=str,
        default="76,77,78,79,80",
        help="Comma-separated env reset seeds, one per run (default: 76,77,78,79,80).",
    )
    parser.add_argument(
        "--agent0-seeds",
        type=str,
        default="1000,1001,1002,1003,1004",
        help="Comma-separated np.random seeds for agent 0 at construction (default: 1000..1004).",
    )
    parser.add_argument(
        "--agent1-seeds",
        type=str,
        default="2000,2001,2002,2003,2004",
        help="Comma-separated np.random seeds for agent 1 at construction (default: 2000..2004).",
    )
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument(
        "--noig",
        action="store_true",
        help="Disable epistemic value (state information gain) in policy evaluation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print episode title banners and totals only (no per-step log).",
    )
    parser.add_argument(
        "--log-steps",
        action="store_true",
        help="Per-step log like the main runner (map, obs, beliefs, policies, rewards); very verbose.",
    )
    args = parser.parse_args()

    def _parse_int_list(s: str) -> list[int]:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [int(p) for p in parts]

    n_runs = int(args.n_runs)
    episode_seeds = _parse_int_list(args.episode_seeds)
    agent0_seeds = _parse_int_list(args.agent0_seeds)
    agent1_seeds = _parse_int_list(args.agent1_seeds)

    if len(episode_seeds) == 1 and n_runs > 1:
        episode_seeds = [episode_seeds[0] + k for k in range(n_runs)]
    if len(agent0_seeds) == 1 and n_runs > 1:
        agent0_seeds = [agent0_seeds[0] + k for k in range(n_runs)]
    if len(agent1_seeds) == 1 and n_runs > 1:
        agent1_seeds = [agent1_seeds[0] + k for k in range(n_runs)]

    if not (len(episode_seeds) == n_runs and len(agent0_seeds) == n_runs and len(agent1_seeds) == n_runs):
        print(
            "Error: need exactly {} seeds in each list (got episode={}, agent0={}, agent1={}).".format(
                n_runs,
                len(episode_seeds),
                len(agent0_seeds),
                len(agent1_seeds),
            )
        )
        sys.exit(1)

    _run_sweep(
        n_runs=n_runs,
        episode_seeds=episode_seeds,
        agent0_seeds=agent0_seeds,
        agent1_seeds=agent1_seeds,
        gamma=float(args.gamma),
        alpha=float(args.alpha),
        no_ig=bool(args.noig),
        verbose=bool(args.verbose),
        log_steps=bool(args.log_steps),
    )


if __name__ == "__main__":
    main()