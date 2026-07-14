"""
Run the FullyCollective paradigm at the semantic-action level on Overcooked.

Setup:
    - ONE IC brain (active inference agent) controls BOTH physical agents.
    - The brain plans over the joint semantic-pair space
      `(JOINT_PAIR_LABEL, a_self, a_other)` with `N_ACTIONS × N_ACTIONS = 400`
      options.  Policy inference picks a single joint pair per env step.
    - Agent 0 = the IC brain (ego_agent_index=0). Agent 1 = puppet: it has no
      beliefs and no policy inference — each step it simply executes the
      primitive action prescribed for it by the IC brain's chosen joint pair.

Per-step loop:
    1. The IC brain observes the world from agent_0's perspective and infers
       beliefs over both self and other state factors.
    2. Compile per-agent semantic libraries:
         - 20 primitive plans for agent_0 (the IC brain's own moves)
         - 20 primitive plans for agent_1 (the puppet's moves)
    3. Construct 400 joint semantic-pair policies and run policy inference.
    4. Sample one joint pair `(a_self_sem, a_other_sem)`.
    5. Take the first primitive of each agent's compiled plan that matches
       the sampled semantic action and step the env with both primitives.

This file deliberately reuses helpers from
`run_independent_semantic_action_level.py` for terminal rendering, policy
sampling and primitive-step bookkeeping, so its structure mirrors the
existing IC / Independent runners.
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
import sal_step_csv_log as sal_csv
import sal_step_detail_log as sal_detail

from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.FullyCollectiveWithSemanticPoliciesActionLevel.model_init import (
    PRIMITIVE_POLICY_STEP,
)


def _compile_primitive_plans_for_agent(agent, policy_state):
    """
    Run dynamic policy generation for one frame from `policy_state`'s ego
    perspective and return that frame's per-semantic-action compiled primitive
    plans (as lists of raw env primitive ints) + the matching metadata.

    Side effect: mutates `agent.policies` / `agent.policy_metadata`. The caller
    is responsible for replacing them with the joint-pair primitive policies
    before `infer_policies()`.
    """
    agent.update_policies(policy_state)
    ind._translate_agent_policies_utils_to_env(agent)
    raw_plans = [list(p) for p in (agent.policies or [])]
    metadata = list(agent.policy_metadata) if agent.policy_metadata else []
    return raw_plans, metadata


def _build_joint_primitive_policies(
    primitive_plans_self: list[list[int]],
    primitive_plans_other: list[list[int]],
    n_actions: int,
    stay_idx: int,
):
    """
    Pair each (a_self_sem, a_other_sem) semantic combination into a joint
    *primitive* policy that the IC brain will roll forward through
    `B_fn_primitive_step` (real per-step physics, mirroring IC's wrap).

    Each policy is a sequence of joint primitive steps:
        [(PRIMITIVE_POLICY_STEP, a_self_prim_t, a_other_prim_t), ...]

    Shorter plans are right-padded with STAY so the two compiled paths share
    a common horizon.  This is the FC analogue of IC's per-semantic-action
    primitive rollout — the only difference is that here the partner action
    is fully specified (joint primitive rollout) instead of left to the
    autonomous-other branch of B_fn.

    Returns:
        joint_policies: list[list[tuple]] of length n_actions ** 2
        joint_pairs: list[tuple[int, int]] parallel index of (a_s_sem, a_o_sem)
    """
    joint_policies: list[list[tuple]] = []
    joint_pairs: list[tuple[int, int]] = []

    def _safe(plans, i):
        if 0 <= i < len(plans) and plans[i]:
            return [int(a) for a in plans[i]]
        return [int(stay_idx)]

    for a_self_sem in range(n_actions):
        plan_s = _safe(primitive_plans_self, a_self_sem)
        for a_other_sem in range(n_actions):
            plan_o = _safe(primitive_plans_other, a_other_sem)
            horizon = max(len(plan_s), len(plan_o))
            pol = []
            for t in range(horizon):
                a_s = plan_s[t] if t < len(plan_s) else int(stay_idx)
                a_o = plan_o[t] if t < len(plan_o) else int(stay_idx)
                pol.append((PRIMITIVE_POLICY_STEP, int(a_s), int(a_o)))
            joint_policies.append(pol)
            joint_pairs.append((int(a_self_sem), int(a_other_sem)))

    return joint_policies, joint_pairs


def _first_primitive_from_plan(plan: list[int], default_action: int) -> int:
    if not plan:
        return int(default_action)
    return int(plan[0])


def _run_fc(
    *,
    n_runs: int,
    episode_seeds: list[int],
    agent_seeds: list[int],
    gamma: float,
    alpha: float,
    no_ig: bool,
    verbose: bool,
    log_steps: bool,
    log_csv: bool = False,
    log_jsonl: bool = False,
    log_dir: str | None = None,
    policy_log_top_k: int = 20,
    log_full_q_pi: bool = False,
    max_steps: int = 2000,
) -> None:
    if len(episode_seeds) != n_runs or len(agent_seeds) != n_runs:
        raise ValueError("episode_seeds and agent_seeds must each have length n_runs")

    try:
        from agents.IndependentActiveInferenceWithDynamicPolicies.agent import Agent
        from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.FullyCollectiveWithSemanticPoliciesActionLevel import (
            A_fn,
            B_fn,
            C_fn,
            D_fn,
            model_init as fc_model_init,
            env_utils,
        )
        from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
    except Exception as e:
        print("[SKIP] Could not load agent or environment: {}".format(e))
        return

    if int(getattr(fc_model_init, "N_PRIMITIVE_ACTIONS", ind.N_PRIMITIVE_ACTIONS)) != ind.N_PRIMITIVE_ACTIONS:
        print(
            "[SKIP] model_init N_PRIMITIVE_ACTIONS ({}) != ind.N_PRIMITIVE_ACTIONS ({}).".format(
                getattr(fc_model_init, "N_PRIMITIVE_ACTIONS", None),
                ind.N_PRIMITIVE_ACTIONS,
            )
        )
        return

    state_factors = list(fc_model_init.states.keys())
    state_sizes = {f: len(v) for f, v in fc_model_init.states.items()}
    observation_labels = fc_model_init.observations
    base_env_params = {"width": fc_model_init.GRID_WIDTH, "height": fc_model_init.GRID_HEIGHT}
    # FC brain plans single-step joint-pair semantic policies.
    policy_len = 1
    max_steps_per_scenario = int(max_steps)
    horizon = max_steps_per_scenario + 10
    env_layout = "cramped_room"

    N_ACTIONS = int(fc_model_init.N_ACTIONS)

    def create_brain(seed=None):
        if seed is not None:
            np.random.seed(seed)
        env_params = {**base_env_params, "ego_agent_index": 0}
        agent = Agent(
            A_fn=A_fn,
            B_fn=B_fn,
            C_fn=C_fn,
            D_fn=D_fn,
            state_factors=state_factors,
            state_sizes=state_sizes,
            observation_labels=observation_labels,
            env_params=env_params,
            observation_state_dependencies=fc_model_init.observation_state_dependencies,
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
        brain,
        episode_name: str,
        episode_seed: int | None,
        *,
        log_steps: bool,
        run_tag: str,
        log_csv: bool = False,
        log_jsonl: bool = False,
        log_dir: str | None = None,
        brain_seed: int | None = None,
        policy_log_top_k: int = 20,
        log_full_q_pi: bool = False,
    ):
        _obs, infos = env_obj.reset(seed=episode_seed)
        state = infos["agent_0"]["state"]

        log_base = log_dir or sal_csv._repo_logs_dir()
        want_detail = bool(log_steps or log_jsonl)

        step_csv = None
        if log_csv:
            step_csv = sal_csv.open_fc_log(
                log_base,
                int(episode_seed or 0),
                int(brain_seed or 0),
            )

        step_jsonl = None
        if log_jsonl:
            step_jsonl = sal_detail.open_jsonl(
                log_base,
                "fc",
                episode_seed=int(episode_seed or 0),
                brain_seed=int(brain_seed or 0),
            )

        config = env_utils.get_D_config_from_state(state, 0)
        brain.reset(config=config)

        prev_reward_info = {"sparse_reward_by_agent": [0, 0]}
        total_reward_0 = 0.0
        total_reward_1 = 0.0

        if verbose or log_steps:
            print("\n" + "=" * 72, flush=True)
            print("  {}".format(episode_name), flush=True)
            print("=" * 72, flush=True)

        for step in range(1, max_steps_per_scenario + 1):
            obs_0 = env_utils.env_obs_to_model_obs(state, 0, reward_info=prev_reward_info)

            map_before = (
                sal_detail.map_lines(state, fc_model_init, ind.render_overcooked_grid)
                if want_detail
                else None
            )

            if log_steps:
                state_str = ind._state_summary(state, fc_model_init, max_agents=2)
                print("\n  --- [{}] Step {} ---".format(run_tag, step), flush=True)
                print("    Env state:  {}".format(state_str), flush=True)
                sal_detail.print_map("Map (before action)", map_before or [])
                for line in ind._agent_summary_lines(state, fc_model_init, max_agents=2):
                    print(line, flush=True)
                print(
                    "    Obs (brain ego=A0): self_pos={} self_ori={} self_held={} "
                    "other_pos={} other_held={} pot={} delivered={}".format(
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

            brain.infer_states(obs_0)

            policy_state_self = ind.build_policy_state_for_agent(
                state, agent_idx=0, env_utils=env_utils, prev_reward_info=prev_reward_info
            )
            policy_state_other = ind.build_policy_state_for_agent(
                state, agent_idx=1, env_utils=env_utils, prev_reward_info=prev_reward_info
            )

            primitive_plans_self, metadata_self = _compile_primitive_plans_for_agent(
                brain, policy_state_self
            )
            primitive_plans_other, metadata_other = _compile_primitive_plans_for_agent(
                brain, policy_state_other
            )

            if len(primitive_plans_self) < N_ACTIONS or len(primitive_plans_other) < N_ACTIONS:
                if log_steps:
                    print(
                        "    [WARN] compiled plans short: self={} other={} expected={}".format(
                            len(primitive_plans_self), len(primitive_plans_other), N_ACTIONS
                        ),
                        flush=True,
                    )

            stay_idx = int(fc_model_init.STAY)

            # Build the 400 joint *primitive* policies (paired primitive rollouts,
            # padded with STAY).  Rolling each through B_fn_primitive_step
            # mirrors IC's per-semantic primitive rollout — same horizon, same
            # physics — and is what makes EFE differentiate across joint pairs.
            joint_policies, joint_pairs = _build_joint_primitive_policies(
                primitive_plans_self,
                primitive_plans_other,
                N_ACTIONS,
                stay_idx,
            )
            brain.policies = joint_policies
            brain.policy_lengths = [len(p) for p in joint_policies]
            brain.policy_metadata = []
            brain.q_pi = np.ones(len(joint_policies), dtype=float) / len(joint_policies)

            brain.infer_policies()

            joint_idx = ind._sample_or_argmax_policy_index(brain)
            a_self_sem, a_other_sem = joint_pairs[joint_idx]

            plan_self = (
                primitive_plans_self[a_self_sem]
                if 0 <= a_self_sem < len(primitive_plans_self)
                else []
            )
            plan_other = (
                primitive_plans_other[a_other_sem]
                if 0 <= a_other_sem < len(primitive_plans_other)
                else []
            )
            a0_prim = _first_primitive_from_plan(plan_self, stay_idx)
            a1_prim = _first_primitive_from_plan(plan_other, stay_idx)

            # Record the chosen primitive step for next-step belief propagation.
            # B_fn understands (PRIMITIVE_POLICY_STEP, a_self, a_other) as a joint
            # primitive timestep — the brain's belief about the partner advances
            # alongside its own.
            brain.action = (PRIMITIVE_POLICY_STEP, int(a0_prim), int(a1_prim))
            brain.step_time()

            (
                _observations,
                state,
                prev_reward_info,
                rewards,
                terminated,
                truncated,
                infos,
            ) = _execute_first_primitive_step(env_obj, a0_prim, a1_prim)

            r0 = float(rewards["agent_0"])
            r1 = float(rewards["agent_1"])
            total_reward_0 += r0
            total_reward_1 += r1

            if step_csv is not None:
                ps = sal_csv._policy_stats(brain, ind._fmt_policy)
                name_s = fc_model_init.ACTION_NAMES.get(int(a_self_sem), str(int(a_self_sem)))
                name_o = fc_model_init.ACTION_NAMES.get(int(a_other_sem), str(int(a_other_sem)))
                step_csv.write(
                    {
                        "paradigm": "fc",
                        "episode_seed": int(episode_seed or 0),
                        "brain_seed": int(brain_seed or 0),
                        "step": int(step),
                        "joint_policy_idx": int(joint_idx),
                        "joint_semantic_self": name_s,
                        "joint_semantic_other": name_o,
                        "a0_semantic_idx": int(a_self_sem),
                        "a1_semantic_idx": int(a_other_sem),
                        "a0_primitive": int(a0_prim),
                        "a0_primitive_name": ind.PRIMITIVE_ACTION_NAMES.get(
                            a0_prim, str(a0_prim)
                        ),
                        "a1_primitive": int(a1_prim),
                        "a1_primitive_name": ind.PRIMITIVE_ACTION_NAMES.get(
                            a1_prim, str(a1_prim)
                        ),
                        "brain_q_pi_entropy": ps["q_pi_entropy"],
                        "brain_top_policy_prob": ps["top_policy_prob"],
                        "reward_a0": r0,
                        "reward_a1": r1,
                        "cumulative_reward_a0": total_reward_0,
                        "cumulative_reward_a1": total_reward_1,
                        "terminated": bool(terminated.get("__all__")),
                        "truncated": bool(truncated.get("__all__")),
                    }
                )

            def _fc_joint_label(pidx: int, _agent) -> str:
                a_s, a_o = joint_pairs[int(pidx)]
                ns = fc_model_init.ACTION_NAMES.get(int(a_s), str(int(a_s)))
                no = fc_model_init.ACTION_NAMES.get(int(a_o), str(int(a_o)))
                return "S:{} | O:{}".format(ns, no)

            if log_steps:
                print(
                    "    Compiled libraries: self_plans={}  other_plans={}".format(
                        len(primitive_plans_self),
                        len(primitive_plans_other),
                    ),
                    flush=True,
                )
                sal_detail.print_agent_beliefs(
                    brain,
                    np_mod=np,
                    model_init=fc_model_init,
                    agent_label="brain",
                    belief_table_fn=ind._belief_table,
                    policy_label_fn=_fc_joint_label,
                    policy_top_k=policy_log_top_k,
                    policy_full=log_full_q_pi,
                    state_belief_title="Beliefs (brain, ego=A0)",
                    policy_belief_title="Policy beliefs (brain)",
                    policy_label_width=34,
                )

                name_s = fc_model_init.ACTION_NAMES.get(int(a_self_sem), str(int(a_self_sem)))
                name_o = fc_model_init.ACTION_NAMES.get(int(a_other_sem), str(int(a_other_sem)))
                print(
                    "    Selected joint pair: self={}  other={}".format(name_s, name_o),
                    flush=True,
                )

                meta_s_str = (
                    "{}:{}".format(
                        metadata_self[a_self_sem]["destination"],
                        metadata_self[a_self_sem]["mode"],
                    )
                    if 0 <= a_self_sem < len(metadata_self)
                    else "?"
                )
                meta_o_str = (
                    "{}:{}".format(
                        metadata_other[a_other_sem]["destination"],
                        metadata_other[a_other_sem]["mode"],
                    )
                    if 0 <= a_other_sem < len(metadata_other)
                    else "?"
                )
                print(
                    "    Self plan ({}):  {}".format(meta_s_str, ind._fmt_policy(plan_self)),
                    flush=True,
                )
                print(
                    "    Other plan ({}): {}".format(meta_o_str, ind._fmt_policy(plan_other)),
                    flush=True,
                )

                print(
                    "    Executed primitive actions: A0={}  A1={}".format(
                        ind.PRIMITIVE_ACTION_NAMES.get(a0_prim, str(a0_prim)),
                        ind.PRIMITIVE_ACTION_NAMES.get(a1_prim, str(a1_prim)),
                    ),
                    flush=True,
                )

                print("    Reward A0: {}  (cumulative: {})".format(r0, total_reward_0), flush=True)
                print("    Reward A1: {}  (cumulative: {})".format(r1, total_reward_1), flush=True)

            if want_detail and map_before is not None:
                map_after = sal_detail.map_lines(state, fc_model_init, ind.render_overcooked_grid)
                if log_steps:
                    sal_detail.print_map("Map (after action)", map_after)
                sal_detail.write_fc_step(
                    step_jsonl,
                    step=step,
                    episode_seed=int(episode_seed or 0),
                    brain_seed=int(brain_seed or 0),
                    map_before=map_before,
                    map_after=map_after,
                    brain=brain,
                    joint_label_fn=_fc_joint_label,
                    joint_pairs=joint_pairs,
                    joint_idx=int(joint_idx),
                    a_self_sem=int(a_self_sem),
                    a_other_sem=int(a_other_sem),
                    a0_prim=int(a0_prim),
                    a1_prim=int(a1_prim),
                    a0_prim_name=ind.PRIMITIVE_ACTION_NAMES.get(a0_prim, str(a0_prim)),
                    a1_prim_name=ind.PRIMITIVE_ACTION_NAMES.get(a1_prim, str(a1_prim)),
                    action_names=fc_model_init.ACTION_NAMES,
                    reward_a0=r0,
                    reward_a1=r1,
                    cumulative_reward_a0=total_reward_0,
                    cumulative_reward_a1=total_reward_1,
                    terminated=bool(terminated.get("__all__")),
                    truncated=bool(truncated.get("__all__")),
                    policy_top_k=policy_log_top_k,
                    include_full_q_pi=bool(log_jsonl or log_full_q_pi),
                )

            if terminated.get("__all__") or truncated.get("__all__"):
                if log_steps:
                    print("    Episode ended.", flush=True)
                break

        if verbose or log_steps:
            print("\n  Scenario total reward A0: {}".format(total_reward_0), flush=True)
            print("  Scenario total reward A1: {}".format(total_reward_1), flush=True)

        csv_path = None
        if step_csv is not None:
            step_csv.close()
            csv_path = step_csv.path
            print("  Step CSV: {}".format(csv_path), flush=True)

        jsonl_path = None
        if step_jsonl is not None:
            step_jsonl.close()
            jsonl_path = step_jsonl.path
            print("  Step JSONL: {}".format(jsonl_path), flush=True)

        return total_reward_0, total_reward_1, csv_path, jsonl_path

    print(
        "[FC] n_runs={} gamma={} alpha={} no_ig={} max_steps={} log_steps={} log_csv={} log_jsonl={}".format(
            n_runs, gamma, alpha, no_ig, max_steps_per_scenario, log_steps, log_csv, log_jsonl
        ),
        flush=True,
    )
    if log_csv or log_jsonl:
        print("[FC] Detail log dir: {}".format(log_dir or sal_csv._repo_logs_dir()), flush=True)

    results = []
    for i in range(n_runs):
        ep_seed = int(episode_seeds[i])
        s_brain = int(agent_seeds[i])
        env = OvercookedMultiAgentEnv(config={"layout": env_layout, "horizon": horizon})
        brain = create_brain(seed=s_brain)

        name = "run {}/{}  episode_seed={}  brain_seed={}".format(i + 1, n_runs, ep_seed, s_brain)
        run_tag = "run{}/{}".format(i + 1, n_runs)
        r0, r1, csv_path, jsonl_path = run_one_episode(
            env,
            brain,
            name,
            ep_seed,
            log_steps=log_steps,
            run_tag=run_tag,
            log_csv=log_csv,
            log_jsonl=log_jsonl,
            log_dir=log_dir,
            brain_seed=s_brain,
            policy_log_top_k=policy_log_top_k,
            log_full_q_pi=log_full_q_pi,
        )
        results.append((ep_seed, s_brain, r0, r1, csv_path, jsonl_path))
        print(
            "  run {:d}: episode_seed={} brain_seed={}  total_reward A0={:.3f} A1={:.3f}  sum={:.3f}".format(
                i + 1, ep_seed, s_brain, r0, r1, r0 + r1
            ),
            flush=True,
        )

    sum_r0 = sum(t[2] for t in results)
    sum_r1 = sum(t[3] for t in results)
    csv_paths = [t[4] for t in results if t[4] is not None]
    jsonl_paths = [t[5] for t in results if t[5] is not None]
    if csv_paths:
        print("\n[FC] Step CSV files:", flush=True)
        for p in csv_paths:
            print("  {}".format(p), flush=True)
    if jsonl_paths:
        print("\n[FC] Step JSONL files:", flush=True)
        for p in jsonl_paths:
            print("  {}".format(p), flush=True)
    print(
        "\n[FC] done. Mean total_reward per run: A0={:.4f} A1={:.4f}  combined_mean={:.4f}".format(
            sum_r0 / n_runs,
            sum_r1 / n_runs,
            (sum_r0 + sum_r1) / n_runs,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "FullyCollective semantic-action-level runner: one IC brain plans over "
            "joint semantic pairs and the puppet executes the prescription."
        )
    )
    parser.add_argument("--n-runs", type=int, default=5, help="Number of episodes (default: 5).")
    parser.add_argument(
        "--episode-seeds",
        type=str,
        default="76,77,78,79,80",
        help="Comma-separated env reset seeds, one per run (default: 76,77,78,79,80).",
    )
    parser.add_argument(
        "--agent-seeds",
        type=str,
        default="1000,1001,1002,1003,1004",
        help="Comma-separated np.random seeds for the IC brain at construction (default: 1000..1004).",
    )
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum primitive steps per episode (default: 2000).",
    )
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
        help="Per-step log (map, obs, beliefs, joint-pair policy posteriors, rewards); very verbose.",
    )
    parser.add_argument(
        "--log-csv",
        action="store_true",
        help="Write per-step CSV (Excel-friendly) under --log-dir (default: repo logs/).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for step CSV / JSONL files (default: <repo>/logs).",
    )
    parser.add_argument(
        "--log-jsonl",
        action="store_true",
        help="Write per-step JSONL with map snapshots, full state beliefs, and full q_pi.",
    )
    parser.add_argument(
        "--policy-log-top-k",
        type=int,
        default=20,
        help="Number of policies to list in stdout logs (default: 20).",
    )
    parser.add_argument(
        "--log-full-q-pi",
        action="store_true",
        help="List every policy in stdout logs (can be huge for IC/FC). JSONL always includes full q_pi when --log-jsonl is set.",
    )
    args = parser.parse_args()

    def _parse_int_list(s: str) -> list[int]:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [int(p) for p in parts]

    n_runs = int(args.n_runs)
    episode_seeds = _parse_int_list(args.episode_seeds)
    agent_seeds = _parse_int_list(args.agent_seeds)

    if len(episode_seeds) == 1 and n_runs > 1:
        episode_seeds = [episode_seeds[0] + k for k in range(n_runs)]
    if len(agent_seeds) == 1 and n_runs > 1:
        agent_seeds = [agent_seeds[0] + k for k in range(n_runs)]

    if not (len(episode_seeds) == n_runs and len(agent_seeds) == n_runs):
        print(
            "Error: need exactly {} seeds in each list (got episode={}, agent={}).".format(
                n_runs,
                len(episode_seeds),
                len(agent_seeds),
            )
        )
        sys.exit(1)

    _run_fc(
        n_runs=n_runs,
        episode_seeds=episode_seeds,
        agent_seeds=agent_seeds,
        gamma=float(args.gamma),
        alpha=float(args.alpha),
        no_ig=bool(args.noig),
        verbose=bool(args.verbose),
        log_steps=bool(args.log_steps),
        log_csv=bool(args.log_csv),
        log_jsonl=bool(args.log_jsonl),
        log_dir=args.log_dir,
        policy_log_top_k=int(args.policy_log_top_k),
        log_full_q_pi=bool(args.log_full_q_pi),
        max_steps=int(args.max_steps),
    )


if __name__ == "__main__":
    main()
