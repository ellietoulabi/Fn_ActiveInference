import argparse
import sys
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

overcooked_src = PROJECT_ROOT / "environments" / "overcooked_ai" / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))


# -----------------------------------------------------------------------------
# Two-agent Overcooked sim: IndividuallyCollective AIF with ego-framed joint
# pair policies and decentralized execution.
#
# IMPORTANT SEMANTICS:
# - Each agent evaluates global joint semantic pair actions (a0, a1).
# - Agent 0 interprets a pair as (self=a0, other=a1).
# - Agent 1 interprets the same pair as (self=a1, other=a0).
# - Each agent chooses its own preferred joint pair independently.
# - Each agent stores ITS OWN SELECTED JOINT PAIR in agent.action, because the
#   Agent class uses self.action inside infer_states() to construct the prior.
# - In the real environment, execution is decentralized:
#       env agent 0 executes only the a0 part of agent 0's chosen pair
#       env agent 1 executes only the a1 part of agent 1's chosen pair
# - Therefore executed env actions may form a hybrid pair, but this hybrid pair
#   is NOT written back into agent.action.
# -----------------------------------------------------------------------------

from utils.visualization.overcooked_terminal_map import orientation_str, render_overcooked_grid


PRIMITIVE_ACTION_NAMES = {
    0: "NORTH",
    1: "SOUTH",
    2: "EAST",
    3: "WEST",
    4: "STAY",
    5: "INTERACT",
}

N_PRIMITIVE_ACTIONS = 6
PAIR_POLICY_HORIZON = 1
MACRO_MAX_PRIMITIVE_STEPS = 8

_ORI_NAMES = {0: "N", 1: "S", 2: "E", 3: "W"}
_HELD_NAMES = {0: "NONE", 1: "ONION", 2: "DISH", 3: "SOUP"}
_POT_NAMES = {0: "P0", 1: "P1", 2: "P2", 3: "P3(READY)"}
_CTR_NAMES = {0: "EMPTY", 1: "ONION", 2: "DISH", 3: "SOUP"}


def _fmt_prob(x: float) -> str:
    return "{:>5.2f}".format(float(x))


def _map_stats(np_mod, p):
    p = np.asarray(p, dtype=float)
    i = int(np.argmax(p)) if p.size else 0
    pm = float(p[i]) if p.size else 0.0
    H = float(-np.sum(p * np.log(p + 1e-16))) if p.size else 0.0
    return i, pm, H


def _fmt_pos(model_init, idx: int) -> str:
    try:
        grid = model_init.walkable_idx_to_grid_idx(int(idx))
        x, y = model_init.index_to_xy(int(grid))
        return f"w{int(idx)} (g{int(grid)} {int(x)},{int(y)})"
    except Exception:
        return f"w{int(idx)}"


def _fmt_pair_step(step_action, semantic_model_init) -> str:
    if isinstance(step_action, (tuple, list)) and len(step_action) == 3:
        a0 = int(step_action[1])
        a1 = int(step_action[2])
        return "A0:{}|A1:{}".format(
            semantic_model_init.ACTION_NAMES.get(a0, str(a0)),
            semantic_model_init.ACTION_NAMES.get(a1, str(a1)),
        )
    return str(step_action)


def _fmt_policy(pol, semantic_model_init) -> str:
    return "→".join([_fmt_pair_step(a, semantic_model_init) for a in pol])


def _construct_joint_pair_policies(joint_label: str, n_semantic_actions: int):
    from itertools import product

    step_pairs = [
        (joint_label, a0, a1)
        for a0 in range(n_semantic_actions)
        for a1 in range(n_semantic_actions)
    ]
    return [[*steps] for steps in product(step_pairs, repeat=PAIR_POLICY_HORIZON)]


def _belief_table(np_mod, qs: dict, model_init, title: str) -> str:
    lines = []
    lines.append(f"    {title}")
    lines.append("      factor               MAP                 p(MAP)    H")
    lines.append("      -------------------  ------------------  -------  -----")

    def row(name: str, value_str: str, pm: float, H: float):
        lines.append("      {:<19}  {:<18}  {:>7}  {:>5.2f}".format(name, value_str, _fmt_prob(pm), H))

    for f in ("self_pos", "self_orientation", "self_held", "other_pos", "other_orientation", "other_held"):
        p = qs.get(f, None)
        if p is None:
            continue
        i, pm, H = _map_stats(np_mod, p)
        if f.endswith("_pos"):
            v = _fmt_pos(model_init, i)
        elif f.endswith("_orientation"):
            v = _ORI_NAMES.get(i, str(i))
        elif f.endswith("_held"):
            v = _HELD_NAMES.get(i, str(i))
        else:
            v = str(i)
        row(f, v, pm, H)

    for f in ("pot_state", "ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
        p = qs.get(f, None)
        if p is None:
            continue
        i, pm, H = _map_stats(np_mod, p)
        if f == "pot_state":
            v = _POT_NAMES.get(i, str(i))
            row(f, v, pm, H)
        else:
            p = np.asarray(p, dtype=float)
            p1 = float(p[1]) if p.size > 1 else 0.0
            v = "1" if i == 1 else "0"
            lines.append(
                "      {:<19}  {:<18}  {:>7}  {:>5.2f}   (p1={})".format(
                    f, v, _fmt_prob(pm), H, _fmt_prob(p1)
                )
            )

    ctr_factors = [k for k in qs.keys() if k.startswith("ctr_")]
    if ctr_factors:
        lines.append("      -- counters (MAP, p(nonempty)) --")
        for k in sorted(ctr_factors, key=lambda s: int(s.split("_")[1])):
            p = np.asarray(qs[k], dtype=float)
            i, pm, _H = _map_stats(np_mod, p)
            p_empty = float(p[0]) if p.size > 0 else 1.0
            p_nonempty = float(1.0 - p_empty)
            v = _CTR_NAMES.get(i, str(i))
            lines.append("      {:<19}  {:<18}  p(nonempty)={}".format(k, v, _fmt_prob(p_nonempty)))

    return "\n".join(lines)


def _state_summary(state, model_init, max_agents: int | None = None):
    parts = []
    players = state.players if max_agents is None else state.players[:max_agents]
    for i, p in enumerate(players):
        pos = p.position
        grid_idx = model_init.xy_to_index(pos[0], pos[1])
        w = model_init.grid_idx_to_walkable_idx(grid_idx)
        w = w if w is not None else -1
        held = "none"
        if p.has_object() and p.held_object:
            held = getattr(p.held_object, "name", str(p.held_object))
        parts.append("A{}@{} held={}".format(i, w, held))

    pot = "empty"
    for (pos, obj) in state.objects.items():
        if not obj or getattr(obj, "name", None) != "soup":
            continue
        grid_idx = model_init.xy_to_index(pos[0], pos[1])
        if grid_idx not in model_init.POT_INDICES:
            continue

        ing = getattr(obj, "ingredients", [])
        n = len(ing) if ing else 0
        is_idle = bool(getattr(obj, "is_idle", False))
        is_cooking = bool(getattr(obj, "is_cooking", False))
        is_ready = bool(getattr(obj, "is_ready", False))
        tick = getattr(obj, "_cooking_tick", None)
        ct = None
        if not is_idle:
            try:
                ct = getattr(obj, "cook_time", None)
            except Exception:
                ct = None

        if is_ready:
            phase = "ready"
        elif is_cooking:
            phase = "cooking"
        elif is_idle:
            phase = "idle"
        else:
            phase = "unknown"

        extra = ""
        if tick is not None and ct is not None and not is_idle:
            extra = " t={}/{}".format(int(tick), int(ct))
        pot = "{}onion({}{})".format(n, phase, extra)
        break

    return " | ".join(parts) + " | pot={}".format(pot)


def _agent_summary_lines(state, model_init, max_agents: int | None = None):
    lines = []
    players = state.players if max_agents is None else state.players[:max_agents]
    for i, p in enumerate(players):
        pos = p.position
        grid_idx = model_init.xy_to_index(pos[0], pos[1])
        w = model_init.grid_idx_to_walkable_idx(grid_idx)
        w = w if w is not None else -1
        held = "none"
        if p.has_object() and p.held_object:
            held = getattr(p.held_object, "name", str(p.held_object))
        facing, _ = orientation_str(p)
        lines.append("    A{}: pos(walkable)={}  holding={}  facing={}".format(i, w, held, facing))
    return lines


def _sample_or_argmax_policy_index(agent) -> int:
    q_pi = np.asarray(agent.get_policy_posterior(), dtype=np.float64).reshape(-1)
    n_pol = len(q_pi)
    if n_pol == 0:
        return 0

    if getattr(agent, "action_selection", "stochastic") == "deterministic":
        return int(np.argmax(q_pi))

    alpha = float(getattr(agent, "alpha", 1.0))
    log_q = np.log(np.maximum(q_pi, 1e-16))
    p_policies = np.exp(log_q * alpha)
    p_policies = np.maximum(p_policies, 0.0)

    s = float(np.sum(p_policies))
    if s <= 0.0 or not np.isfinite(s):
        p_policies = np.ones(n_pol, dtype=np.float64) / float(n_pol)
    else:
        p_policies = p_policies / s
        if n_pol > 1:
            p_policies[-1] = 1.0 - float(np.sum(p_policies[:-1]))
        else:
            p_policies[0] = 1.0

    return int(np.random.choice(n_pol, p=p_policies))


def _independent_joint_first_steps(agent_0, agent_1):
    """
    Each agent selects its OWN preferred joint step independently.

    Crucially, each agent stores ITS OWN selected joint step as agent.action,
    because the Agent class uses self.action during the next infer_states()
    call to construct the transition prior.

    Returns:
      pol_idx_0, step_0, pol_idx_1, step_1
    """
    pol_idx_0 = _sample_or_argmax_policy_index(agent_0)
    pol_idx_1 = _sample_or_argmax_policy_index(agent_1)

    step_0 = agent_0.policies[pol_idx_0][0]
    step_1 = agent_1.policies[pol_idx_1][0]

    # Keep each agent internally aligned with the joint action it selected.
    agent_0.action = step_0
    agent_0.step_time()

    agent_1.action = step_1
    agent_1.step_time()

    return pol_idx_0, step_0, pol_idx_1, step_1


def run_agent_vs_env_scenarios():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--noprint",
        action="store_true",
        help="If set, suppress all logs and print only the step number each iteration.",
    )
    parser.add_argument(
        "--noig",
        action="store_true",
        help="Disable epistemic value (state information gain) in policy evaluation.",
    )
    args, _unknown = parser.parse_known_args()
    noprint = bool(args.noprint)
    verbose = not noprint
    no_ig = bool(args.noig)

    try:
        from agents.ActiveInferenceFixedPolicies.agent import Agent
        from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollectiveWithSemanticPolicies import (
            A_fn,
            B_fn,
            C_fn,
            D_fn,
            model_init as mon_model_init,
            env_utils,
        )
        from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
    except Exception as e:
        if verbose:
            print("\n[SKIP] Agent vs env: could not load agent or environment: {}".format(e))
        return

    model_init_agent = mon_model_init
    if int(getattr(model_init_agent, "N_PRIMITIVE_ACTIONS", N_PRIMITIVE_ACTIONS)) != N_PRIMITIVE_ACTIONS:
        if verbose:
            print(
                "\n[SKIP] model_init N_PRIMITIVE_ACTIONS ({}) != script N_PRIMITIVE_ACTIONS ({}).".format(
                    getattr(model_init_agent, "N_PRIMITIVE_ACTIONS", None),
                    N_PRIMITIVE_ACTIONS,
                )
            )
        return

    N_SEMANTIC_ACTIONS = int(getattr(model_init_agent, "N_ACTIONS", 0))
    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    base_env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}
    policy_len = PAIR_POLICY_HORIZON
    joint_label = str(getattr(model_init_agent, "JOINT_PAIR_LABEL", "__joint_pair__"))

    def create_agent(seed=None, ego_agent_index: int = 0):
        if seed is not None:
            np.random.seed(seed)

        policies = _construct_joint_pair_policies(joint_label, N_SEMANTIC_ACTIONS)
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
            actions=list(range(N_SEMANTIC_ACTIONS)),
            gamma=8.0,
            alpha=32.0,
            policies=policies,
            policy_len=policy_len,
            inference_horizon=policy_len,
            action_selection="stochastic",
            sampling_mode="full",
            inference_algorithm="VANILLA",
            num_iter=16,
            dF_tol=0.01,
        )
        if no_ig:
            agent.use_states_info_gain = False
        return agent

    max_steps_per_scenario = 1000
    horizon = max_steps_per_scenario * MACRO_MAX_PRIMITIVE_STEPS + 10

    env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": horizon})
    if verbose:
        print("[Env] Using multi-agent env: layout=cramped_room")

    agent_0 = create_agent(seed=48, ego_agent_index=0)
    agent_1 = create_agent(seed=49, ego_agent_index=1)

    if verbose:
        n_pol = len(getattr(agent_0, "policies", []))
        print(
            "[Policies] joint pairs: horizon={} semantic_actions={} n_policies(per agent)={}".format(
                PAIR_POLICY_HORIZON,
                N_SEMANTIC_ACTIONS,
                n_pol,
            )
        )

    def _semantic_idx_to_macro_params(semantic_idx: int):
        dst, mode = model_init_agent.semantic_action_from_index(int(semantic_idx))
        target_w, target_ori = model_init_agent.SEMANTIC_DEST_TARGET_POSE[dst]
        terminal_prim = model_init_agent.INTERACT if mode == "interact" else model_init_agent.STAY
        return int(target_w), int(target_ori), str(mode), int(terminal_prim)

    def _choose_nav_primitive(state, agent_idx: int, target_w: int, target_ori: int) -> int:
        obs_nav = env_utils.env_obs_to_model_obs(state, agent_idx, reward_info=None)
        cur_w = int(obs_nav["self_pos_obs"])
        cur_ori = int(obs_nav["self_orientation_obs"])

        if cur_w == target_w and cur_ori != target_ori:
            if 0 <= target_ori < 4:
                return int(target_ori)
            return int(model_init_agent.STAY)

        if cur_w != target_w:
            cur_grid = model_init_agent.walkable_idx_to_grid_idx(cur_w)
            tgt_grid = model_init_agent.walkable_idx_to_grid_idx(target_w)
            cur_x, cur_y = model_init_agent.index_to_xy(cur_grid)
            tgt_x, tgt_y = model_init_agent.index_to_xy(tgt_grid)
            dx = tgt_x - cur_x
            dy = tgt_y - cur_y

            if abs(dx) >= abs(dy):
                if dx > 0:
                    cand = int(model_init_agent.EAST)
                    nx, ny = cur_x + 1, cur_y
                elif dx < 0:
                    cand = int(model_init_agent.WEST)
                    nx, ny = cur_x - 1, cur_y
                else:
                    cand = int(model_init_agent.STAY)
                    nx, ny = cur_x, cur_y
            else:
                if dy > 0:
                    cand = int(model_init_agent.SOUTH)
                    nx, ny = cur_x, cur_y + 1
                elif dy < 0:
                    cand = int(model_init_agent.NORTH)
                    nx, ny = cur_x, cur_y - 1
                else:
                    cand = int(model_init_agent.STAY)
                    nx, ny = cur_x, cur_y

            cand_grid = model_init_agent.xy_to_index(nx, ny)
            new_walkable = model_init_agent.grid_idx_to_walkable_idx(cand_grid)
            if new_walkable is None:
                return int(model_init_agent.STAY)
            return cand

        return int(model_init_agent.STAY)

    def _execute_semantic_individual_macro_step(env, state, a0_sem, a1_sem, prev_reward_info):
        """
        Execute decentralized semantic choices in the real env.

        IMPORTANT:
        This function does NOT write anything into agent.action.
        It only realizes the environment consequences.
        """
        t0_w, t0_ori, _m0, terminal0 = _semantic_idx_to_macro_params(int(a0_sem))
        t1_w, t1_ori, _m1, terminal1 = _semantic_idx_to_macro_params(int(a1_sem))

        done0 = False
        done1 = False
        r0_sum = 0.0
        r1_sum = 0.0

        terminated_out = {"__all__": False}
        truncated_out = {"__all__": False}

        for _sub in range(MACRO_MAX_PRIMITIVE_STEPS):
            obs_nav0 = env_utils.env_obs_to_model_obs(state, 0, reward_info=None)
            obs_nav1 = env_utils.env_obs_to_model_obs(state, 1, reward_info=None)

            cur_w0 = int(obs_nav0["self_pos_obs"])
            cur_ori0 = int(obs_nav0["self_orientation_obs"])
            cur_w1 = int(obs_nav1["self_pos_obs"])
            cur_ori1 = int(obs_nav1["self_orientation_obs"])

            reached0 = (cur_w0 == t0_w) and (cur_ori0 == t0_ori)
            reached1 = (cur_w1 == t1_w) and (cur_ori1 == t1_ori)

            if done0:
                action0 = int(model_init_agent.STAY)
            else:
                action0 = terminal0 if reached0 else _choose_nav_primitive(state, 0, t0_w, t0_ori)

            if done1:
                action1 = int(model_init_agent.STAY)
            else:
                action1 = terminal1 if reached1 else _choose_nav_primitive(state, 1, t1_w, t1_ori)

            observations, rewards, terminated, truncated, infos = env.step(
                {"agent_0": int(action0), "agent_1": int(action1)}
            )

            prev_reward_info = {
                "sparse_reward_by_agent": [
                    infos["agent_0"].get("sparse_reward", 0),
                    infos["agent_1"].get("sparse_reward", 0),
                ]
            }

            state = infos["agent_0"]["state"]
            r0_sum += float(rewards["agent_0"])
            r1_sum += float(rewards["agent_1"])

            terminated_out = terminated
            truncated_out = truncated

            # mark done only after terminal primitive has been issued at target
            if not done0 and reached0:
                done0 = True
            if not done1 and reached1:
                done1 = True

            if terminated_out.get("__all__") or truncated_out.get("__all__"):
                break
            if done0 and done1:
                break

        return state, prev_reward_info, r0_sum, r1_sum, terminated_out, truncated_out

    def run_one_episode(episode_name, seed=None):
        obs, infos = env.reset(seed=seed)
        state = infos["agent_0"]["state"]

        config_0 = env_utils.get_D_config_from_state(state, 0)
        config_1 = env_utils.get_D_config_from_state(state, 1)
        agent_0.reset(config=config_0)
        agent_1.reset(config=config_1)

        prev_reward_info = {"sparse_reward_by_agent": [0, 0]}
        total_reward_0 = 0.0
        total_reward_1 = 0.0

        if verbose:
            print("\n" + "=" * 72)
            print("  {}".format(episode_name))
            print("=" * 72)

        for step in range(1, max_steps_per_scenario + 1):
            # perception
            obs_0 = env_utils.env_obs_to_model_obs(state, 0, reward_info=prev_reward_info)
            obs_1 = env_utils.env_obs_to_model_obs(state, 1, reward_info=prev_reward_info)

            if noprint:
                print(step, flush=True)
            else:
                state_str = _state_summary(state, model_init_agent, max_agents=2)
                print("\n  --- Step {} ---".format(step))
                print("    Env state:  {}".format(state_str))
                print("    Map (before action):")
                for row in render_overcooked_grid(state, model_init_agent):
                    print("      " + row)
                for line in _agent_summary_lines(state, model_init_agent, max_agents=2):
                    print(line)
                print(
                    "    Obs A0: self_pos={} self_ori={} self_held={} other_pos={} other_held={} pot={} delivered={}".format(
                        obs_0["self_pos_obs"],
                        obs_0["self_orientation_obs"],
                        obs_0["self_held_obs"],
                        obs_0["other_pos_obs"],
                        obs_0["other_held_obs"],
                        obs_0["pot_state_obs"],
                        obs_0["soup_delivered_obs"],
                    )
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
                    )
                )

            # inference
            agent_0.infer_states(obs_0)
            agent_1.infer_states(obs_1)

            agent_0.infer_policies()
            agent_1.infer_policies()

            # action selection + internal time update
            pol_idx_0, joint_first_0, pol_idx_1, joint_first_1 = _independent_joint_first_steps(agent_0, agent_1)

            # decentralized external execution:
            # A0 executes only its own part from its chosen joint pair
            # A1 executes only its own part from its chosen joint pair
            a0_sem_executed = int(joint_first_0[1])
            a1_sem_executed = int(joint_first_1[2])

            state, prev_reward_info, r0, r1, terminated, truncated = _execute_semantic_individual_macro_step(
                env, state, a0_sem_executed, a1_sem_executed, prev_reward_info
            )

            total_reward_0 += float(r0)
            total_reward_1 += float(r1)

            if verbose:
                qs_0 = agent_0.get_state_beliefs()
                print(_belief_table(np, qs_0, model_init_agent, title="Beliefs A0"))

                qs_1 = agent_1.get_state_beliefs()
                print(_belief_table(np, qs_1, model_init_agent, title="Beliefs A1"))

                print("    Policy beliefs A0:")
                q_pi_0 = agent_0.get_policy_posterior()
                H_pi_0 = float(-np.sum(q_pi_0 * np.log(q_pi_0 + 1e-16)))
                top_0 = agent_0.get_top_policies(top_k=5)
                print("      entropy {:.3f}:".format(H_pi_0))
                if top_0:
                    best_pol, best_prob, best_idx = top_0[0]
                    print(
                        "      BEST: idx={} p={:.3f}  {}".format(
                            int(best_idx),
                            float(best_prob),
                            _fmt_policy(best_pol, model_init_agent),
                        )
                    )
                for rank, (pol, prob, _pidx) in enumerate(top_0, 1):
                    pol_str = _fmt_policy(pol, model_init_agent)
                    bar = "█" * int(prob * 20)
                    print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

                print("    Policy beliefs A1:")
                q_pi_1 = agent_1.get_policy_posterior()
                H_pi_1 = float(-np.sum(q_pi_1 * np.log(q_pi_1 + 1e-16)))
                top_1 = agent_1.get_top_policies(top_k=5)
                print("      entropy {:.3f}:".format(H_pi_1))
                if top_1:
                    best_pol, best_prob, best_idx = top_1[0]
                    print(
                        "      BEST: idx={} p={:.3f}  {}".format(
                            int(best_idx),
                            float(best_prob),
                            _fmt_policy(best_pol, model_init_agent),
                        )
                    )
                for rank, (pol, prob, _pidx) in enumerate(top_1, 1):
                    pol_str = _fmt_policy(pol, model_init_agent)
                    bar = "█" * int(prob * 20)
                    print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

                print(
                    "    A0 selected policy idx={}  first joint step: {}".format(
                        int(pol_idx_0),
                        _fmt_pair_step(joint_first_0, model_init_agent),
                    )
                )
                print(
                    "    A1 selected policy idx={}  first joint step: {}".format(
                        int(pol_idx_1),
                        _fmt_pair_step(joint_first_1, model_init_agent),
                    )
                )

                print(
                    "    Decentralized executed semantic actions: A0={}  A1={}".format(
                        model_init_agent.ACTION_NAMES.get(a0_sem_executed, str(a0_sem_executed)),
                        model_init_agent.ACTION_NAMES.get(a1_sem_executed, str(a1_sem_executed)),
                    )
                )

                print("    Reward A0: {}  (cumulative: {})".format(r0, total_reward_0))
                print("    Reward A1: {}  (cumulative: {})".format(r1, total_reward_1))

            if terminated.get("__all__") or truncated.get("__all__"):
                if verbose:
                    print("    Episode ended.")
                break

        if verbose:
            print("\n  Scenario total reward A0: {}".format(total_reward_0))
            print("  Scenario total reward A1: {}".format(total_reward_1))

        return total_reward_0, total_reward_1

    run_one_episode(
        "IndividuallyCollectiveWithSemanticPolicies: two agents, cramped_room (seed=76)",
        seed=76,
    )

    if verbose:
        print("\n" + "=" * 72)
        print("  Two-agent run finished.")
        print("=" * 72)


if __name__ == "__main__":
    run_agent_vs_env_scenarios()