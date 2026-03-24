

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
# Two-agent Overcooked sim: IndividuallyCollective AIF with joint pair policies
#
# Each planning step is (JOINT_PAIR_LABEL, a0, a1): env agent 0 executes a0, agent 1 a1.
# Each ego's B_fn maps that to (self_action, other_action) via ego_agent_index in env_params.
# Both agents share one policy index per env step (sample/argmax from q_pi0 * q_pi1).
# -----------------------------------------------------------------------------

from utils.visualization.overcooked_terminal_map import orientation_str, render_overcooked_grid


ACTION_NAMES = {
    0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST", 4: "STAY", 5: "INTERACT",
}

# Joint pair policies: two planning steps; each step assigns one primitive per agent.
N_PRIMITIVE_ACTIONS = 6
PAIR_POLICY_HORIZON = 2

_ORI_NAMES = {0: "N", 1: "S", 2: "E", 3: "W"}
_HELD_NAMES = {0: "NONE", 1: "ONION", 2: "DISH", 3: "SOUP"}
_POT_NAMES = {0: "P0", 1: "P1", 2: "P2", 3: "P3(READY)"}
_CTR_NAMES = {0: "EMPTY", 1: "ONION", 2: "DISH", 3: "SOUP"}


def _fmt_prob(x: float) -> str:
    return "{:>5.2f}".format(float(x))


def _map_stats(np, p):
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


def _fmt_pair_step(step_action) -> str:
    """Format one joint policy step (label, a0, a1) with six primitive names."""
    if isinstance(step_action, (tuple, list)) and len(step_action) == 3:
        a0 = int(step_action[1])
        a1 = int(step_action[2])
        return "A0:{}|A1:{}".format(
            ACTION_NAMES.get(a0, str(a0)),
            ACTION_NAMES.get(a1, str(a1)),
        )
    return str(step_action)


def _fmt_policy(pol) -> str:
    return "→".join([_fmt_pair_step(a) for a in pol])


def _pair_step_action_to_primitive(step_action, agent_idx: int, model_init) -> int:
    """
    Global joint step (JOINT_PAIR_LABEL, a0, a1): return primitive for env agent_idx.
    Falls back to STAY if malformed.
    """
    try:
        if isinstance(step_action, (tuple, list)) and len(step_action) == 3:
            return int(step_action[1 + int(agent_idx)])
    except Exception:
        pass
    return int(getattr(model_init, "STAY", 4))


def _construct_joint_pair_policies(joint_label: str):
    """All sequences of PAIR_POLICY_HORIZON steps; each step is (joint_label, a0, a1)."""
    from itertools import product

    step_pairs = [
        (joint_label, a0, a1)
        for a0 in range(N_PRIMITIVE_ACTIONS)
        for a1 in range(N_PRIMITIVE_ACTIONS)
    ]
    return [[*steps] for steps in product(step_pairs, repeat=PAIR_POLICY_HORIZON)]


def _belief_table(np, qs: dict, model_init, title: str) -> str:
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
        i, pm, H = _map_stats(np, p)
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
        i, pm, H = _map_stats(np, p)
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
            i, pm, _H = _map_stats(np, p)
            p_empty = float(p[0]) if p.size > 0 else 1.0
            p_nonempty = float(1.0 - p_empty)
            v = _CTR_NAMES.get(i, str(i))
            lines.append("      {:<19}  {:<18}  p(nonempty)={}".format(k, v, _fmt_prob(p_nonempty)))

    return "\n".join(lines)


def _state_summary(state, model_init, max_agents: int | None = None):
    """One-line summary of env state: positions (walkable), held, pot."""
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
    """Return list of lines: per-agent position (walkable), holding, facing."""
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


def run_agent_vs_env_scenarios():
    """
    Run two Monotonic IndividuallyCollective AIF agents in the Overcooked cramped_room env.
    """
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
        from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollective import (
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
    if int(getattr(model_init_agent, "N_ACTIONS", 0)) != N_PRIMITIVE_ACTIONS:
        if verbose:
            print(
                "\n[SKIP] model_init N_ACTIONS ({}) != script N_PRIMITIVE_ACTIONS ({}).".format(
                    getattr(model_init_agent, "N_ACTIONS", None),
                    N_PRIMITIVE_ACTIONS,
                )
            )
        return
    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    base_env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}
    policy_len = PAIR_POLICY_HORIZON
    joint_label = str(getattr(model_init_agent, "JOINT_PAIR_LABEL", "__joint_pair__"))

    def create_agent(seed=None, ego_agent_index: int = 0):
        if seed is not None:
            np.random.seed(seed)
        policies = _construct_joint_pair_policies(joint_label)
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
            actions=list(range(N_PRIMITIVE_ACTIONS)),
            gamma = 4.0,
            alpha = 16.0,
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

    max_steps_per_scenario = 50
    horizon = max_steps_per_scenario + 10

    env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": horizon})
    if verbose:
        print("[Env] Using multi-agent env: layout=cramped_room")

    agent_0 = create_agent(seed=48, ego_agent_index=0)
    agent_1 = create_agent(seed=49, ego_agent_index=1)
    if verbose:
        n_pol = len(getattr(agent_0, "policies", []))
        print(
            "[Policies] joint pairs: horizon={} primitives={} n_policies(per agent)={}".format(
                PAIR_POLICY_HORIZON,
                N_PRIMITIVE_ACTIONS,
                n_pol,
            )
        )

    def _coordinated_joint_first_step(agent_0, agent_1):
        """
        Both agents already ran infer_policies with ego-aware B. Choose one policy index
        so they execute the same global joint first step: (label, a0, a1).

        Uses q_pi0 * q_pi1 (renormalized) for stochastic sampling so both assign mass to
        the same joint plans; deterministic uses argmax of the product.
        """
        n_pol = len(agent_0.policies)
        q0 = np.asarray(agent_0.get_policy_posterior(), dtype=np.float64).reshape(-1)
        q1 = np.asarray(agent_1.get_policy_posterior(), dtype=np.float64).reshape(-1)
        q_prod = q0 * q1
        if agent_0.action_selection == "deterministic":
            pol_idx = int(np.argmax(q_prod))
        else:
            alpha = float(agent_0.alpha)
            log_q = np.log(np.maximum(q_prod, 1e-16))
            p_policies = np.exp(log_q * alpha)
            p_policies = np.maximum(p_policies, 0.0)
            s = float(np.sum(p_policies))
            if s <= 0.0 or not np.isfinite(s):
                p_policies = np.ones(n_pol, dtype=np.float64) / float(n_pol)
            else:
                p_policies = p_policies / s
                if len(p_policies) > 1:
                    p_policies[-1] = 1.0 - float(np.sum(p_policies[:-1]))
                else:
                    p_policies[0] = 1.0
            pol_idx = int(np.random.choice(n_pol, p=p_policies))
        joint_step = agent_0.policies[pol_idx][0]
        agent_0.action = joint_step
        agent_0.step_time()
        agent_1.action = joint_step
        agent_1.step_time()
        return pol_idx, joint_step

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
            obs_0 = env_utils.env_obs_to_model_obs(state, 0, reward_info=prev_reward_info)
            obs_1 = env_utils.env_obs_to_model_obs(state, 1, reward_info=prev_reward_info)
            if noprint:
                # Only step number, nothing else.
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
                print("    Obs A0: self_pos={} self_ori={} self_held={} other_pos={} other_held={} pot={} delivered={}".format(
                    obs_0["self_pos_obs"], obs_0["self_orientation_obs"], obs_0["self_held_obs"],
                    obs_0["other_pos_obs"], obs_0["other_held_obs"], obs_0["pot_state_obs"], obs_0["soup_delivered_obs"]))
                print("    Obs A1: self_pos={} self_ori={} self_held={} other_pos={} other_held={} pot={} delivered={}".format(
                    obs_1["self_pos_obs"], obs_1["self_orientation_obs"], obs_1["self_held_obs"],
                    obs_1["other_pos_obs"], obs_1["other_held_obs"], obs_1["pot_state_obs"], obs_1["soup_delivered_obs"]))

            agent_0.infer_states(obs_0)
            agent_1.infer_states(obs_1)
            agent_0.infer_policies()
            agent_1.infer_policies()
            pol_idx, joint_first = _coordinated_joint_first_step(agent_0, agent_1)
            action_0_step = joint_first
            action_1_step = joint_first
            action_0 = _pair_step_action_to_primitive(joint_first, 0, model_init_agent)
            action_1 = _pair_step_action_to_primitive(joint_first, 1, model_init_agent)
            observations, rewards, terminated, truncated, infos = env.step({"agent_0": action_0, "agent_1": action_1})
            prev_reward_info = {
                "sparse_reward_by_agent": [
                    infos["agent_0"].get("sparse_reward", 0),
                    infos["agent_1"].get("sparse_reward", 0),
                ]
            }
            state = infos["agent_0"]["state"]
            r0 = rewards["agent_0"]
            r1 = rewards["agent_1"]
            total_reward_0 += r0
            total_reward_1 += r1

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
                    print("      BEST: idx={} p={:.3f}  {}".format(int(best_idx), float(best_prob), _fmt_policy(best_pol)))
                for rank, (pol, prob, _pidx) in enumerate(top_0, 1):
                    pol_str = _fmt_policy(pol)
                    bar = "█" * int(prob * 20)
                    print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

                print("    Policy beliefs A1:")
                q_pi_1 = agent_1.get_policy_posterior()
                H_pi_1 = float(-np.sum(q_pi_1 * np.log(q_pi_1 + 1e-16)))
                top_1 = agent_1.get_top_policies(top_k=5)
                print("      entropy {:.3f}:".format(H_pi_1))
                if top_1:
                    best_pol, best_prob, best_idx = top_1[0]
                    print("      BEST: idx={} p={:.3f}  {}".format(int(best_idx), float(best_prob), _fmt_policy(best_pol)))
                for rank, (pol, prob, _pidx) in enumerate(top_1, 1):
                    pol_str = _fmt_policy(pol)
                    bar = "█" * int(prob * 20)
                    print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

                print(
                    "    Coordinated policy idx={}  joint first step: {}".format(
                        int(pol_idx),
                        _fmt_pair_step(joint_first),
                    )
                )
                print("    Action A0(step): {} [{}]".format(_fmt_pair_step(action_0_step), action_0_step))
                print("    Action A1(step): {} [{}]".format(_fmt_pair_step(action_1_step), action_1_step))
                print("    Action A0(exec): {} [{}]".format(ACTION_NAMES.get(action_0, action_0), action_0))
                print("    Action A1(exec): {} [{}]".format(ACTION_NAMES.get(action_1, action_1), action_1))
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

    run_one_episode("IndividuallyCollective: two agents, cramped_room (seed=76)", seed=76)

    if verbose:
        print("\n" + "=" * 72)
        print("  Two-agent run finished.")
        print("=" * 72)


if __name__ == "__main__":
    run_agent_vs_env_scenarios()
