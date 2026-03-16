"""
Run two Monotonic IndividuallyCollective Active Inference agents in Overcooked cramped_room
with step-by-step logs.

Uses OvercookedMultiAgentEnv so both agents are controlled by AIF.

Run from project root: python run_individually_collective.py  (or python3 run_individually_collective.py)
"""

import argparse
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Add overcooked_ai src so we can load the MDP
overcooked_src = PROJECT_ROOT / "environments" / "overcooked_ai" / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))


# -----------------------------------------------------------------------------
# Single agent vs environment: IndividuallyCollective AIF agent in Overcooked
# -----------------------------------------------------------------------------

from utils.visualization.overcooked_terminal_map import orientation_str, render_overcooked_grid


ACTION_NAMES = {
    0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST", 4: "STAY", 5: "INTERACT",
}

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


def _fmt_step_action(model_init, step_action: int) -> str:
    """
    For interleaved step-actions (0..11): decode to ACTOR:PRIMITIVE.
    Falls back to primitive action name if decode not available.
    """
    try:
        actor, a = model_init.decode_interleaved_step(int(step_action))
        actor_s = getattr(model_init, "ACTOR_NAMES", {}).get(int(actor), str(actor))
        return "{}:{}".format(actor_s, ACTION_NAMES.get(int(a), str(a)))
    except Exception:
        return ACTION_NAMES.get(int(step_action), str(step_action))


def _fmt_policy(model_init, pol) -> str:
    return "→".join([_fmt_step_action(model_init, int(a)) for a in pol])


def _step_action_to_primitive(model_init, step_action: int) -> int:
    """
    Execute only SELF-assigned primitives; OTHER-assigned means STAY (can't directly control OTHER).
    """
    try:
        actor, a = model_init.decode_interleaved_step(int(step_action))
        return int(a) if int(actor) == int(model_init.SELF) else int(model_init.STAY)
    except Exception:
        return int(step_action)


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

def _construct_ego_first_policies(model_init, policy_len: int):
    """
    Ego-first interleaved policies:
    - step 1 must be SELF (0..N_ACTIONS-1)
    - steps 2..T can be SELF or OTHER (0..N_INTERLEAVED_STEP_ACTIONS-1)
    """
    n_actions = int(getattr(model_init, "N_ACTIONS", 6))
    n_inter = int(getattr(model_init, "N_INTERLEAVED_STEP_ACTIONS", 2 * n_actions))

    step1 = list(range(n_actions))          # SELF:*  (actor=0)
    later = list(range(n_inter))            # SELF:* or OTHER:*

    if policy_len <= 1:
        return [[a] for a in step1]

    # Explicit loops (fast enough for len=3; 6*12*12=864)
    policies = []
    if policy_len == 2:
        for a1 in step1:
            for a2 in later:
                policies.append([a1, a2])
        return policies

    if policy_len == 3:
        for a1 in step1:
            for a2 in later:
                for a3 in later:
                    policies.append([a1, a2, a3])
        return policies

    # Generic (small horizons only)
    from itertools import product
    for a1 in step1:
        for rest in product(later, repeat=policy_len - 1):
            policies.append([a1, *rest])
    return policies

def _first_step_is_other(model_init, policy) -> bool:
    """
    True iff the first step decodes to actor OTHER.
    If decode isn't available, assume it's not-other.
    """
    try:
        actor, _ = model_init.decode_interleaved_step(int(policy[0]))
        return int(actor) == int(model_init.OTHER)
    except Exception:
        return False


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
        "--fullpolicies",
        action="store_true",
        help="If set, use the full interleaved policy set (12^T). Default is ego-first (6*12^(T-1)).",
    )
    parser.add_argument(
        "--noig",
        action="store_true",
        help="Disable epistemic value (state information gain) in policy evaluation.",
    )
    args, _unknown = parser.parse_known_args()
    noprint = bool(args.noprint)
    verbose = not noprint
    ego_first = not bool(args.fullpolicies)
    no_ig = bool(args.noig)

    try:
        import numpy as np
    except ImportError:
        if verbose:
            print("\n[SKIP] Agent vs env: numpy not available.")
        return
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
    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}
    policy_len = 3

    # Agent params tuned for speed.
    def create_agent(seed=None):
        if seed is not None:
            np.random.seed(seed)
        policies = None
        if ego_first:
            policies = _construct_ego_first_policies(model_init_agent, policy_len=policy_len)
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
            actions=list(range(getattr(model_init_agent, "N_INTERLEAVED_STEP_ACTIONS", model_init_agent.N_ACTIONS))),
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

    agent_0 = create_agent(seed=48)
    agent_1 = create_agent(seed=49)
    if verbose:
        print(
            "[Policies] n_policies(A0)={} n_policies(A1)={} policy_len={} actions={} ego_first={}".format(
                len(getattr(agent_0, "policies", [])),
                len(getattr(agent_1, "policies", [])),
                int(getattr(agent_0, "policy_len", 0)),
                len(getattr(agent_0, "actions", [])),
                ego_first,
            )
        )

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

            action_0_step = int(agent_0.step(obs_0))
            action_1_step = int(agent_1.step(obs_1))
            action_0 = _step_action_to_primitive(model_init_agent, action_0_step)
            action_1 = _step_action_to_primitive(model_init_agent, action_1_step)
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
                    print("      BEST: idx={} p={:.3f}  {}".format(int(best_idx), float(best_prob), _fmt_policy(model_init_agent, best_pol)))
                    if _first_step_is_other(model_init_agent, best_pol):
                        print("      NOTE: best policy starts with OTHER ⇒ executed action will be STAY.")
                for rank, (pol, prob, pol_idx) in enumerate(top_0, 1):
                    pol_str = _fmt_policy(model_init_agent, pol)
                    bar = "█" * int(prob * 20)
                    print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

                print("    Policy beliefs A1:")
                q_pi_1 = agent_1.get_policy_posterior()
                H_pi_1 = float(-np.sum(q_pi_1 * np.log(q_pi_1 + 1e-16)))
                top_1 = agent_1.get_top_policies(top_k=5)
                print("      entropy {:.3f}:".format(H_pi_1))
                if top_1:
                    best_pol, best_prob, best_idx = top_1[0]
                    print("      BEST: idx={} p={:.3f}  {}".format(int(best_idx), float(best_prob), _fmt_policy(model_init_agent, best_pol)))
                    if _first_step_is_other(model_init_agent, best_pol):
                        print("      NOTE: best policy starts with OTHER ⇒ executed action will be STAY.")
                for rank, (pol, prob, pol_idx) in enumerate(top_1, 1):
                    pol_str = _fmt_policy(model_init_agent, pol)
                    bar = "█" * int(prob * 20)
                    print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

                print("    Action A0(step): {} [{}]".format(_fmt_step_action(model_init_agent, action_0_step), action_0_step))
                print("    Action A1(step): {} [{}]".format(_fmt_step_action(model_init_agent, action_1_step), action_1_step))
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
