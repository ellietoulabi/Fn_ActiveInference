import argparse
import sys
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

overcooked_src = PROJECT_ROOT / "environments" / "overcooked_ai" / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))

from utils.visualization.overcooked_terminal_map import orientation_str, render_overcooked_grid
from agents.ActiveInferenceWithDynamicPolicies import utils as dyn_utils
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollectiveWithSemanticPoliciesActionLevel.model_init import (
    PRIMITIVE_POLICY_STEP,
)


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

_ORI_NAMES = {0: "N", 1: "S", 2: "E", 3: "W"}
_HELD_NAMES = {0: "NONE", 1: "ONION", 2: "DISH", 3: "SOUP"}
_POT_NAMES = {0: "P0", 1: "P1", 2: "P2", 3: "P3(READY)"}
_CTR_NAMES = {0: "EMPTY", 1: "ONION", 2: "DISH", 3: "SOUP"}

# utils.py uses 0=NORTH,1=SOUTH,2=WEST,3=EAST,4=STAY,5=INTERACT
# env/model here uses 0=NORTH,1=SOUTH,2=EAST,3=WEST,4=STAY,5=INTERACT
_UTILS_TO_ENV_ACTION = {
    0: 0,  # NORTH
    1: 1,  # SOUTH
    2: 3,  # WEST
    3: 2,  # EAST
    4: 4,  # STAY
    5: 5,  # INTERACT
}


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


def _env_primitive_from_policy_step(step) -> int:
    """Map stored policy step to env primitive index (unwraps B_fn marker tuples)."""
    if isinstance(step, (tuple, list)) and len(step) >= 2 and step[0] == PRIMITIVE_POLICY_STEP:
        return int(step[1])
    return int(step)


def _fmt_policy(pol) -> str:
    return "→".join(
        [PRIMITIVE_ACTION_NAMES.get(_env_primitive_from_policy_step(a), str(a)) for a in pol]
    )


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

    # Posterior can pick up small negatives / NaNs from EFE numerics; sanitize before sampling.
    q_pi = np.nan_to_num(q_pi, nan=0.0, posinf=0.0, neginf=0.0)
    q_pi = np.maximum(q_pi, 0.0)
    zq = float(np.sum(q_pi))
    if zq <= 0.0 or not np.isfinite(zq):
        q_pi = np.ones(n_pol, dtype=np.float64) / float(n_pol)
    else:
        q_pi = q_pi / zq

    if getattr(agent, "action_selection", "stochastic") == "deterministic":
        return int(np.argmax(q_pi))

    alpha = float(getattr(agent, "alpha", 1.0))
    log_q = np.log(np.maximum(q_pi, 1e-16))
    p_policies = np.exp(log_q * alpha)
    p_policies = np.maximum(p_policies, 0.0)
    p_policies = np.nan_to_num(p_policies, nan=0.0, posinf=0.0, neginf=0.0)

    s = float(np.sum(p_policies))
    if s <= 0.0 or not np.isfinite(s):
        p_policies = np.ones(n_pol, dtype=np.float64) / float(n_pol)
    else:
        p_policies = p_policies / s

    # Avoid the old "last bin = 1 - sum(rest)" trick: it can go negative when q_pi or sums drift.
    p_policies = np.clip(p_policies, 0.0, 1.0)
    s2 = float(np.sum(p_policies))
    if s2 <= 0.0 or not np.isfinite(s2):
        p_policies = np.ones(n_pol, dtype=np.float64) / float(n_pol)
    else:
        p_policies = p_policies / s2

    return int(np.random.choice(n_pol, p=p_policies))


def _translate_policy_utils_to_env(policy):
    return [_UTILS_TO_ENV_ACTION[int(a)] for a in policy]


def _translate_facing_utils_to_env(facing):
    if facing is None:
        return None
    return _UTILS_TO_ENV_ACTION[int(facing)]


def _translate_agent_policies_utils_to_env(agent):
    """
    utils.py currently generates primitive policies in its own primitive ordering.
    Translate them to the env/model primitive ordering before infer_policies().
    """
    if agent.policies:
        agent.policies = [_translate_policy_utils_to_env(pol) for pol in agent.policies]

    if getattr(agent, "policy_metadata", None):
        translated = []
        for m in agent.policy_metadata:
            mc = dict(m)
            mc["path"] = _translate_policy_utils_to_env(mc.get("path", []))
            mc["actions"] = _translate_policy_utils_to_env(mc.get("actions", []))
            mc["required_facing"] = _translate_facing_utils_to_env(mc.get("required_facing", None))
            mc["final_facing"] = _translate_facing_utils_to_env(mc.get("final_facing", None))
            translated.append(mc)
        agent.policy_metadata = translated


def _wrap_policies_for_primitive_B_rollout(agent):
    """
    infer_policies() rolls policies forward through B_fn. Scalar steps 0..5 collide with
    semantic indices 0..N_ACTIONS-1; wrap each primitive as (PRIMITIVE_POLICY_STEP, a)
    so B_fn uses primitive physics (B_fn_primitive_step) instead of semantic teleport.
    """
    if not agent.policies:
        return
    agent.policies = [
        [(PRIMITIVE_POLICY_STEP, int(a)) for a in pol] for pol in agent.policies
    ]


def _extract_counter_contents_from_state(state):
    """
    Build cntr1..cntr5 contents from the Overcooked raw state.
    """
    counter_contents = {name: "empty" for name in dyn_utils.COUNTER_TILES.keys()}

    rc_to_counter_name = {rc: name for name, rc in dyn_utils.COUNTER_TILES.items()}

    for pos_xy, obj in state.objects.items():
        if obj is None:
            continue
        obj_name = getattr(obj, "name", None)
        if obj_name not in {"onion", "dish", "soup"}:
            continue

        rc = (int(pos_xy[1]), int(pos_xy[0]))  # xy -> (row, col)
        cntr_name = rc_to_counter_name.get(rc, None)
        if cntr_name is not None:
            counter_contents[cntr_name] = str(obj_name).lower()

    return counter_contents


def _extract_pot_status_from_state(state):
    """
    Extract pot_state, pot_onions, soup_ready from raw Overcooked state.
    """
    pot_rc = dyn_utils.DESTINATION_TO_TILE["pot"]
    pot_xy = (int(pot_rc[1]), int(pot_rc[0]))  # (row,col) -> xy

    pot_state = "empty"
    pot_onions = 0
    ready = False

    obj = state.objects.get(pot_xy, None)
    if obj is None or getattr(obj, "name", None) != "soup":
        return pot_state, pot_onions, ready

    ingredients = getattr(obj, "ingredients", []) or []
    n = int(len(ingredients))
    is_idle = bool(getattr(obj, "is_idle", False))
    is_cooking = bool(getattr(obj, "is_cooking", False))
    is_ready = bool(getattr(obj, "is_ready", False))

    pot_onions = n
    ready = is_ready

    if is_ready:
        pot_state = "ready"
    elif is_cooking:
        pot_state = "cooking"
    elif n <= 0:
        pot_state = "empty"
    elif n == 1:
        pot_state = "one_onion"
    elif n == 2:
        pot_state = "two_onions"
    else:
        pot_state = "three_onions"

    if is_idle and n == 0:
        pot_state = "empty"

    return pot_state, pot_onions, ready


def build_policy_state_for_agent(state, agent_idx: int, env_utils, prev_reward_info=None):
    """
    Build the normalized policy-state dict expected by utils.generate_policies_from_state(...).
    """
    other_idx = 1 - int(agent_idx)

    obs_self = env_utils.env_obs_to_model_obs(state, agent_idx, reward_info=prev_reward_info)
    obs_other = env_utils.env_obs_to_model_obs(state, other_idx, reward_info=prev_reward_info)

    pot_state, pot_onions, ready = _extract_pot_status_from_state(state)
    counter_contents = _extract_counter_contents_from_state(state)

    ori_name = {
        0: "NORTH",
        1: "SOUTH",
        2: "EAST",
        3: "WEST",
    }

    held_name = {
        0: "nothing",
        1: "onion",
        2: "dish",
        3: "soup",
    }

    return dyn_utils.build_policy_state(
        self_pos=int(obs_self["self_pos_obs"]),
        self_orient=ori_name[int(obs_self["self_orientation_obs"])],
        self_held=held_name[int(obs_self["self_held_obs"])],
        other_pos=int(obs_other["self_pos_obs"]),
        other_orient=ori_name[int(obs_other["self_orientation_obs"])],
        other_held=held_name[int(obs_other["self_held_obs"])],
        pot_state=pot_state,
        pot_onions=int(pot_onions),
        soup_ready=bool(ready),
        counter_contents=counter_contents,
    )


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
        from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollectiveWithSemanticPoliciesActionLevel import (
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

    N_SEMANTIC_ACTIONS = len(dyn_utils.DESTINATIONS) * len(dyn_utils.MODES)
    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    base_env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}
    policy_len = PAIR_POLICY_HORIZON

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
            actions=list(range(N_PRIMITIVE_ACTIONS)),
            gamma=4.0,
            alpha=8.0,
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

    max_steps_per_scenario = 2000
    horizon = max_steps_per_scenario + 10

    env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": horizon})
    if verbose:
        print("[Env] Using multi-agent env: layout=cramped_room")

    agent_0 = create_agent(seed=48, ego_agent_index=0)
    agent_1 = create_agent(seed=49, ego_agent_index=1)

    if verbose:
        print(
            "[Policies] dynamic semantic policies per agent: destinations={} modes={} total={}".format(
                len(dyn_utils.DESTINATIONS),
                len(dyn_utils.MODES),
                N_SEMANTIC_ACTIONS,
            )
        )

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

    def run_one_episode(episode_name, seed=None):
        _obs, infos = env.reset(seed=seed)
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

            # 1) real-time state inference from actual env observations
            agent_0.infer_states(obs_0)
            agent_1.infer_states(obs_1)

            # 2) build policy-generation state from current env state
            policy_state_0 = build_policy_state_for_agent(state, agent_idx=0, env_utils=env_utils, prev_reward_info=prev_reward_info)
            policy_state_1 = build_policy_state_for_agent(state, agent_idx=1, env_utils=env_utils, prev_reward_info=prev_reward_info)

            # 3) generate current-step primitive policies from semantic library
            agent_0.update_policies(policy_state_0)
            agent_1.update_policies(policy_state_1)

            # translate utils primitive ordering -> env/model primitive ordering
            _translate_agent_policies_utils_to_env(agent_0)
            _translate_agent_policies_utils_to_env(agent_1)

            _wrap_policies_for_primitive_B_rollout(agent_0)
            _wrap_policies_for_primitive_B_rollout(agent_1)

            # 4) policy inference over the current-step generated policies
            agent_0.infer_policies()
            agent_1.infer_policies()

            # 5) each agent independently selects one current primitive policy
            pol_idx_0 = _sample_or_argmax_policy_index(agent_0)
            pol_idx_1 = _sample_or_argmax_policy_index(agent_1)

            pol_0 = agent_0.policies[pol_idx_0]
            pol_1 = agent_1.policies[pol_idx_1]

            meta_0 = agent_0.get_policy_metadata()[pol_idx_0] if agent_0.get_policy_metadata() else None
            meta_1 = agent_1.get_policy_metadata()[pol_idx_1] if agent_1.get_policy_metadata() else None

            a0_prim = (
                _env_primitive_from_policy_step(pol_0[0])
                if len(pol_0) > 0
                else int(model_init_agent.STAY)
            )
            a1_prim = (
                _env_primitive_from_policy_step(pol_1[0])
                if len(pol_1) > 0
                else int(model_init_agent.STAY)
            )

            # For infer_states(..., use_action_for_state_inference=True), B_fn must see a
            # PRIMITIVE_POLICY_STEP tuple — bare ints 0..5 collide with semantic indices 0..21.
            # Ego frame: (marker, self_primitive, other_primitive); partner order swaps for A1.
            agent_0.action = (PRIMITIVE_POLICY_STEP, int(a0_prim), int(a1_prim))
            agent_0.step_time()
            agent_1.action = (PRIMITIVE_POLICY_STEP, int(a1_prim), int(a0_prim))
            agent_1.step_time()

            _observations, state, prev_reward_info, rewards, terminated, truncated, infos = _execute_first_primitive_step(
                env, a0_prim, a1_prim
            )

            r0 = float(rewards["agent_0"])
            r1 = float(rewards["agent_1"])
            total_reward_0 += r0
            total_reward_1 += r1

            if verbose:
                print(
                    "    Generated policies: A0={}  A1={}".format(
                        len(agent_0.policies or []),
                        len(agent_1.policies or []),
                    )
                )

                qs_0 = agent_0.get_state_beliefs()
                print(_belief_table(np, qs_0, model_init_agent, title="Beliefs A0"))

                qs_1 = agent_1.get_state_beliefs()
                print(_belief_table(np, qs_1, model_init_agent, title="Beliefs A1"))

                print("    Policy beliefs A0:")
                q_pi_0 = np.asarray(agent_0.get_policy_posterior(), dtype=float)
                H_pi_0 = float(-np.sum(q_pi_0 * np.log(q_pi_0 + 1e-16)))
                top_0 = agent_0.get_top_policies(top_k=5)
                print("      entropy {:.3f}:".format(H_pi_0))
                if top_0:
                    best_pol, best_prob, best_idx = top_0[0]
                    print(
                        "      BEST: idx={} p={:.3f}  {}".format(
                            int(best_idx),
                            float(best_prob),
                            _fmt_policy(best_pol),
                        )
                    )
                for rank, (pol, prob, _pidx) in enumerate(top_0, 1):
                    pol_str = _fmt_policy(pol)
                    bar = "█" * int(float(prob) * 20)
                    print("        #{:d} [{:>24}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

                print("    Policy beliefs A1:")
                q_pi_1 = np.asarray(agent_1.get_policy_posterior(), dtype=float)
                H_pi_1 = float(-np.sum(q_pi_1 * np.log(q_pi_1 + 1e-16)))
                top_1 = agent_1.get_top_policies(top_k=5)
                print("      entropy {:.3f}:".format(H_pi_1))
                if top_1:
                    best_pol, best_prob, best_idx = top_1[0]
                    print(
                        "      BEST: idx={} p={:.3f}  {}".format(
                            int(best_idx),
                            float(best_prob),
                            _fmt_policy(best_pol),
                        )
                    )
                for rank, (pol, prob, _pidx) in enumerate(top_1, 1):
                    pol_str = _fmt_policy(pol)
                    bar = "█" * int(float(prob) * 20)
                    print("        #{:d} [{:>24}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

                if meta_0 is not None:
                    print(
                        "    A0 selected policy idx={}  semantic=({}, {})".format(
                            int(pol_idx_0),
                            meta_0["destination"],
                            meta_0["mode"],
                        )
                    )
                else:
                    print("    A0 selected policy idx={}".format(int(pol_idx_0)))

                if meta_1 is not None:
                    print(
                        "    A1 selected policy idx={}  semantic=({}, {})".format(
                            int(pol_idx_1),
                            meta_1["destination"],
                            meta_1["mode"],
                        )
                    )
                else:
                    print("    A1 selected policy idx={}".format(int(pol_idx_1)))

                print("    Primitive plan A0: {}".format(_fmt_policy(pol_0)))
                print("    Primitive plan A1: {}".format(_fmt_policy(pol_1)))

                print(
                    "    Executed primitive actions: A0={}  A1={}".format(
                        PRIMITIVE_ACTION_NAMES.get(a0_prim, str(a0_prim)),
                        PRIMITIVE_ACTION_NAMES.get(a1_prim, str(a1_prim)),
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