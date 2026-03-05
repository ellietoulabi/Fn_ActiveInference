"""
Run a single Monotonic IndividuallyCollective Active Inference agent in the real Overcooked
single-agent layout (cramped_room_single) with step-by-step logs.

Uses OvercookedSingleAgentEnv and cramped_room_single.layout so there is only one player;
no dummy agent.

Run from project root: python run_individually_collective.py  (or python3 run_individually_collective.py)
"""

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

from utils.visualization.overcooked_terminal_map import orientation_str
from utils.visualization.overcooked_terminal_map_single import render_overcooked_grid_single


ACTION_NAMES = {
    0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST", 4: "STAY", 5: "INTERACT",
}


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
    Run one Monotonic IndividuallyCollective AIF agent in the Overcooked cramped_room_single env
    (single-player layout; no dummy).
    """
    try:
        import numpy as np
    except ImportError:
        print("\n[SKIP] Agent vs env: numpy not available.")
        return
    try:
        from agents.ActiveInference.agent import Agent
        from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndividuallyCollective import (
            A_fn,
            B_fn,
            C_fn,
            D_fn,
            model_init as mon_model_init,
            env_utils,
        )
        try:
            from environments.overcooked_single_agent_gym import OvercookedSingleAgentEnv
            SingleAgentEnv = OvercookedSingleAgentEnv
            single_agent_layout = "cramped_room_single"
        except Exception:
            from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
            SingleAgentEnv = None
            single_agent_layout = None
    except Exception as e:
        print("\n[SKIP] Agent vs env: could not load agent or environment: {}".format(e))
        return

    model_init_agent = mon_model_init
    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}

    def create_agent(seed=None):
        if seed is not None:
            np.random.seed(seed)
        return Agent(
            A_fn=A_fn,
            B_fn=B_fn,
            C_fn=C_fn,
            D_fn=D_fn,
            state_factors=state_factors,
            state_sizes=state_sizes,
            observation_labels=observation_labels,
            env_params=env_params,
            observation_state_dependencies=model_init_agent.observation_state_dependencies,
            actions=list(range(model_init_agent.N_ACTIONS)),
            gamma=4.0,
            alpha=16.0,
            policy_len=3,
            inference_horizon=3,
            action_selection="deterministic",
            sampling_mode="full",
            inference_algorithm="VANILLA",
            num_iter=16,
            dF_tol=0.001,
        )

    max_steps_per_scenario = 50
    horizon = max_steps_per_scenario + 10

    if SingleAgentEnv is not None and single_agent_layout is not None:
        try:
            env = SingleAgentEnv(config={"layout": single_agent_layout, "horizon": horizon})
            use_single_env = True
            print("[Env] Using single-agent env: layout={}".format(single_agent_layout))
        except Exception as e:
            from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
            env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": horizon})
            use_single_env = False
            print("[Env] Single-agent env failed ({}), using two-agent cramped_room with dummy.".format(e))
    else:
        from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
        env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": horizon})
        use_single_env = False

    agent = create_agent(seed=42)
    DUMMY_ACTION = 4  # STAY (only used when use_single_env is False)

    def run_one_episode(episode_name, seed=None):
        obs, infos = env.reset(seed=seed)
        state = infos["agent_0"]["state"]
        # Force env player to start at walkable index 0 (so map and obs match prior)
        grid_idx = model_init_agent.WALKABLE_INDICES[0]
        x, y = model_init_agent.index_to_xy(grid_idx)
        state.players[0].update_pos_and_or((x, y), state.players[0].orientation)
        config = env_utils.get_D_config_from_state(state, 0)
        # Force agent prior to start at walkable index 0
        config["self_start_pos"] = 0
        agent.reset(config=config)
        prev_reward_info = {"sparse_reward_by_agent": [0] if use_single_env else [0, 0]}
        total_reward = 0.0

        print("\n" + "=" * 72)
        print("  {}".format(episode_name))
        print("=" * 72)

        for step in range(1, max_steps_per_scenario + 1):
            state_str = _state_summary(state, model_init_agent, max_agents=1)
            obs_display = env_utils.env_obs_to_model_obs(state, 0, reward_info=prev_reward_info)
            print("\n  --- Step {} ---".format(step))
            print("    Env state:  {}".format(state_str))
            print("    Map (before action):")
            for row in render_overcooked_grid_single(state, model_init_agent):
                print("      " + row)
            for line in _agent_summary_lines(state, model_init_agent, max_agents=1):
                print(line)
            print("    Obs (model): self_pos_obs={} self_orientation_obs={} self_held_obs={} pot_state_obs={} soup_delivered_obs={}".format(
                obs_display["self_pos_obs"], obs_display["self_orientation_obs"],
                obs_display["self_held_obs"], obs_display["pot_state_obs"],
                obs_display["soup_delivered_obs"]))

            action = int(agent.step(obs_display))
            if use_single_env:
                observations, rewards, terminated, truncated, infos = env.step(action)
                prev_reward_info = {"sparse_reward_by_agent": [infos["agent_0"].get("sparse_reward", 0)]}
            else:
                observations, rewards, terminated, truncated, infos = env.step({"agent_0": action, "agent_1": DUMMY_ACTION})
                prev_reward_info = {"sparse_reward_by_agent": [infos["agent_0"].get("sparse_reward", 0), infos["agent_1"].get("sparse_reward", 0)]}
            state = infos["agent_0"]["state"]
            r = rewards["agent_0"]
            total_reward += r

            qs = agent.get_state_beliefs()
            print("    Beliefs over states:")
            for factor in state_factors:
                p = qs[factor]
                map_idx = int(np.argmax(p))
                max_p = float(np.max(p))
                H = float(-np.sum(p * np.log(p + 1e-16)))
                print("      {} {} (p={:.2f}, H={:.2f})".format(factor, map_idx, max_p, H))

            print("    Policy beliefs:")
            q_pi = agent.get_policy_posterior()
            H_pi = float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
            top = agent.get_top_policies(top_k=5)
            print("      entropy {:.3f}:".format(H_pi))
            for rank, (pol, prob, pol_idx) in enumerate(top, 1):
                pol_str = "→".join([ACTION_NAMES.get(int(a), str(a))[0] for a in pol])
                bar = "█" * int(prob * 20)
                print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

            print("    Action:     {} [{}]".format(ACTION_NAMES.get(action, action), action))
            print("    Reward:    {}  (cumulative: {})".format(r, total_reward))

            if terminated.get("__all__") or truncated.get("__all__"):
                print("    Episode ended.")
                break

        print("\n  Scenario total reward: {}".format(total_reward))
        return total_reward

    run_one_episode("IndividuallyCollective: single agent, cramped_room_single (seed=76)", seed=76)

    print("\n" + "=" * 72)
    print("  Single-agent run finished.")
    print("=" * 72)


if __name__ == "__main__":
    run_agent_vs_env_scenarios()
