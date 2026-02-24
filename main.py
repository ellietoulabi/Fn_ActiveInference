"""
Run Monotonic Independent MA Active Inference agents against the real Overcooked
`cramped_room` environment with step-by-step logs.

Run from project root: python main.py  (or python3 main.py)
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
# Agent vs environment: run Monotonic Independent agent in real Overcooked
# -----------------------------------------------------------------------------

ACTION_NAMES = {
    0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST", 4: "STAY", 5: "INTERACT",
}


def _state_summary(state, model_init):
    """One-line summary of env state: positions (walkable), held, pot."""
    parts = []
    for i, p in enumerate(state.players):
        pos = p.position
        grid_idx = model_init.xy_to_index(pos[0], pos[1])
        w = model_init.grid_idx_to_walkable_idx(grid_idx)
        w = w if w is not None else -1
        held = "none"
        if p.has_object() and p.held_object:
            held = getattr(p.held_object, "name", str(p.held_object))
        parts.append("A{}@{} held={}".format(i, w, held))
    pot = "empty"
    # Only treat soup that is actually in a pot cell as pot contents.
    for (pos, obj) in state.objects.items():
        if not obj or getattr(obj, "name", None) != "soup":
            continue
        grid_idx = model_init.xy_to_index(pos[0], pos[1])
        if grid_idx not in model_init.POT_INDICES:
            # Soup on counters / serving / elsewhere should not be interpreted as pot contents.
            continue

        ing = getattr(obj, "ingredients", [])
        n = len(ing) if ing else 0
        is_idle = bool(getattr(obj, "is_idle", False))
        is_cooking = bool(getattr(obj, "is_cooking", False))
        is_ready = bool(getattr(obj, "is_ready", False))
        tick = getattr(obj, "_cooking_tick", None)
        ct = None
        if not is_idle:
            # In this Overcooked implementation, `cook_time` accesses `recipe.time`,
            # which raises while soup is still idle (recipe undefined).
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


def _render_ascii_map(state, model_init):
    """
    Render a simple ASCII map for cramped_room with current agent positions overlaid.

    Legend:
      X = counter / wall
      P = pot
      S = serving
      O = onion dispenser
      D = dish dispenser
      1,2 = agent positions
      ' ' = walkable floor
    """
    w, h = model_init.GRID_WIDTH, model_init.GRID_HEIGHT
    grid = [[" " for _ in range(w)] for _ in range(h)]

    # Static terrain from model_init
    for idx in model_init.COUNTER_INDICES:
        x, y = model_init.index_to_xy(idx)
        grid[y][x] = "X"
    for idx in model_init.POT_INDICES:
        x, y = model_init.index_to_xy(idx)
        grid[y][x] = "P"
    for idx in model_init.SERVING_INDICES:
        x, y = model_init.index_to_xy(idx)
        grid[y][x] = "S"
    for idx in model_init.ONION_DISPENSER_INDICES:
        x, y = model_init.index_to_xy(idx)
        grid[y][x] = "O"
    for idx in model_init.DISH_DISPENSER_INDICES:
        x, y = model_init.index_to_xy(idx)
        grid[y][x] = "D"

    # Overlay agents
    for i, p in enumerate(state.players):
        x, y = p.position
        ch = "1" if i == 0 else "2"
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = ch

    return ["".join(row) for row in grid]


def run_agent_vs_env_scenarios():
    """
    Run Monotonic Independent AIF agents against the real Overcooked cramped_room env.
    Two short scenarios with step-by-step logs.
    """
    try:
        import numpy as np
    except ImportError:
        print("\n[SKIP] Agent vs env: numpy not available.")
        return
    try:
        from agents.ActiveInference.agent import Agent
        from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent import (
            A_fn,
            B_fn,
            C_fn,
            D_fn,
            model_init as mon_model_init,
            env_utils,
        )
        from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
    except Exception as e:
        print("\n[SKIP] Agent vs env: could not load agent or environment: {}".format(e))
        return

    # Use Monotonic model for agents
    model_init_agent = mon_model_init
    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}

    def create_agent(agent_idx, seed=None):
        if seed is not None:
            np.random.seed(seed + agent_idx)
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
            action_selection="stochastic",
            sampling_mode="full",
            inference_algorithm="VANILLA",
            num_iter=16,
            dF_tol=0.001,
        )

    max_steps_per_scenario = 100
    horizon = max_steps_per_scenario + 10

    env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": horizon})
    agent0 = create_agent(0, seed=42)
    agent1 = create_agent(1, seed=43)

    def run_one_episode(episode_name, seed=None):
        obs, infos = env.reset(seed=seed)
        state = infos["agent_0"]["state"]
        config0 = env_utils.get_D_config_from_state(state, 0)
        config1 = env_utils.get_D_config_from_state(state, 1)
        agent0.reset(config=config0)
        agent1.reset(config=config1)
        prev_reward_info = {"sparse_reward_by_agent": [0, 0]}
        total_reward = 0.0

        print("\n" + "=" * 72)
        print("  {}".format(episode_name))
        print("=" * 72)

        for step in range(1, max_steps_per_scenario + 1):
            # Print state and map before action (what the agents see when deciding)
            state_str = _state_summary(state, model_init_agent)
            obs0_display = env_utils.env_obs_to_model_obs(state, 0, reward_info=prev_reward_info)
            obs1_display = env_utils.env_obs_to_model_obs(state, 1, reward_info=prev_reward_info)
            print("\n  --- Step {} ---".format(step))
            print("    Env state:  {}".format(state_str))
            print("    Map (before action):")
            for row in _render_ascii_map(state, model_init_agent):
                print("      " + row)
            print("    Obs (model) A0: pos_obs={} ori_obs={} held_obs={} pot_obs={} soup_delivered_obs={}".format(
                obs0_display["agent_pos_obs"], obs0_display["agent_orientation_obs"], obs0_display["agent_held_obs"],
                obs0_display["pot_state_obs"], obs0_display["soup_delivered_obs"]))
            print("    Obs (model) A1: pos_obs={} ori_obs={} held_obs={} pot_obs={} soup_delivered_obs={}".format(
                obs1_display["agent_pos_obs"], obs1_display["agent_orientation_obs"], obs1_display["agent_held_obs"],
                obs1_display["pot_state_obs"], obs1_display["soup_delivered_obs"]))

            # Agents choose actions from current obs, then env steps
            obs0 = obs0_display
            obs1 = obs1_display
            action0 = int(agent0.step(obs0))
            action1 = int(agent1.step(obs1))
            actions = {"agent_0": action0, "agent_1": action1}
            observations, rewards, terminated, truncated, infos = env.step(actions)
            state = infos["agent_0"]["state"]
            r = rewards["agent_0"]
            total_reward += r
            prev_reward_info = {
                "sparse_reward_by_agent": [
                    infos["agent_0"].get("sparse_reward", 0),
                    infos["agent_1"].get("sparse_reward", 0),
                ]
            }

            # Beliefs over states
            qs0 = agent0.get_state_beliefs()
            qs1 = agent1.get_state_beliefs()
            print("    Beliefs over states:")
            print("      {:<18} {:<28} {:<28}".format("Factor", "Agent 0", "Agent 1"))
            print("      {:<18} {:<28} {:<28}".format("-" * 18, "-" * 28, "-" * 28))
            for factor in state_factors:
                p0 = qs0[factor]
                p1 = qs1[factor]
                map0 = int(np.argmax(p0))
                map1 = int(np.argmax(p1))
                max0 = float(np.max(p0))
                max1 = float(np.max(p1))
                H0 = float(-np.sum(p0 * np.log(p0 + 1e-16)))
                H1 = float(-np.sum(p1 * np.log(p1 + 1e-16)))
                s0 = f"{map0} (p={max0:.2f}, H={H0:.2f})"
                s1 = f"{map1} (p={max1:.2f}, H={H1:.2f})"
                print("      {:<18} {:<28} {:<28}".format(factor, s0, s1))

            # Beliefs over policies
            print("    Policy beliefs:")
            for idx, agent in enumerate([agent0, agent1]):
                q_pi = agent.get_policy_posterior()
                H_pi = float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
                top = agent.get_top_policies(top_k=3)
                print("      Agent {} (entropy {:.3f}):".format(idx, H_pi))
                for rank, (pol, prob, pol_idx) in enumerate(top, 1):
                    pol_str = "→".join([ACTION_NAMES.get(int(a), str(a))[0] for a in pol])
                    bar = "█" * int(prob * 20)
                    print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

            # Chosen actions and reward
            print("    Actions:    A0 -> {} [{}]  |  A1 -> {} [{}]".format(
                ACTION_NAMES.get(action0, action0), action0,
                ACTION_NAMES.get(action1, action1), action1))
            print("    Reward:    {}  (cumulative: {})".format(r, total_reward))

            if terminated.get("__all__") or truncated.get("__all__"):
                print("    Episode ended.")
                break

        print("\n  Scenario total reward: {}".format(total_reward))
        return total_reward

    # Scenario 1: default start, seed 42
    run_one_episode("Scenario 1: Cramped room, 20 steps (seed=42)", seed=76)
    # Scenario 2: different seed

    print("\n" + "=" * 72)
    print("  Agent-vs-env scenarios finished.")
    print("=" * 72)


if __name__ == "__main__":
    run_agent_vs_env_scenarios()
