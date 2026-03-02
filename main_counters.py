"""
Run Monotonic Independent MA Active Inference agents against the real Overcooked
`cramped_room` environment with step-by-step logs, INCLUDING counter modalities.

Run from project root:
    python3 main_counters.py
"""

import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Add overcooked_ai src so we can load the MDP
overcooked_src = PROJECT_ROOT / "environments" / "overcooked_ai" / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))

from utils.visualization.overcooked_terminal_map import render_overcooked_grid, orientation_str

ACTION_NAMES = {0: "NORTH", 1: "SOUTH", 2: "EAST", 3: "WEST", 4: "STAY", 5: "INTERACT"}


def _counter_slot_summary(state, model_init):
    """Compact summary of the 5 modeled counter slots from env state."""
    names = {
        model_init.HELD_NONE: "none",
        model_init.HELD_ONION: "onion",
        model_init.HELD_DISH: "dish",
        model_init.HELD_SOUP: "soup",
    }
    parts = []
    for i, grid_idx in enumerate(getattr(model_init, "COUNTER_SLOT_GRID_IDXS", [])):
        x, y = model_init.index_to_xy(int(grid_idx))
        obj = state.objects.get((x, y), None)
        held = model_init.object_name_to_held_type(getattr(obj, "name", None) if obj is not None else None)
        parts.append(f"c{i}@({x},{y})={names.get(held, str(held))}")
    return " | ".join(parts) if parts else "(no counter slots defined)"


def _state_summary(state, model_init):
    """One-line summary of env state: positions (walkable), held, pot, counters."""
    parts = []
    for i, p in enumerate(state.players):
        pos = p.position
        grid_idx = model_init.xy_to_index(pos[0], pos[1])
        w = model_init.grid_idx_to_walkable_idx(grid_idx)
        w = w if w is not None else -1
        held = "none"
        if p.has_object() and p.held_object:
            held = getattr(p.held_object, "name", str(p.held_object))
        parts.append(f"A{i}@{w} held={held}")

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
        if is_ready:
            phase = "ready"
        elif is_cooking:
            phase = "cooking"
        elif is_idle:
            phase = "idle"
        else:
            phase = "unknown"
        pot = f"{n}onion({phase})"
        break

    return " | ".join(parts) + f" | pot={pot} | {_counter_slot_summary(state, model_init)}"


def _agent_summary_lines(state, model_init):
    """Return list of lines: per-agent position (walkable), holding, facing."""
    lines = []
    for i, p in enumerate(state.players):
        pos = p.position
        grid_idx = model_init.xy_to_index(pos[0], pos[1])
        w = model_init.grid_idx_to_walkable_idx(grid_idx)
        w = w if w is not None else -1
        held = "none"
        if p.has_object() and p.held_object:
            held = getattr(p.held_object, "name", str(p.held_object))
        facing, _ = orientation_str(p)
        lines.append(f"    A{i}: pos(walkable)={w}  holding={held}  facing={facing}")
    return lines


def run_agent_vs_env():
    try:
        import numpy as np
    except ImportError:
        print("\n[SKIP] numpy not available")
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
        print(f"\n[SKIP] Could not load agent or environment: {e}")
        return

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

    max_steps = 20
    env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": max_steps + 10})
    agent0 = create_agent(0, seed=42)
    agent1 = create_agent(1, seed=85)

    _, infos = env.reset(seed=76)
    state = infos["agent_0"]["state"]
    agent0.reset(config=env_utils.get_D_config_from_state(state, 0))
    agent1.reset(config=env_utils.get_D_config_from_state(state, 1))
    prev_reward_info = {"sparse_reward_by_agent": [0, 0]}
    total_reward = 0.0

    print("\n" + "=" * 72)
    print("  Scenario: cramped_room with counter obs (seed=76)")
    print("=" * 72)

    for step in range(1, max_steps + 1):
        obs0 = env_utils.env_obs_to_model_obs(state, 0, reward_info=prev_reward_info)
        obs1 = env_utils.env_obs_to_model_obs(state, 1, reward_info=prev_reward_info)

        print(f"\n  --- Step {step} ---")
        print(f"    Env state:  {_state_summary(state, model_init_agent)}")
        print("    Map (before action):")
        for row in render_overcooked_grid(state, model_init_agent):
            print("      " + row)
        for line in _agent_summary_lines(state, model_init_agent):
            print(line)

        def _obs_line(o):
            ctr = [o[f"counter_{i}_obs"] for i in range(5)]
            return (
                f"pos_obs={o['agent_pos_obs']} ori_obs={o['agent_orientation_obs']} "
                f"held_obs={o['agent_held_obs']} pot_obs={o['pot_state_obs']} "
                f"soup_delivered_obs={o['soup_delivered_obs']} counters={ctr}"
            )

        print(f"    Obs (model) A0: {_obs_line(obs0)}")
        print(f"    Obs (model) A1: {_obs_line(obs1)}")

        action0 = int(agent0.step(obs0))
        action1 = int(agent1.step(obs1))
        actions = {"agent_0": action0, "agent_1": action1}
        _, rewards, terminated, truncated, infos = env.step(actions)
        state = infos["agent_0"]["state"]

        r = float(rewards["agent_0"])
        total_reward += r
        prev_reward_info = {
            "sparse_reward_by_agent": [
                infos["agent_0"].get("sparse_reward", 0),
                infos["agent_1"].get("sparse_reward", 0),
            ]
        }

        print(
            "    Actions:    A0 -> {} [{}]  |  A1 -> {} [{}]".format(
                ACTION_NAMES.get(action0, action0),
                action0,
                ACTION_NAMES.get(action1, action1),
                action1,
            )
        )
        print(f"    Reward:    {r}  (cumulative: {total_reward})")

        if terminated.get("__all__") or truncated.get("__all__"):
            print("    Episode ended.")
            break

    print("\n  Scenario total reward: {}".format(total_reward))


if __name__ == "__main__":
    run_agent_vs_env()

