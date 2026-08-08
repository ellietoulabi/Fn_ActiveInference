"""
Run two Monotonic IndividuallyCollective Active Inference agents in Overcooked cramped_room
with step-by-step logs.

Uses OvercookedMultiAgentEnv so both agents are controlled by AIF.

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

from utils.visualization.overcooked_terminal_map import orientation_str, render_overcooked_grid


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
    Run two Monotonic IndividuallyCollective AIF agents in the Overcooked cramped_room env.
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
        from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
    except Exception as e:
        print("\n[SKIP] Agent vs env: could not load agent or environment: {}".format(e))
        return

    model_init_agent = mon_model_init
    state_factors = list(model_init_agent.states.keys())
    state_sizes = {f: len(v) for f, v in model_init_agent.states.items()}
    observation_labels = model_init_agent.observations
    env_params = {"width": model_init_agent.GRID_WIDTH, "height": model_init_agent.GRID_HEIGHT}

    # Agent params tuned for speed.
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
            gamma = 1.5,
            alpha = 2.0,
            policy_len=3,
            inference_horizon=3,
            action_selection="stochastic",
            sampling_mode="full",
            inference_algorithm="VANILLA",
            num_iter=16,
            dF_tol=0.01,
        )

    max_steps_per_scenario = 2000
    horizon = max_steps_per_scenario + 10

    env = OvercookedMultiAgentEnv(config={"layout": "cramped_room", "horizon": horizon})
    print("[Env] Using multi-agent env: layout=cramped_room")

    agent_0 = create_agent(seed=48)
    agent_1 = create_agent(seed=49)

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

        print("\n" + "=" * 72)
        print("  {}".format(episode_name))
        print("=" * 72)

        for step in range(1, max_steps_per_scenario + 1):
            state_str = _state_summary(state, model_init_agent, max_agents=2)
            obs_0 = env_utils.env_obs_to_model_obs(state, 0, reward_info=prev_reward_info)
            obs_1 = env_utils.env_obs_to_model_obs(state, 1, reward_info=prev_reward_info)
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

            action_0 = int(agent_0.step(obs_0))
            action_1 = int(agent_1.step(obs_1))
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

            qs_0 = agent_0.get_state_beliefs()
            print("    Beliefs A0:")
            for factor in state_factors:
                p = qs_0[factor]
                map_idx = int(np.argmax(p))
                max_p = float(np.max(p))
                H = float(-np.sum(p * np.log(p + 1e-16)))
                print("      {} {} (p={:.2f}, H={:.2f})".format(factor, map_idx, max_p, H))
            qs_1 = agent_1.get_state_beliefs()
            print("    Beliefs A1:")
            for factor in state_factors:
                p = qs_1[factor]
                map_idx = int(np.argmax(p))
                max_p = float(np.max(p))
                H = float(-np.sum(p * np.log(p + 1e-16)))
                print("      {} {} (p={:.2f}, H={:.2f})".format(factor, map_idx, max_p, H))

            print("    Policy beliefs A0:")
            q_pi_0 = agent_0.get_policy_posterior()
            H_pi_0 = float(-np.sum(q_pi_0 * np.log(q_pi_0 + 1e-16)))
            top_0 = agent_0.get_top_policies(top_k=5)
            print("      entropy {:.3f}:".format(H_pi_0))
            for rank, (pol, prob, pol_idx) in enumerate(top_0, 1):
                pol_str = "→".join([ACTION_NAMES.get(int(a), str(a))[0] for a in pol])
                bar = "█" * int(prob * 20)
                print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))
            print("    Policy beliefs A1:")
            q_pi_1 = agent_1.get_policy_posterior()
            H_pi_1 = float(-np.sum(q_pi_1 * np.log(q_pi_1 + 1e-16)))
            top_1 = agent_1.get_top_policies(top_k=5)
            print("      entropy {:.3f}:".format(H_pi_1))
            for rank, (pol, prob, pol_idx) in enumerate(top_1, 1):
                pol_str = "→".join([ACTION_NAMES.get(int(a), str(a))[0] for a in pol])
                bar = "█" * int(prob * 20)
                print("        #{:d} [{:>8}] {:<20} {:.3f}".format(rank, pol_str, bar, float(prob)))

            print("    Action A0: {} [{}]".format(ACTION_NAMES.get(action_0, action_0), action_0))
            print("    Action A1: {} [{}]".format(ACTION_NAMES.get(action_1, action_1), action_1))
            print("    Reward A0: {}  (cumulative: {})".format(r0, total_reward_0))
            print("    Reward A1: {}  (cumulative: {})".format(r1, total_reward_1))

            if terminated.get("__all__") or truncated.get("__all__"):
                print("    Episode ended.")
                break

        print("\n  Scenario total reward A0: {}".format(total_reward_0))
        print("  Scenario total reward A1: {}".format(total_reward_1))
        return total_reward_0, total_reward_1

    run_one_episode("IndividuallyCollective: two agents, cramped_room (seed=76)", seed=76)

    print("\n" + "=" * 72)
    print("  Two-agent run finished.")
    print("=" * 72)


if __name__ == "__main__":
    run_agent_vs_env_scenarios()
