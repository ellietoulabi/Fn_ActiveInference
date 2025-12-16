"""
Run two Active Inference agents in the TwoAgentRedBlueButton environment under different paradigms:

1) Independent:
   - Each agent runs its own single-agent model and chooses only its own action.

2) FullyCollective:
   - A single central planner chooses a JOINT action (a1, a2) each step.

3) IndividuallyCollective:
   - Each agent plans over JOINT actions, but executes only its own component:
       a1 := first component from agent1's chosen joint action
       a2 := second component from agent2's chosen joint action

This script can run one mode (`--mode independent|fully_collective|individually_collective`)
or all three (`--mode all`) and logs per-step data to CSV.

NOTE: This script does NOT modify agents/ActiveInference.
"""

import os
import sys
from pathlib import Path
import csv
from datetime import datetime
import argparse
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from environments.RedBlueButton.TwoAgentRedBlueButton import TwoAgentRedBlueButtonEnv
from agents.ActiveInference.agent import Agent


ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PRESS", 5: "NOOP"}


def generate_random_config(rng, grid_width=3, grid_height=3):
    """Random button positions avoiding agent starts (0,0) and (2,2)."""
    available = []
    for x in range(grid_width):
        for y in range(grid_height):
            if (x, y) not in [(0, 0), (grid_width - 1, grid_height - 1)]:
                available.append((x, y))
    rng.shuffle(available)
    return {"red_pos": available[0], "blue_pos": available[1]}


def _make_env(config, max_steps):
    return TwoAgentRedBlueButtonEnv(
        width=3,
        height=3,
        red_button_pos=config["red_pos"],
        blue_button_pos=config["blue_pos"],
        agent1_start_pos=(0, 0),
        agent2_start_pos=(2, 2),
        max_steps=max_steps,
    )


def _make_agent_from_model_pkg(model_pkg, env_params, actions, config, keep_factors=None):
    """Construct an ActiveInference Agent using A/B/C/D from a model package."""
    state_factors = list(model_pkg.model_init.states.keys())
    state_sizes = {f: len(v) for f, v in model_pkg.model_init.states.items()}
    agent = Agent(
        A_fn=model_pkg.A_fn,
        B_fn=model_pkg.B_fn,
        C_fn=model_pkg.C_fn,
        D_fn=model_pkg.D_fn,
        state_factors=state_factors,
        state_sizes=state_sizes,
        observation_labels=model_pkg.model_init.observations,
        env_params=env_params,
        observation_state_dependencies=model_pkg.model_init.observation_state_dependencies,
        actions=actions,
        # Keep defaults unless you pass overrides via argparse in the future
        policy_len=3,
        gamma=2.0,
        alpha=1.0,
        num_iter=16,
    )
    agent.reset(config=config, keep_factors=keep_factors)
    return agent


def run_episode_independent(env, model_pkg, agent1, agent2, episode_num, max_steps, csv_writer=None, seed=None, config_idx=None):
    obs, _ = env.reset()
    episode_reward = 0.0

    for step in range(1, max_steps + 1):
        obs1 = model_pkg.env_utils.env_obs_to_model_obs(obs, agent_id=1, width=env.width)
        obs2 = model_pkg.env_utils.env_obs_to_model_obs(obs, agent_id=2, width=env.width)

        a1 = int(agent1.step(obs1))
        a2 = int(agent2.step(obs2))

        grid = env.render(mode="silent")
        map_str = "|".join(["".join(row) for row in grid])

        obs, reward, terminated, truncated, info = env.step((a1, a2))
        done = terminated or truncated
        episode_reward += reward

        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "paradigm": "independent",
                    "seed": seed,
                    "config_idx": config_idx,
                    "episode": episode_num,
                    "step": step,
                    "action1_exec": a1,
                    "action2_exec": a2,
                    "action1_exec_name": ACTION_NAMES.get(a1, str(a1)),
                    "action2_exec_name": ACTION_NAMES.get(a2, str(a2)),
                    "action1_plan": "",
                    "action2_plan": "",
                    "map": map_str,
                    "reward": reward,
                    "result": info.get("result", "neutral"),
                    "button_pressed": info.get("button_just_pressed", ""),
                    "pressed_by": info.get("button_pressed_by", ""),
                }
            )

        if done:
            return {"success": info.get("result") == "win", "steps": step, "reward": episode_reward, "outcome": info.get("result", "neutral")}

    return {"success": False, "steps": max_steps, "reward": episode_reward, "outcome": "timeout"}


def run_episode_fully_collective(env, model_pkg, central_agent, episode_num, max_steps, csv_writer=None, seed=None, config_idx=None):
    obs, _ = env.reset()
    episode_reward = 0.0

    for step in range(1, max_steps + 1):
        obs_joint = model_pkg.env_utils.env_obs_to_model_obs(obs, width=env.width)
        a_joint = int(central_agent.step(obs_joint))
        a1, a2 = model_pkg.env_utils.model_action_to_env_action(a_joint)
        a1, a2 = int(a1), int(a2)

        grid = env.render(mode="silent")
        map_str = "|".join(["".join(row) for row in grid])

        obs, reward, terminated, truncated, info = env.step((a1, a2))
        done = terminated or truncated
        episode_reward += reward

        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "paradigm": "fully_collective",
                    "seed": seed,
                    "config_idx": config_idx,
                    "episode": episode_num,
                    "step": step,
                    "action1_exec": a1,
                    "action2_exec": a2,
                    "action1_exec_name": ACTION_NAMES.get(a1, str(a1)),
                    "action2_exec_name": ACTION_NAMES.get(a2, str(a2)),
                    "action1_plan": a_joint,  # store encoded joint action
                    "action2_plan": a_joint,
                    "map": map_str,
                    "reward": reward,
                    "result": info.get("result", "neutral"),
                    "button_pressed": info.get("button_just_pressed", ""),
                    "pressed_by": info.get("button_pressed_by", ""),
                }
            )

        if done:
            return {"success": info.get("result") == "win", "steps": step, "reward": episode_reward, "outcome": info.get("result", "neutral")}

    return {"success": False, "steps": max_steps, "reward": episode_reward, "outcome": "timeout"}


def run_episode_individually_collective(env, model_pkg, agent1, agent2, episode_num, max_steps, csv_writer=None, seed=None, config_idx=None):
    obs, _ = env.reset()
    episode_reward = 0.0

    for step in range(1, max_steps + 1):
        obs_joint = model_pkg.env_utils.env_obs_to_model_obs(obs, width=env.width)

        # Each agent plans a JOINT action
        joint1 = int(agent1.step(obs_joint))
        joint2 = int(agent2.step(obs_joint))

        # Each executes only its own component
        a1, _ = model_pkg.env_utils.decode_joint_action(joint1)
        _, a2 = model_pkg.env_utils.decode_joint_action(joint2)
        a1, a2 = int(a1), int(a2)

        grid = env.render(mode="silent")
        map_str = "|".join(["".join(row) for row in grid])

        obs, reward, terminated, truncated, info = env.step((a1, a2))
        done = terminated or truncated
        episode_reward += reward

        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "paradigm": "individually_collective",
                    "seed": seed,
                    "config_idx": config_idx,
                    "episode": episode_num,
                    "step": step,
                    "action1_exec": a1,
                    "action2_exec": a2,
                    "action1_exec_name": ACTION_NAMES.get(a1, str(a1)),
                    "action2_exec_name": ACTION_NAMES.get(a2, str(a2)),
                    "action1_plan": joint1,
                    "action2_plan": joint2,
                    "map": map_str,
                    "reward": reward,
                    "result": info.get("result", "neutral"),
                    "button_pressed": info.get("button_just_pressed", ""),
                    "pressed_by": info.get("button_pressed_by", ""),
                }
            )

        if done:
            return {"success": info.get("result") == "win", "steps": step, "reward": episode_reward, "outcome": info.get("result", "neutral")}

    return {"success": False, "steps": max_steps, "reward": episode_reward, "outcome": "timeout"}


def run_one_paradigm(mode, seeds, episodes, episodes_per_config, max_steps, out_csv_path):
    rng_global = np.random.default_rng(0)

    # Import model packages
    if mode == "independent":
        from generative_models.MA_ActiveInference.RedBlueButton import Independent as model_pkg
        action_space = list(range(6))
    elif mode == "fully_collective":
        from generative_models.MA_ActiveInference.RedBlueButton import FullyCollective as model_pkg
        action_space = list(range(model_pkg.model_init.N_JOINT_ACTIONS))
    elif mode == "individually_collective":
        from generative_models.MA_ActiveInference.RedBlueButton import IndividuallyCollective as model_pkg
        action_space = list(range(model_pkg.model_init.N_JOINT_ACTIONS))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # CSV setup
    out_csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "paradigm",
                "seed",
                "config_idx",
                "episode",
                "step",
                "action1_exec",
                "action2_exec",
                "action1_exec_name",
                "action2_exec_name",
                "action1_plan",
                "action2_plan",
                "map",
                "reward",
                "result",
                "button_pressed",
                "pressed_by",
            ],
        )
        writer.writeheader()

        all_results = []

        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            np.random.seed(seed)

            num_configs = (episodes + episodes_per_config - 1) // episodes_per_config
            configs = [generate_random_config(rng) for _ in range(num_configs)]

            env = None

            # Agents (created per config)
            agent1 = None
            agent2 = None
            central_agent = None

            for ep in range(1, episodes + 1):
                config_idx = (ep - 1) // episodes_per_config
                config = configs[config_idx]

                if (ep - 1) % episodes_per_config == 0 or env is None:
                    env = _make_env(config, max_steps=max_steps)

                    # Make env_params consistent (Agent passes these to B_fn)
                    env_params = {"width": env.width, "height": env.height}

                    if mode == "independent":
                        d1 = model_pkg.env_utils.get_D_config_from_env(env, agent_id=1)
                        d2 = model_pkg.env_utils.get_D_config_from_env(env, agent_id=2)
                        agent1 = _make_agent_from_model_pkg(model_pkg, env_params, action_space, d1, keep_factors=["red_button_pos", "blue_button_pos"])
                        agent2 = _make_agent_from_model_pkg(model_pkg, env_params, action_space, d2, keep_factors=["red_button_pos", "blue_button_pos"])

                    elif mode == "fully_collective":
                        d = model_pkg.env_utils.get_D_config_from_env(env)
                        central_agent = _make_agent_from_model_pkg(model_pkg, env_params, action_space, d, keep_factors=["red_button_pos", "blue_button_pos"])

                    elif mode == "individually_collective":
                        d = model_pkg.env_utils.get_D_config_from_env(env)
                        agent1 = _make_agent_from_model_pkg(model_pkg, env_params, action_space, d, keep_factors=["red_button_pos", "blue_button_pos"])
                        agent2 = _make_agent_from_model_pkg(model_pkg, env_params, action_space, d, keep_factors=["red_button_pos", "blue_button_pos"])

                # Episode run
                if mode == "independent":
                    res = run_episode_independent(env, model_pkg, agent1, agent2, ep, max_steps, writer, seed=seed, config_idx=config_idx)
                elif mode == "fully_collective":
                    res = run_episode_fully_collective(env, model_pkg, central_agent, ep, max_steps, writer, seed=seed, config_idx=config_idx)
                else:
                    res = run_episode_individually_collective(env, model_pkg, agent1, agent2, ep, max_steps, writer, seed=seed, config_idx=config_idx)

                all_results.append(res)

                if ep % 100 == 0:
                    recent = all_results[-100:]
                    recent_wins = sum(1 for r in recent if r["success"])
                    print(f"  [{mode}] seed={seed} ep={ep}/{episodes} last100 winrate={recent_wins/100:.2f}")

        # Summary
        total = len(all_results)
        wins = sum(1 for r in all_results if r["success"])
        avg_steps = float(np.mean([r["steps"] for r in all_results])) if total else 0.0
        avg_reward = float(np.mean([r["reward"] for r in all_results])) if total else 0.0

    return {"mode": mode, "episodes": total, "wins": wins, "win_rate": wins / max(1, total), "avg_steps": avg_steps, "avg_reward": avg_reward}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["all", "independent", "fully_collective", "individually_collective"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episodes-per-config", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    modes = ["independent", "fully_collective", "individually_collective"] if args.mode == "all" else [args.mode]

    summaries = []
    for mode in modes:
        out_csv = log_dir / f"two_aif_{mode}_seeds{args.seeds}_ep{args.episodes}_cfg{args.episodes_per_config}_{timestamp}.csv"
        print("=" * 80)
        print(f"RUNNING MODE: {mode}")
        print(f"Logging: {out_csv}")
        print("=" * 80)
        summaries.append(
            run_one_paradigm(
                mode=mode,
                seeds=args.seeds,
                episodes=args.episodes,
                episodes_per_config=args.episodes_per_config,
                max_steps=args.max_steps,
                out_csv_path=out_csv,
            )
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for s in summaries:
        print(
            f"{s['mode']:<22} win_rate={s['win_rate']*100:5.1f}% "
            f"({s['wins']}/{s['episodes']})  avg_steps={s['avg_steps']:.1f}  avg_reward={s['avg_reward']:+.2f}"
        )


if __name__ == "__main__":
    main()


