"""
Run two OPSRL agents playing TwoAgentRedBlueButton.

Each agent is an independent MAOPSRLAgent (agents/OPSRL/ma_agent.py) -- its
own posterior, own Q-table, own Thompson-sampled exploration -- given the
same ego-relative observation Independent AIF and the MAPPO comparator give
each agent (own position, partner's position, whether *I* am on the
red/blue button, both buttons' pressed state; see
run_two_aif_agents_independent.py / run_two_ppo_agents.py). Both agents act
in the same shared TwoAgentRedBlueButtonEnv and learn from the single team
reward it returns each step -- there is no communication between them and
no joint state/action modeling, matching OPSRL's role in Stage 1 as a
Bayesian model-based RL baseline, extended here to the two-agent setting
rather than compared against it as in Stage 1.

Like the Stage-1 nine-agent OPSRL baseline (compare_nine_agents.py), each
agent's posterior accumulates across the *entire* seed run, including every
button relocation -- OPSRL has no belief-retention/reset mechanism the way
AIF does; it must re-learn stale (state, action) -> outcome statistics after
each relocation from scratch via continued interaction, which is exactly
the H1 comparison point.

Runs the same evaluation protocol (seeds, episodes, episodes_per_config,
configs, CSV/stats schema) as the AIF and PPO two-agent scripts so results
are directly comparable.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
import csv
import json
from datetime import datetime

import numpy as np
from tqdm import tqdm

from environments.RedBlueButton.TwoAgentRedBlueButton import TwoAgentRedBlueButtonEnv
from agents.OPSRL.ma_agent import MAOPSRLAgent


ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PRESS", 5: "NOOP"}


# =============================================================================
# Observation helpers
# =============================================================================

def xy_to_index(x, y, width=3):
    """Convert (x, y) coordinates to flat grid index."""
    return y * width + x


def two_agent_obs_to_own_obs(env_obs, agent_id, width=3):
    """
    Ego-relative observation for one physical agent, matching the factors
    Independent AIF and MAPPO give each agent as their own local view (see
    run_two_aif_agents_independent.py::two_agent_obs_to_single_agent_obs and
    run_two_ppo_agents.py::OWN_OBS_KEYS), reduced to the flat-index/binary
    form MAOPSRLAgent._obs_to_state expects.
    """
    if agent_id == 1:
        position = env_obs["agent1_position"]
        other_position = env_obs["agent2_position"]
        on_red = env_obs["agent1_on_red_button"]
        on_blue = env_obs["agent1_on_blue_button"]
    else:
        position = env_obs["agent2_position"]
        other_position = env_obs["agent1_position"]
        on_red = env_obs["agent2_on_red_button"]
        on_blue = env_obs["agent2_on_blue_button"]

    return {
        "my_pos": xy_to_index(int(position[0]), int(position[1]), width),
        "other_pos": xy_to_index(int(other_position[0]), int(other_position[1]), width),
        "on_red_button": int(on_red),
        "on_blue_button": int(on_blue),
        "red_button_pressed": int(env_obs["red_button_pressed"]),
        "blue_button_pressed": int(env_obs["blue_button_pressed"]),
    }


def env_obs_to_log_obs(env_obs, width=3):
    """Model-style indices for CSV logging -- same field names/semantics as
    the AIF and PPO two-agent scripts' CSV rows, for direct comparability."""
    pos1 = env_obs.get("agent1_position", (0, 0))
    pos2 = env_obs.get("agent2_position", (0, 0))
    return {
        "agent1_pos": xy_to_index(int(pos1[0]), int(pos1[1]), width),
        "agent2_pos": xy_to_index(int(pos2[0]), int(pos2[1]), width),
        "agent1_on_red_button": int(env_obs.get("agent1_on_red_button", 0)),
        "agent1_on_blue_button": int(env_obs.get("agent1_on_blue_button", 0)),
        "agent2_on_red_button": int(env_obs.get("agent2_on_red_button", 0)),
        "agent2_on_blue_button": int(env_obs.get("agent2_on_blue_button", 0)),
        "red_button_state": int(env_obs.get("red_button_pressed", 0)),
        "blue_button_state": int(env_obs.get("blue_button_pressed", 0)),
        "game_result": int(env_obs.get("win_lose_neutral", 0)),
    }


# =============================================================================
# Agent creation
# =============================================================================

def create_opsrl_agent(seed, width=3, height=3, max_steps=50, gamma=0.95,
                        horizon=None, thompson_samples=1,
                        bernoullized_reward=True, scale_prior_reward=1.0,
                        prior_transition="uniform", stage_dependent=False,
                        reward_free=False):
    return MAOPSRLAgent(
        width=width,
        height=height,
        n_actions=6,
        gamma=gamma,
        horizon=horizon if horizon is not None else max_steps,
        bernoullized_reward=bernoullized_reward,
        scale_prior_reward=scale_prior_reward,
        thompson_samples=thompson_samples,
        prior_transition=prior_transition,
        reward_free=reward_free,
        stage_dependent=stage_dependent,
        seed=seed,
    )


# =============================================================================
# Episode / seed loop
# =============================================================================

def run_episode(env, agent1, agent2, episode_num, max_steps=50, verbose=False,
                 csv_writer=None, episode_progress=False, config_idx=0,
                 print_steps=False):
    """Run one episode with two independent OPSRL agents."""
    # Fresh Thompson sample + backward induction for this episode (mirrors
    # OPSRLAgent._run_episode()'s own preamble -- see MAOPSRLAgent.resample_and_plan).
    agent1.resample_and_plan()
    agent2.resample_and_plan()

    obs, _ = env.reset()

    if verbose:
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}")
        print(f"{'='*80}")
        print(f"Environment: Red at {env.red_button}, Blue at {env.blue_button}")
        env.render()

    episode_reward = 0.0
    outcome = "timeout"
    step = 0
    step_iter = range(1, max_steps + 1)
    if episode_progress and not verbose:
        step_iter = tqdm(step_iter, desc=f"Ep {episode_num}", leave=False, position=2, unit="step")

    for step in step_iter:
        own_obs1 = two_agent_obs_to_own_obs(obs, 1, width=env.width)
        own_obs2 = two_agent_obs_to_own_obs(obs, 2, width=env.width)
        state1 = agent1._obs_to_state(own_obs1)
        state2 = agent2._obs_to_state(own_obs2)

        # hh is the elapsed-step index into the finite-horizon backward-
        # induction table -- same convention (and same fix) as
        # compare_nine_agents.py::run_opsrl_episode's `hh = min(step - 1,
        # agent.horizon - 1)`: `step` here is 1-indexed, so `step - 1` gives
        # the correct 0-indexed elapsed-step count, and the min(...) guards
        # against max_steps/horizon ever diverging.
        hh = min(step - 1, agent1.horizon - 1)

        action1 = agent1._get_action(state1, hh=hh)
        action2 = agent2._get_action(state2, hh=hh)

        grid = env.render(mode="silent")
        map_str = "|".join(["".join(row) for row in grid])

        next_obs, reward, terminated, truncated, info = env.step((action1, action2))
        reward = float(reward)
        done = terminated or truncated
        episode_reward += reward

        own_next_obs1 = two_agent_obs_to_own_obs(next_obs, 1, width=env.width) if not done else None
        own_next_obs2 = two_agent_obs_to_own_obs(next_obs, 2, width=env.width) if not done else None
        next_state1 = agent1._obs_to_state(own_next_obs1)
        next_state2 = agent2._obs_to_state(own_next_obs2)

        agent1._update(state1, action1, next_state1, reward, hh=hh)
        agent2._update(state2, action2, next_state2, reward, hh=hh)

        if verbose:
            a1n = ACTION_NAMES.get(action1, str(action1))
            a2n = ACTION_NAMES.get(action2, str(action2))
            print(f"  Step {step}: A1={a1n}(s={state1}) A2={a2n}(s={state2}) "
                  f"r={reward:+.2f} cum={episode_reward:+.2f} {info.get('result', 'neutral')}")
        elif print_steps:
            a1n = ACTION_NAMES.get(action1, str(action1))
            a2n = ACTION_NAMES.get(action2, str(action2))
            print(f"  Ep {episode_num} Step {step}: A1={a1n} A2={a2n} r={reward:+.2f} "
                  f"cum={episode_reward:+.2f} {info.get('result', 'neutral')}")

        if csv_writer is not None:
            log_obs = env_obs_to_log_obs(obs, width=env.width)
            row = {
                "episode": episode_num,
                "step": step,
                "config_idx": config_idx,
                **log_obs,
                "action1": action1,
                "action1_name": ACTION_NAMES.get(action1, str(action1)),
                "action2": action2,
                "action2_name": ACTION_NAMES.get(action2, str(action2)),
                "map": map_str,
                "reward": reward,
                "cumulative_reward": episode_reward,
                "terminated": terminated,
                "truncated": truncated,
                "result": info.get("result", "neutral"),
                "button_pressed": info.get("button_just_pressed", ""),
                "pressed_by": info.get("button_pressed_by", ""),
                "agent1_state": state1,
                "agent2_state": state2,
            }
            csv_writer.writerow(row)

        obs = next_obs
        if done:
            outcome = info.get("result", "neutral")
            break

    agent1.episode += 1
    agent2.episode += 1

    if verbose:
        status = "✅ WIN" if outcome == "win" else "❌ FAIL"
        print(f"\nResult: {status} - {outcome} (steps: {step}, total reward: {episode_reward:+.2f})")

    return {
        "outcome": outcome,
        "reward": episode_reward,
        "steps": step,
        "success": outcome == "win",
    }


def generate_random_config(rng, grid_width=3, grid_height=3):
    """Generate random button positions (avoiding agent start positions)."""
    available_positions = []
    for x in range(grid_width):
        for y in range(grid_height):
            if (x, y) not in [(0, 0), (grid_width - 1, grid_height - 1)]:
                available_positions.append((x, y))

    rng.shuffle(available_positions)
    return {"red_pos": available_positions[0], "blue_pos": available_positions[1]}


def run_seed_experiment(seed, num_episodes, episodes_per_config, max_steps,
                         opsrl_kwargs, verbose=False, csv_writer=None,
                         episode_progress=False, print_steps=False,
                         progress_callback=None):
    """Run experiment for a single seed. Agents persist (accumulate posterior)
    across the WHOLE seed run, including every button relocation -- only the
    environment changes at config boundaries, matching compare_nine_agents.py's
    treatment of the single-agent OPSRL baseline."""
    rng = np.random.default_rng(seed)
    results = []
    configs = []
    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    for _ in range(num_configs):
        configs.append(generate_random_config(rng))

    # Independent RNG streams per agent (same per-agent-seed convention as
    # run_two_ppo_agents.py's torch.Generator seeding) -- without this, both
    # agents' Thompson sampling would draw identically, making their
    # exploration mirror each other rather than being genuinely independent.
    agent1 = create_opsrl_agent(seed=int(seed) * 2 + 1, max_steps=max_steps, **opsrl_kwargs)
    agent2 = create_opsrl_agent(seed=int(seed) * 2 + 2, max_steps=max_steps, **opsrl_kwargs)

    env = None
    for episode in tqdm(range(1, num_episodes + 1), disable=verbose, desc=f"Seed {seed}", leave=True, unit="ep", position=1):
        config_idx = (episode - 1) // episodes_per_config
        config = configs[config_idx]
        if (episode - 1) % episodes_per_config == 0 or env is None:
            env = TwoAgentRedBlueButtonEnv(
                width=3,
                height=3,
                red_button_pos=config["red_pos"],
                blue_button_pos=config["blue_pos"],
                agent1_start_pos=(0, 0),
                agent2_start_pos=(2, 2),
                max_steps=max_steps,
            )
            if verbose:
                print(f"\n{'='*80}")
                print(f"SEED {seed} - CONFIG {config_idx + 1}")
                print(f"Red at {config['red_pos']}, Blue at {config['blue_pos']}")
                print(f"Episodes {episode}-{min(episode + episodes_per_config - 1, num_episodes)}")
                print(f"{'='*80}")

        result = run_episode(
            env, agent1, agent2, episode,
            max_steps=max_steps,
            verbose=verbose,
            csv_writer=csv_writer,
            episode_progress=episode_progress,
            config_idx=config_idx,
            print_steps=print_steps,
        )
        result["seed"] = seed
        result["config_idx"] = config_idx
        results.append(result)
        if progress_callback is not None:
            progress_callback(1)

        if episode % 100 == 0:
            recent = results[-100:]
            recent_wins = sum(1 for r in recent if r["success"])
            recent_rate = 100.0 * recent_wins / max(1, len(recent))
            print(f"  Seed {seed}, Episode {episode}/{num_episodes}: "
                  f"Last 100 win rate: {recent_rate:.1f}% ({recent_wins}/100)")

    return results, configs


def main():
    print("=" * 80)
    print("TWO OPSRL AGENTS - RED BLUE BUTTON ENVIRONMENT")
    print("=" * 80)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run (if --seed not provided)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run (overrides --seeds if provided)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episodes-per-config", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="Print full step info each step")
    parser.add_argument("--print-steps", action="store_true", help="Print one line per step when not using --verbose")
    parser.add_argument("--episode-progress", action="store_true", help="Show per-episode progress over steps (tqdm)")
    parser.add_argument("--stats-output", type=str, default=None, help="Also write stats JSON to this path (for comparison scripts)")

    # OPSRL hyperparameters (defaults match the Stage-1 nine-agent OPSRL
    # baseline in compare_nine_agents.py: gamma=0.95, horizon=max_steps,
    # stage_dependent=False, prior_transition='uniform').
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--horizon", type=int, default=None, help="Defaults to --max-steps if not given")
    parser.add_argument("--thompson-samples", type=int, default=1)
    parser.add_argument("--scale-prior-reward", type=float, default=1.0)
    parser.add_argument("--prior-transition", type=str, choices=["uniform", "optimistic"], default="uniform")
    parser.add_argument("--stage-dependent", action="store_true")
    parser.add_argument("--reward-free", action="store_true")
    parser.add_argument("--no-bernoullized-reward", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        SEEDS_TO_RUN = [args.seed]
        SEED_TAG = f"seed{args.seed}"
    else:
        SEEDS_TO_RUN = list(range(args.seeds))
        SEED_TAG = f"seeds{args.seeds}"

    NUM_SEEDS = len(SEEDS_TO_RUN)
    NUM_EPISODES_PER_SEED = args.episodes
    EPISODES_PER_CONFIG = args.episodes_per_config
    MAX_STEPS = args.max_steps
    VERBOSE = args.verbose

    opsrl_kwargs = dict(
        gamma=args.gamma,
        horizon=args.horizon,
        thompson_samples=args.thompson_samples,
        bernoullized_reward=not args.no_bernoullized_reward,
        scale_prior_reward=args.scale_prior_reward,
        prior_transition=args.prior_transition,
        stage_dependent=args.stage_dependent,
        reward_free=args.reward_free,
    )

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"two_opsrl_agents_{SEED_TAG}_ep{NUM_EPISODES_PER_SEED}_step{MAX_STEPS}_redblue_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    csv_fieldnames = [
        "seed", "episode", "step", "config_idx",
        "agent1_pos", "agent2_pos",
        "agent1_on_red_button", "agent1_on_blue_button",
        "agent2_on_red_button", "agent2_on_blue_button",
        "red_button_state", "blue_button_state", "game_result",
        "action1", "action1_name", "action2", "action2_name",
        "map", "reward", "cumulative_reward", "terminated", "truncated",
        "result", "button_pressed", "pressed_by",
        "agent1_state", "agent2_state",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, extrasaction="ignore")
    csv_writer.writeheader()

    print(f"\nExperiment Parameters:")
    print(f"  Seeds to run: {SEEDS_TO_RUN}")
    print(f"  Episodes per seed: {NUM_EPISODES_PER_SEED}")
    print(f"  Episodes per config: {EPISODES_PER_CONFIG}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  OPSRL: gamma={args.gamma} horizon={args.horizon or MAX_STEPS} "
          f"thompson_samples={args.thompson_samples} prior_transition={args.prior_transition} "
          f"stage_dependent={args.stage_dependent}")
    print(f"  CSV log: {csv_path}")

    all_results = []
    seed_summaries = []
    total_episodes = NUM_SEEDS * NUM_EPISODES_PER_SEED
    with tqdm(total=total_episodes, desc="Total", unit="ep", leave=True, position=0) as pbar:
        try:
            for seed_idx, seed in enumerate(SEEDS_TO_RUN):
                print(f"\n{'='*80}")
                print(f"SEED {seed} ({seed_idx + 1}/{NUM_SEEDS})")
                print(f"{'='*80}")

                class SeedCSVWriter:
                    def __init__(self, writer, seed):
                        self.writer = writer
                        self.seed = seed

                    def writerow(self, row):
                        row["seed"] = self.seed
                        self.writer.writerow(row)

                seed_csv_writer = SeedCSVWriter(csv_writer, seed)

                results, configs = run_seed_experiment(
                    seed=seed,
                    num_episodes=NUM_EPISODES_PER_SEED,
                    episodes_per_config=EPISODES_PER_CONFIG,
                    max_steps=MAX_STEPS,
                    opsrl_kwargs=opsrl_kwargs,
                    verbose=VERBOSE,
                    csv_writer=seed_csv_writer,
                    episode_progress=args.episode_progress,
                    print_steps=args.print_steps,
                    progress_callback=pbar.update,
                )

                all_results.extend(results)

                successes = sum(1 for r in results if r["success"])
                success_rate = 100 * successes / len(results)
                avg_reward = np.mean([r["reward"] for r in results])
                avg_steps = np.mean([r["steps"] for r in results])

                mid = len(results) // 2
                first_half_wins = sum(1 for r in results[:mid] if r["success"])
                second_half_wins = sum(1 for r in results[mid:] if r["success"])

                seed_summaries.append({
                    "seed": seed,
                    "successes": successes,
                    "total": len(results),
                    "success_rate": success_rate,
                    "avg_reward": avg_reward,
                    "avg_steps": avg_steps,
                    "first_half_wins": first_half_wins,
                    "second_half_wins": second_half_wins,
                })

                print(f"\nSeed {seed} Summary:")
                print(f"  Success rate: {successes}/{len(results)} ({success_rate:.1f}%)")
                print(f"  Average reward: {avg_reward:+.2f}")
                print(f"  Average steps: {avg_steps:.1f}")
                print(f"  Learning: First half {first_half_wins}/{mid}, Second half {second_half_wins}/{mid}")
        finally:
            csv_file.close()

    seed_summaries_serializable = [
        {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in s.items()}
        for s in seed_summaries
    ]
    stats = {
        "paradigm": "opsrl",
        "n_seeds": NUM_SEEDS,
        "n_episodes_per_seed": NUM_EPISODES_PER_SEED,
        "episodes_per_config": EPISODES_PER_CONFIG,
        "max_steps": MAX_STEPS,
        "opsrl_hyperparams": {**opsrl_kwargs, "horizon": opsrl_kwargs["horizon"] or MAX_STEPS},
        "total_episodes": len(all_results),
        "total_successes": sum(1 for r in all_results if r["success"]),
        "success_rate": float(100 * sum(1 for r in all_results if r["success"]) / max(1, len(all_results))),
        "mean_reward": float(np.mean([r["reward"] for r in all_results])),
        "std_reward": float(np.std([r["reward"] for r in all_results])),
        "mean_steps": float(np.mean([r["steps"] for r in all_results])),
        "std_steps": float(np.std([r["steps"] for r in all_results])),
        "seed_summaries": seed_summaries_serializable,
    }
    stats_filename = f"two_opsrl_agents_{SEED_TAG}_ep{NUM_EPISODES_PER_SEED}_step{MAX_STEPS}_redblue_{timestamp}_stats.json"
    stats_path = log_dir / stats_filename
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\n✓ Logs saved:")
    print(f"    CSV:   {csv_path}")
    print(f"    Stats: {stats_path}")

    print("\n" + "=" * 80)
    print("OVERALL RESULTS SUMMARY")
    print("=" * 80)

    total_successes = sum(s["successes"] for s in seed_summaries)
    total_episodes_run = sum(s["total"] for s in seed_summaries)
    overall_success_rate = 100 * total_successes / total_episodes_run
    overall_avg_reward = np.mean([r["reward"] for r in all_results])
    overall_avg_steps = np.mean([r["steps"] for r in all_results])

    print(f"\nTotal episodes: {total_episodes_run}")
    print(f"Total successes: {total_successes} ({overall_success_rate:.1f}%)")
    print(f"Overall average reward: {overall_avg_reward:+.2f}")
    print(f"Overall average steps: {overall_avg_steps:.1f}")

    print(f"\n{'Seed':<6} {'Success Rate':<18} {'Avg Reward':<12} {'1st Half':<10} {'2nd Half':<10}")
    print("-" * 65)
    for s in seed_summaries:
        mid = s["total"] // 2
        print(f"{s['seed']:<6} {s['successes']:>4}/{s['total']:<4} ({s['success_rate']:>5.1f}%)  "
              f"{s['avg_reward']:>+7.2f}     "
              f"{s['first_half_wins']:>3}/{mid:<5}   "
              f"{s['second_half_wins']:>3}/{mid:<5}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
