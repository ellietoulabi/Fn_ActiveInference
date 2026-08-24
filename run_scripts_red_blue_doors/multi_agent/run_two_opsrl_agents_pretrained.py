"""
Run two OPSRL agents on TwoAgentRedBlueButton, WARM-STARTED via a
domain-randomized pretraining phase run to convergence, before the real,
scored evaluation protocol begins.

Why this exists (full discussion in ai/02-debug.md, MA Red-Blue-Button):
the cold-start protocol (run_two_opsrl_agents.py) conflates two different
questions -- "can OPSRL learn navigation/press mechanics at all" and "can
OPSRL adapt when the button relocates" -- and only the second is what H1/H2
actually claim to test, since AIF is deployed with a hand-specified,
already-correct model and never has to solve the first problem. This script
isolates the second question: agents are pretrained against a stream of
domain-randomized button configurations (a fresh random layout every
pretraining episode, not blocked into fixed configs, so the posterior is
forced to generalize rather than memorize one layout) until their windowed
win rate plateaus. Only THEN does the real, scored protocol begin --
episodes/episodes-per-config/relocations, logged and reported exactly like
the cold-start script -- so what gets measured is adaptation speed from an
already-warm starting point, with button-location belief for each
evaluation config still genuinely undiscovered (exactly the situation AIF is
also in at the start of every config).

Mirrors the existing --mode pretrained/online precedent in
run_two_ppo_agents.py for the identical underlying fairness question (how
much training does an RL baseline need before a "zero-shot" comparison
against AIF is actually apples-to-apples), and reports the pretraining
budget explicitly (episodes, env steps, whether it truly converged) rather
than hiding it -- see ai/04-writeup.md's MAPPO fairness item 4: "report both
honestly rather than picking a single compute-budget metric that hides one
side of the asymmetry."

Isolation, per explicit direction: this file, and the class it drives
(agents/OPSRL/ma_agent_pretrained_convergence.py::MAOPSRLAgentPretrainedConvergence),
do not edit:
  - agents/OPSRL/agent.py (Stage 1's shared OPSRLAgent)
  - agents/OPSRL/ma_agent.py (the cold-start MAOPSRLAgent)
  - run_two_opsrl_agents.py (the cold-start runner) -- only IMPORTED from for
    generic, protocol-agnostic utilities (config generation, ego-obs
    conversion, CSV field mapping, the per-episode step loop), never edited.
  - any Active Inference agent or script, anywhere.
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
from agents.OPSRL.ma_agent_pretrained_convergence import MAOPSRLAgentPretrainedConvergence

# Reused (imported, never edited) from the cold-start script -- these are
# generic, protocol-agnostic helpers (config generation, ego-relative
# observation conversion, CSV field mapping, and the single-episode step
# loop), not "the OPSRL agent" itself. run_episode() in particular is
# exactly as valid for a pretraining episode as for a scored one -- the only
# difference is which env/config it's pointed at and whether the caller
# counts the result.
from run_scripts_red_blue_doors.multi_agent.run_two_opsrl_agents import (
    ACTION_NAMES,
    generate_random_config,
    two_agent_obs_to_own_obs,
    env_obs_to_log_obs,
    xy_to_index,
    run_episode,
)


# =============================================================================
# Agent creation
# =============================================================================

def create_opsrl_agent_pretrained(seed, width=3, height=3, max_steps=50, gamma=0.95,
                                   horizon=None, thompson_samples=10,
                                   bernoullized_reward=True, scale_prior_reward=1.0,
                                   prior_transition="uniform", stage_dependent=False,
                                   reward_free=False):
    return MAOPSRLAgentPretrainedConvergence(
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
# Pretraining loop
# =============================================================================

def pretrain_pair_to_convergence(agent1, agent2, pretrain_rng, max_steps,
                                  window=50, patience=3, min_delta=0.02,
                                  max_episodes=3000, min_episodes=200,
                                  progress_desc=None):
    """
    Run episodes of two OPSRL agents against a domain-randomized sequence of
    button configurations -- a FRESH random config drawn every pretraining
    episode (not blocked, unlike the scored protocol), so the posterior is
    forced to generalize across layouts rather than memorize one -- until
    the windowed win rate plateaus.

    Convergence check: every `window` episodes, compute the win rate over
    that block; compare consecutive blocks. If the absolute change stays
    below `min_delta` for `patience` consecutive comparisons (and at least
    `min_episodes` total episodes have run, to avoid an early lucky/unlucky
    streak triggering a false-positive stop), pretraining stops.
    `max_episodes` is a hard safety cap -- Thompson sampling can be noisy
    (confirmed directly during earlier debugging: non-monotonic win-rate
    swings are normal, not a bug), so without a cap a bad seed could run
    indefinitely; hitting the cap prints a warning rather than silently
    reporting a false "converged".

    Returns a dict: n_episodes, n_env_steps, converged (bool),
    final_windowed_win_rate, win_rate_windows (list, one per window).
    """
    history = []
    win_rate_windows = []
    total_env_steps = 0
    episode = 0
    plateaued_count = 0
    converged = False

    pbar = tqdm(total=max_episodes, desc=progress_desc or "Pretrain", leave=False, unit="ep")
    try:
        while episode < max_episodes:
            episode += 1
            config = generate_random_config(pretrain_rng)
            env = TwoAgentRedBlueButtonEnv(
                width=3,
                height=3,
                red_button_pos=config["red_pos"],
                blue_button_pos=config["blue_pos"],
                agent1_start_pos=(0, 0),
                agent2_start_pos=(2, 2),
                max_steps=max_steps,
            )
            result = run_episode(env, agent1, agent2, episode, max_steps=max_steps,
                                  verbose=False, csv_writer=None)
            history.append(1 if result["success"] else 0)
            total_env_steps += result["steps"]
            pbar.update(1)

            if episode % window == 0 and episode >= min_episodes:
                recent_wr = sum(history[-window:]) / window
                win_rate_windows.append(recent_wr)
                if len(win_rate_windows) >= 2:
                    delta = abs(win_rate_windows[-1] - win_rate_windows[-2])
                    if delta < min_delta:
                        plateaued_count += 1
                    else:
                        plateaued_count = 0
                    if plateaued_count >= patience:
                        converged = True
                        break
    finally:
        pbar.close()

    if not converged:
        tail = win_rate_windows[-5:] if win_rate_windows else []
        print(f"    [pretrain] WARNING: did not converge within max_episodes={max_episodes}; "
              f"stopping anyway. Last windowed win rates: {tail}")

    return {
        "n_episodes": episode,
        "n_env_steps": total_env_steps,
        "converged": converged,
        "final_windowed_win_rate": win_rate_windows[-1] if win_rate_windows else None,
        "win_rate_windows": win_rate_windows,
    }


# =============================================================================
# Seed loop
# =============================================================================

def run_seed_experiment(seed, num_episodes, episodes_per_config, max_steps,
                         opsrl_kwargs, pretrain_kwargs,
                         verbose=False, csv_writer=None,
                         episode_progress=False, print_steps=False,
                         progress_callback=None):
    """
    1) Pretrain both agents jointly against domain-randomized configs (a
       SEPARATE RNG stream from the scored-evaluation configs -- seeded
       distinctly so pretraining can never accidentally draw the exact
       config sequence evaluation will later score, keeping pretrain/eval
       cleanly separated) until convergence.
    2) Run the standard scored evaluation protocol on the now-warm agents.
       Agents are constructed (and pretrained) ONCE per seed and never
       rebuilt across config boundaries during the scored protocol either --
       matching the 2026-08-20 fix applied to the MA AIF scripts (see
       ai/02-debug.md), so no paradigm anywhere in this comparison gets a
       privileged "the environment just changed" reset.
    """
    eval_rng = np.random.default_rng(seed)
    pretrain_rng = np.random.default_rng(int(seed) * 1_000_003 + 987)

    results = []
    configs = []
    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    for _ in range(num_configs):
        configs.append(generate_random_config(eval_rng))

    agent1 = create_opsrl_agent_pretrained(seed=int(seed) * 2 + 1, max_steps=max_steps, **opsrl_kwargs)
    agent2 = create_opsrl_agent_pretrained(seed=int(seed) * 2 + 2, max_steps=max_steps, **opsrl_kwargs)

    print(f"  Seed {seed}: pretraining (domain-randomized configs) until convergence...")
    pretrain_stats = pretrain_pair_to_convergence(
        agent1, agent2, pretrain_rng, max_steps,
        window=pretrain_kwargs["window"],
        patience=pretrain_kwargs["patience"],
        min_delta=pretrain_kwargs["min_delta"],
        max_episodes=pretrain_kwargs["max_episodes"],
        min_episodes=pretrain_kwargs["min_episodes"],
        progress_desc=f"Pretrain seed{seed}",
    )
    print(f"  Seed {seed}: pretraining done -- {pretrain_stats['n_episodes']} episodes, "
          f"{pretrain_stats['n_env_steps']} env steps, converged={pretrain_stats['converged']}, "
          f"final windowed win rate={pretrain_stats['final_windowed_win_rate']}")

    env = None
    for episode in tqdm(range(1, num_episodes + 1), disable=verbose, desc=f"Seed {seed} (scored)",
                         leave=True, unit="ep", position=1):
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
            # Agents are NOT reconstructed here -- already warm from
            # pretraining, and never reset at config boundaries during the
            # scored protocol either (see docstring above).

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
            print(f"  Seed {seed}, Episode {episode}/{num_episodes} (scored): "
                  f"Last 100 win rate: {recent_rate:.1f}% ({recent_wins}/100)")

    return results, configs, pretrain_stats


def main():
    print("=" * 80)
    print("TWO OPSRL AGENTS (PRETRAINED-TO-CONVERGENCE) - RED BLUE BUTTON ENVIRONMENT")
    print("=" * 80)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run (if --seed not provided)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run (overrides --seeds if provided)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episodes-per-config", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-steps", action="store_true")
    parser.add_argument("--episode-progress", action="store_true")
    parser.add_argument("--stats-output", type=str, default=None)

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--horizon", type=int, default=None, help="Defaults to --max-steps if not given")
    parser.add_argument("--thompson-samples", type=int, default=10,
                         help="Was 1 by default until 2026-08-23 -- see ai/02-debug.md entry.")
    parser.add_argument("--scale-prior-reward", type=float, default=1.0)
    parser.add_argument("--prior-transition", type=str, choices=["uniform", "optimistic"], default="uniform")
    parser.add_argument("--stage-dependent", action="store_true")
    parser.add_argument("--reward-free", action="store_true")
    parser.add_argument("--no-bernoullized-reward", action="store_true")

    parser.add_argument("--pretrain-window", type=int, default=50,
                         help="Episodes per rolling window when checking pretraining convergence")
    parser.add_argument("--pretrain-patience", type=int, default=3,
                         help="Consecutive plateaued windows required before declaring convergence")
    parser.add_argument("--pretrain-min-delta", type=float, default=0.02,
                         help="Windowed win-rate change below this counts as plateaued")
    parser.add_argument("--pretrain-max-episodes", type=int, default=3000,
                         help="Hard safety cap on pretraining episodes")
    parser.add_argument("--pretrain-min-episodes", type=int, default=200,
                         help="Minimum pretraining episodes before convergence can be declared")
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
    pretrain_kwargs = dict(
        window=args.pretrain_window,
        patience=args.pretrain_patience,
        min_delta=args.pretrain_min_delta,
        max_episodes=args.pretrain_max_episodes,
        min_episodes=args.pretrain_min_episodes,
    )

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"two_opsrl_agents_pretrained_convergence_{SEED_TAG}_ep{NUM_EPISODES_PER_SEED}_step{MAX_STEPS}_redblue_{timestamp}.csv"
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
    print(f"  Scored episodes per seed: {NUM_EPISODES_PER_SEED}")
    print(f"  Episodes per config: {EPISODES_PER_CONFIG}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  OPSRL: gamma={args.gamma} horizon={args.horizon or MAX_STEPS} "
          f"thompson_samples={args.thompson_samples} prior_transition={args.prior_transition}")
    print(f"  Pretrain: window={args.pretrain_window} patience={args.pretrain_patience} "
          f"min_delta={args.pretrain_min_delta} max_episodes={args.pretrain_max_episodes} "
          f"min_episodes={args.pretrain_min_episodes}")
    print(f"  CSV log: {csv_path}")

    all_results = []
    seed_summaries = []
    total_episodes = NUM_SEEDS * NUM_EPISODES_PER_SEED
    with tqdm(total=total_episodes, desc="Total (scored)", unit="ep", leave=True, position=0) as pbar:
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

                results, configs, pretrain_stats = run_seed_experiment(
                    seed=seed,
                    num_episodes=NUM_EPISODES_PER_SEED,
                    episodes_per_config=EPISODES_PER_CONFIG,
                    max_steps=MAX_STEPS,
                    opsrl_kwargs=opsrl_kwargs,
                    pretrain_kwargs=pretrain_kwargs,
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
                    "pretrain_n_episodes": pretrain_stats["n_episodes"],
                    "pretrain_n_env_steps": pretrain_stats["n_env_steps"],
                    "pretrain_converged": pretrain_stats["converged"],
                    "pretrain_final_windowed_win_rate": pretrain_stats["final_windowed_win_rate"],
                })

                print(f"\nSeed {seed} Summary (scored, post-pretraining):")
                print(f"  Success rate: {successes}/{len(results)} ({success_rate:.1f}%)")
                print(f"  Average reward: {avg_reward:+.2f}")
                print(f"  Average steps: {avg_steps:.1f}")
                print(f"  Learning: First half {first_half_wins}/{mid}, Second half {second_half_wins}/{mid}")
                print(f"  Pretraining cost: {pretrain_stats['n_episodes']} episodes, "
                      f"{pretrain_stats['n_env_steps']} env steps, converged={pretrain_stats['converged']}")
        finally:
            csv_file.close()

    def _ser(v):
        return float(v) if isinstance(v, (np.floating, np.integer)) else v

    seed_summaries_serializable = [{k: _ser(v) for k, v in s.items()} for s in seed_summaries]
    stats = {
        "paradigm": "opsrl_pretrained_convergence",
        "n_seeds": NUM_SEEDS,
        "n_episodes_per_seed": NUM_EPISODES_PER_SEED,
        "episodes_per_config": EPISODES_PER_CONFIG,
        "max_steps": MAX_STEPS,
        "opsrl_hyperparams": {**opsrl_kwargs, "horizon": opsrl_kwargs["horizon"] or MAX_STEPS},
        "pretrain_hyperparams": pretrain_kwargs,
        "total_episodes": len(all_results),
        "total_successes": sum(1 for r in all_results if r["success"]),
        "success_rate": float(100 * sum(1 for r in all_results if r["success"]) / max(1, len(all_results))),
        "mean_reward": float(np.mean([r["reward"] for r in all_results])),
        "std_reward": float(np.std([r["reward"] for r in all_results])),
        "mean_steps": float(np.mean([r["steps"] for r in all_results])),
        "std_steps": float(np.std([r["steps"] for r in all_results])),
        "mean_pretrain_episodes": float(np.mean([s["pretrain_n_episodes"] for s in seed_summaries])),
        "mean_pretrain_env_steps": float(np.mean([s["pretrain_n_env_steps"] for s in seed_summaries])),
        "pretrain_convergence_rate": float(100 * sum(1 for s in seed_summaries if s["pretrain_converged"]) / max(1, len(seed_summaries))),
        "seed_summaries": seed_summaries_serializable,
    }
    stats_filename = f"two_opsrl_agents_pretrained_convergence_{SEED_TAG}_ep{NUM_EPISODES_PER_SEED}_step{MAX_STEPS}_redblue_{timestamp}_stats.json"
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
    print("OVERALL RESULTS SUMMARY (scored, post-pretraining)")
    print("=" * 80)
    total_successes = sum(s["successes"] for s in seed_summaries)
    total_episodes_run = sum(s["total"] for s in seed_summaries)
    overall_success_rate = 100 * total_successes / total_episodes_run
    print(f"\nTotal scored episodes: {total_episodes_run}")
    print(f"Total successes: {total_successes} ({overall_success_rate:.1f}%)")
    print(f"Mean pretraining cost: {stats['mean_pretrain_episodes']:.0f} episodes, "
          f"{stats['mean_pretrain_env_steps']:.0f} env steps "
          f"({stats['pretrain_convergence_rate']:.0f}% of seeds converged within the cap)")

    print(f"\n{'Seed':<6} {'Success Rate':<18} {'Avg Reward':<12} {'Pretrain Eps':<14} {'Converged':<10}")
    print("-" * 65)
    for s in seed_summaries:
        print(f"{s['seed']:<6} {s['successes']:>4}/{s['total']:<4} ({s['success_rate']:>5.1f}%)  "
              f"{s['avg_reward']:>+7.2f}     {s['pretrain_n_episodes']:>10}    {str(s['pretrain_converged']):<10}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
