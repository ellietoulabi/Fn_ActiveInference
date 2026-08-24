"""
Run two OPSRL agents on TwoAgentRedBlueButton across a SWEEP of fixed
pretraining budgets, tracing out an adaptation-quality-vs-pretraining-budget
curve rather than reporting one warm-started number.

Companion to run_two_opsrl_agents_pretrained.py (convergence-based
pretraining, one discovered budget) -- see that file's docstring and
ai/02-debug.md, MA Red-Blue-Button, for the shared rationale (isolating
"can it adapt to relocation" from "can it learn mechanics at all", the same
underlying question already resolved for MAPPO via its --mode
pretrained/online split and its checkpoint-interval-evaluation
recommendation in ai/04-writeup.md).

For each seed and each budget level in --budgets: construct fresh
MAOPSRLAgentPretrainedFixedBudget agents, pretrain them for EXACTLY that
many episodes against domain-randomized configs (no convergence check --
the budget is externally chosen, not discovered), then run the standard
scored evaluation protocol and log the result tagged with that budget. This
lets a later plot show scored success rate (or post-relocation recovery
speed) as a function of pretraining budget, alongside the convergence-based
single point and the cold-start (budget=0) baseline.

Isolation, per explicit direction: this file, and the class it drives
(agents/OPSRL/ma_agent_pretrained_fixed_budget.py::MAOPSRLAgentPretrainedFixedBudget),
do not edit:
  - agents/OPSRL/agent.py (Stage 1's shared OPSRLAgent)
  - agents/OPSRL/ma_agent.py (the cold-start MAOPSRLAgent)
  - agents/OPSRL/ma_agent_pretrained_convergence.py (the convergence variant)
  - run_two_opsrl_agents.py / run_two_opsrl_agents_pretrained.py -- only
    IMPORTED from for generic, protocol-agnostic utilities, never edited.
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
from agents.OPSRL.ma_agent_pretrained_fixed_budget import MAOPSRLAgentPretrainedFixedBudget

# Reused (imported, never edited) generic utilities -- see the matching note
# in run_two_opsrl_agents_pretrained.py.
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

def create_opsrl_agent_sweep(seed, width=3, height=3, max_steps=50, gamma=0.95,
                              horizon=None, thompson_samples=10,
                              bernoullized_reward=True, scale_prior_reward=1.0,
                              prior_transition="uniform", stage_dependent=False,
                              reward_free=False):
    return MAOPSRLAgentPretrainedFixedBudget(
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
# Fixed-budget pretraining loop
# =============================================================================

def pretrain_pair_fixed_budget(agent1, agent2, pretrain_rng, max_steps, n_episodes, progress_desc=None):
    """
    Run exactly n_episodes of pretraining against domain-randomized configs
    (fresh random layout every episode, matching the convergence variant's
    generalization goal) -- no convergence check, the budget is externally
    fixed by the sweep. n_episodes=0 is valid and simply skips pretraining
    entirely (the budget=0 point on the sweep curve is, by construction,
    identical to the cold-start baseline).
    """
    history = []
    total_env_steps = 0
    if n_episodes > 0:
        for episode in tqdm(range(1, n_episodes + 1), desc=progress_desc or "Pretrain",
                             leave=False, unit="ep"):
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

    final_win_rate = (sum(history[-50:]) / len(history[-50:])) if history else None
    return {
        "n_episodes": n_episodes,
        "n_env_steps": total_env_steps,
        "final_win_rate_last50": final_win_rate,
    }


# =============================================================================
# Seed x budget loop
# =============================================================================

def run_seed_at_budget(seed, budget, num_episodes, episodes_per_config, max_steps,
                        opsrl_kwargs, verbose=False, csv_writer=None,
                        episode_progress=False, print_steps=False,
                        progress_callback=None):
    """
    Pretrain a fresh agent pair for exactly `budget` episodes, then run the
    standard scored evaluation protocol on them. Agents are constructed
    (and pretrained) ONCE for this (seed, budget) pair and never rebuilt
    across config boundaries during the scored protocol -- same
    no-privileged-reset behavior as the cold-start and convergence-based
    scripts, and (as of 2026-08-20) the MA AIF scripts.
    """
    eval_rng = np.random.default_rng(seed)
    # Distinct pretrain RNG stream per (seed, budget) -- offset by budget too,
    # so different budget levels for the same seed don't all pretrain on the
    # identical config sequence prefix.
    pretrain_rng = np.random.default_rng(int(seed) * 1_000_003 + int(budget) * 131 + 987)

    results = []
    configs = []
    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    for _ in range(num_configs):
        configs.append(generate_random_config(eval_rng))

    agent1 = create_opsrl_agent_sweep(seed=int(seed) * 2 + 1, max_steps=max_steps, **opsrl_kwargs)
    agent2 = create_opsrl_agent_sweep(seed=int(seed) * 2 + 2, max_steps=max_steps, **opsrl_kwargs)

    print(f"  Seed {seed}, budget {budget}: pretraining...")
    pretrain_stats = pretrain_pair_fixed_budget(
        agent1, agent2, pretrain_rng, max_steps, budget,
        progress_desc=f"Pretrain seed{seed} budget{budget}",
    )
    print(f"  Seed {seed}, budget {budget}: pretraining done -- "
          f"{pretrain_stats['n_env_steps']} env steps, "
          f"final win rate (last 50 pretrain episodes)={pretrain_stats['final_win_rate_last50']}")

    env = None
    for episode in tqdm(range(1, num_episodes + 1), disable=verbose,
                         desc=f"Seed {seed} budget{budget} (scored)", leave=True, unit="ep", position=1):
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
        result["pretrain_budget"] = budget
        results.append(result)
        if progress_callback is not None:
            progress_callback(1)

    return results, configs, pretrain_stats


def main():
    print("=" * 80)
    print("TWO OPSRL AGENTS (PRETRAINING-BUDGET SWEEP) - RED BLUE BUTTON ENVIRONMENT")
    print("=" * 80)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run (if --seed not provided)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run (overrides --seeds if provided)")
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 50, 200, 500],
                         help="Pretraining episode budgets to sweep over (0 == cold start)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episodes-per-config", type=int, default=20)
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
    args = parser.parse_args()

    if args.seed is not None:
        SEEDS_TO_RUN = [args.seed]
        SEED_TAG = f"seed{args.seed}"
    else:
        SEEDS_TO_RUN = list(range(args.seeds))
        SEED_TAG = f"seeds{args.seeds}"

    BUDGETS = sorted(set(args.budgets))
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
    csv_filename = f"two_opsrl_agents_pretrained_sweep_{SEED_TAG}_ep{NUM_EPISODES_PER_SEED}_step{MAX_STEPS}_redblue_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    csv_fieldnames = [
        "seed", "pretrain_budget", "episode", "step", "config_idx",
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
    print(f"  Pretraining budgets: {BUDGETS}")
    print(f"  Scored episodes per (seed, budget): {NUM_EPISODES_PER_SEED}")
    print(f"  Episodes per config: {EPISODES_PER_CONFIG}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  CSV log: {csv_path}")

    all_results = []
    budget_summaries = []
    total_runs = NUM_SEEDS * len(BUDGETS) * NUM_EPISODES_PER_SEED
    with tqdm(total=total_runs, desc="Total (scored)", unit="ep", leave=True, position=0) as pbar:
        try:
            for budget in BUDGETS:
                print(f"\n{'#'*80}")
                print(f"BUDGET = {budget} pretraining episodes")
                print(f"{'#'*80}")
                for seed_idx, seed in enumerate(SEEDS_TO_RUN):
                    print(f"\n{'='*80}")
                    print(f"SEED {seed} ({seed_idx + 1}/{NUM_SEEDS}), budget {budget}")
                    print(f"{'='*80}")

                    class SeedBudgetCSVWriter:
                        def __init__(self, writer, seed, budget):
                            self.writer = writer
                            self.seed = seed
                            self.budget = budget

                        def writerow(self, row):
                            row["seed"] = self.seed
                            row["pretrain_budget"] = self.budget
                            self.writer.writerow(row)

                    seed_csv_writer = SeedBudgetCSVWriter(csv_writer, seed, budget)

                    results, configs, pretrain_stats = run_seed_at_budget(
                        seed=seed,
                        budget=budget,
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

                    budget_summaries.append({
                        "seed": seed,
                        "budget": budget,
                        "successes": successes,
                        "total": len(results),
                        "success_rate": success_rate,
                        "avg_reward": avg_reward,
                        "avg_steps": avg_steps,
                        "pretrain_n_env_steps": pretrain_stats["n_env_steps"],
                        "pretrain_final_win_rate_last50": pretrain_stats["final_win_rate_last50"],
                    })

                    print(f"\nSeed {seed}, budget {budget} Summary (scored):")
                    print(f"  Success rate: {successes}/{len(results)} ({success_rate:.1f}%)")
                    print(f"  Average reward: {avg_reward:+.2f}")
        finally:
            csv_file.close()

    def _ser(v):
        return float(v) if isinstance(v, (np.floating, np.integer)) else v

    budget_summaries_serializable = [{k: _ser(v) for k, v in s.items()} for s in budget_summaries]

    # Aggregate curve: mean success rate per budget level, across seeds.
    curve = []
    for budget in BUDGETS:
        rows = [s for s in budget_summaries if s["budget"] == budget]
        curve.append({
            "budget": budget,
            "n_seeds": len(rows),
            "mean_success_rate": float(np.mean([r["success_rate"] for r in rows])),
            "std_success_rate": float(np.std([r["success_rate"] for r in rows])),
            "mean_pretrain_env_steps": float(np.mean([r["pretrain_n_env_steps"] for r in rows])),
        })

    stats = {
        "paradigm": "opsrl_pretrained_sweep",
        "n_seeds": NUM_SEEDS,
        "budgets": BUDGETS,
        "n_episodes_per_run": NUM_EPISODES_PER_SEED,
        "episodes_per_config": EPISODES_PER_CONFIG,
        "max_steps": MAX_STEPS,
        "opsrl_hyperparams": {**opsrl_kwargs, "horizon": opsrl_kwargs["horizon"] or MAX_STEPS},
        "total_scored_episodes": len(all_results),
        "curve": curve,
        "budget_summaries": budget_summaries_serializable,
    }
    stats_filename = f"two_opsrl_agents_pretrained_sweep_{SEED_TAG}_ep{NUM_EPISODES_PER_SEED}_step{MAX_STEPS}_redblue_{timestamp}_stats.json"
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
    print("BUDGET SWEEP CURVE (mean success rate across seeds, per pretraining budget)")
    print("=" * 80)
    print(f"\n{'Budget':<10} {'Seeds':<8} {'Mean Success Rate':<20} {'Mean Pretrain EnvSteps':<24}")
    print("-" * 65)
    for c in curve:
        print(f"{c['budget']:<10} {c['n_seeds']:<8} {c['mean_success_rate']:>6.1f}% ± {c['std_success_rate']:<10.1f} "
              f"{c['mean_pretrain_env_steps']:>10.0f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
