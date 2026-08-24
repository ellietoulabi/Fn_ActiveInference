"""
Run OPSRL on SingleAgentRedBlueButton (Stage 1) across a SWEEP of fixed
domain-randomized pretraining budgets (0, 50, 200, 500 episodes by default --
0 deliberately included as the cold-start-equivalent anchor point), each
followed by the same scored evaluation protocol as the cold-start and
convergence-based scripts. Produces a curve of scored success rate vs.
pretraining budget, rather than committing to one warm-started number --
mirrors run_two_opsrl_agents_pretrained_sweep.py's MA rationale
(ai/02-debug.md) and ai/04-writeup.md's "report both honestly" MAPPO
fairness precedent.

Isolation: does not edit agents/OPSRL/agent.py,
agents/OPSRL/agent_pretrained_convergence.py, or compare_nine_agents.py --
only imports the now-fixed OPSRLAgent (ai/02-debug.md, 2026-08-22 entry) via
OPSRLAgentPretrainedFixedBudget, plus this directory's own
opsrl_pretrained_common.py helpers.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from agents.OPSRL.agent_pretrained_fixed_budget import OPSRLAgentPretrainedFixedBudget
from run_scripts_red_blue_doors.single_agent.opsrl_pretrained_common import (
    generate_random_config, run_opsrl_episode, open_csv_writer,
)

BASE_SEED = 42


def create_agent(seed, max_steps, gamma=0.95, thompson_samples=10,
                  bernoullized_reward=True, scale_prior_reward=1.0,
                  prior_transition='uniform', stage_dependent=False):
    dummy_env = SingleAgentRedBlueButtonEnv(width=3, height=3, red_button_pos=(0, 2),
                                             blue_button_pos=(2, 0), agent_start_pos=(0, 0),
                                             max_steps=max_steps)
    return OPSRLAgentPretrainedFixedBudget(
        env=dummy_env, gamma=gamma, horizon=max_steps,
        bernoullized_reward=bernoullized_reward, scale_prior_reward=scale_prior_reward,
        thompson_samples=thompson_samples, prior_transition=prior_transition,
        reward_free=False, stage_dependent=stage_dependent, seed=seed,
    )


def pretrain_fixed_budget(agent, pretrain_rng, max_steps, budget, progress_desc=None):
    """Pretrain for exactly `budget` domain-randomized episodes (no
    convergence check -- the budget is externally chosen). budget=0 skips
    pretraining entirely (the cold-start-equivalent anchor)."""
    total_env_steps = 0
    if budget <= 0:
        return {'n_episodes': 0, 'n_env_steps': 0}

    pbar = tqdm(total=budget, desc=progress_desc or "Pretrain", leave=False, unit="ep")
    try:
        for _ in range(budget):
            config = generate_random_config(pretrain_rng)
            env = SingleAgentRedBlueButtonEnv(
                width=3, height=3,
                red_button_pos=config['red_pos'], blue_button_pos=config['blue_pos'],
                agent_start_pos=(0, 0), max_steps=max_steps,
            )
            result = run_opsrl_episode(env, agent, max_steps=max_steps, csv_writer=None)
            total_env_steps += result['steps']
            pbar.update(1)
    finally:
        pbar.close()

    return {'n_episodes': budget, 'n_env_steps': total_env_steps}


def run_seed_at_budget(seed_idx, budget, num_episodes, episodes_per_config, max_steps,
                        opsrl_kwargs, csv_writer=None):
    current_seed = BASE_SEED + seed_idx
    eval_rng = np.random.default_rng(current_seed)
    pretrain_rng = np.random.default_rng(int(current_seed) * 1_000_003 + 987)

    agent = create_agent(seed=current_seed, max_steps=max_steps, **opsrl_kwargs)

    pretrain_stats = pretrain_fixed_budget(
        agent, pretrain_rng, max_steps, budget,
        progress_desc=f"Pretrain seed{current_seed} budget{budget}",
    )

    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    configs = [generate_random_config(eval_rng) for _ in range(num_configs)]

    results = []
    env = None
    for episode in range(1, num_episodes + 1):
        config_idx = (episode - 1) // episodes_per_config
        config = configs[config_idx]
        if (episode - 1) % episodes_per_config == 0 or env is None:
            env = SingleAgentRedBlueButtonEnv(
                width=3, height=3,
                red_button_pos=config['red_pos'], blue_button_pos=config['blue_pos'],
                agent_start_pos=(0, 0), max_steps=max_steps,
            )
        result = run_opsrl_episode(env, agent, max_steps=max_steps, csv_writer=csv_writer,
                                    seed=current_seed, episode_num=episode, config_idx=config_idx)
        results.append(result)

    return results, pretrain_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--episodes-per-config', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--budgets', type=int, nargs='+', default=[0, 50, 200, 500])
    parser.add_argument('--log-dir', type=Path, default=None)
    args = parser.parse_args()

    opsrl_kwargs = dict(gamma=0.95, thompson_samples=10, bernoullized_reward=True,
                         scale_prior_reward=1.0, prior_transition='uniform', stage_dependent=False)

    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    curve = []
    for budget in args.budgets:
        print(f"=== budget={budget} ===")
        budget_results = []
        budget_pretrain_steps = []
        for seed_idx in range(args.seeds):
            print(f"  seed_idx={seed_idx} (seed={BASE_SEED + seed_idx})...")
            csv_file = writer = None
            if args.log_dir is not None:
                csv_path = args.log_dir / (
                    f"nine_agents_comparison_budget{budget}_ep{args.episodes}_"
                    f"step{args.max_steps}_seed{seed_idx}_{timestamp}.csv"
                )
                csv_file, writer = open_csv_writer(csv_path)
            results, pretrain_stats = run_seed_at_budget(
                seed_idx, budget, args.episodes, args.episodes_per_config, args.max_steps,
                opsrl_kwargs, csv_writer=writer,
            )
            if csv_file is not None:
                csv_file.close()
            budget_results.extend(results)
            budget_pretrain_steps.append(pretrain_stats['n_env_steps'])

        n = len(budget_results)
        successes = sum(r['success'] for r in budget_results)
        deadlocks = sum(r['deadlock'] for r in budget_results)
        row = {
            'budget': budget,
            'success_rate_mean': successes / n,
            'deadlock_rate': deadlocks / n,
            'mean_pretrain_env_steps': float(np.mean(budget_pretrain_steps)),
        }
        curve.append(row)
        print(f"  budget={budget}: success_rate={row['success_rate_mean']:.4f} "
              f"deadlock_rate={row['deadlock_rate']:.4f} "
              f"mean_pretrain_env_steps={row['mean_pretrain_env_steps']:.1f}")

    print()
    print("=" * 70)
    print("OPSRL pretraining-budget sweep, scored success rate:")
    for row in curve:
        print(f"  budget={row['budget']:5d}  success_rate={row['success_rate_mean']:.4f}  "
              f"deadlock_rate={row['deadlock_rate']:.4f}")
    print("=" * 70)

    if args.log_dir is not None:
        stats_path = args.log_dir / f"stats_pretrained_sweep_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump({'curve': curve}, f, indent=2)
        print(f"Stats written to {stats_path}")


if __name__ == '__main__':
    main()
