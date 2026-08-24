"""
Run OPSRL on SingleAgentRedBlueButton (Stage 1), WARM-STARTED via a
domain-randomized pretraining phase run to convergence, before the real,
scored evaluation protocol begins.

Why this exists (mirrors run_two_opsrl_agents_pretrained.py's MA rationale,
ai/02-debug.md): the cold-start protocol (compare_nine_agents.py's OPSRL row)
conflates "can OPSRL learn navigation/press mechanics at all" with "can
OPSRL adapt when the button relocates" -- only the second is what H1 claims
to test, since AIF is deployed with a hand-specified, already-correct model
and never has to solve the first problem from scratch. This script isolates
the second question: the agent is pretrained against a stream of
domain-randomized configs (a fresh random layout every pretraining episode,
not blocked into fixed configs, so the posterior generalizes rather than
memorizes one layout) until its windowed win rate plateaus. Only THEN does
the real, scored protocol begin (episodes/episodes-per-config/relocations),
so what's measured is adaptation speed from an already-warm starting point,
with button-location belief for each evaluation config still genuinely
undiscovered -- exactly the situation AIF is also in at the start of every
config.

Isolation: does not edit agents/OPSRL/agent.py, agents/OPSRL/ma_agent.py, or
compare_nine_agents.py -- only imports the now-fixed OPSRLAgent (see
ai/02-debug.md, 2026-08-22 entry) via OPSRLAgentPretrainedConvergence, plus
this directory's own opsrl_pretrained_common.py helpers.
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
from agents.OPSRL.agent_pretrained_convergence import OPSRLAgentPretrainedConvergence
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
    return OPSRLAgentPretrainedConvergence(
        env=dummy_env, gamma=gamma, horizon=max_steps,
        bernoullized_reward=bernoullized_reward, scale_prior_reward=scale_prior_reward,
        thompson_samples=thompson_samples, prior_transition=prior_transition,
        reward_free=False, stage_dependent=stage_dependent, seed=seed,
    )


def pretrain_to_convergence(agent, pretrain_rng, max_steps, window=50, patience=3,
                             min_delta=0.02, max_episodes=3000, min_episodes=200,
                             progress_desc=None):
    """Domain-randomized pretraining until windowed win rate plateaus. See
    module docstring and run_two_opsrl_agents_pretrained.py's MA analogue for
    the full rationale (same convergence-detection design, ported directly)."""
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
            env = SingleAgentRedBlueButtonEnv(
                width=3, height=3,
                red_button_pos=config['red_pos'], blue_button_pos=config['blue_pos'],
                agent_start_pos=(0, 0), max_steps=max_steps,
            )
            result = run_opsrl_episode(env, agent, max_steps=max_steps, csv_writer=None)
            history.append(1 if result['success'] else 0)
            total_env_steps += result['steps']
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
        'n_episodes': episode, 'n_env_steps': total_env_steps, 'converged': converged,
        'final_windowed_win_rate': win_rate_windows[-1] if win_rate_windows else None,
        'win_rate_windows': win_rate_windows,
    }


def run_seed(seed_idx, num_episodes, episodes_per_config, max_steps,
             opsrl_kwargs, pretrain_kwargs, csv_writer=None):
    current_seed = BASE_SEED + seed_idx
    eval_rng = np.random.default_rng(current_seed)
    # Separate RNG stream from scored-eval configs (offset seed), matching
    # the MA convergence script's own separation -- pretraining can never
    # accidentally draw the exact config sequence evaluation will score.
    pretrain_rng = np.random.default_rng(int(current_seed) * 1_000_003 + 987)

    agent = create_agent(seed=current_seed, max_steps=max_steps, **opsrl_kwargs)

    print(f"  Seed {current_seed}: pretraining (domain-randomized configs) until convergence...")
    pretrain_stats = pretrain_to_convergence(
        agent, pretrain_rng, max_steps,
        window=pretrain_kwargs['window'], patience=pretrain_kwargs['patience'],
        min_delta=pretrain_kwargs['min_delta'], max_episodes=pretrain_kwargs['max_episodes'],
        min_episodes=pretrain_kwargs['min_episodes'],
        progress_desc=f"Pretrain seed{current_seed}",
    )
    print(f"  Seed {current_seed}: pretraining done -- {pretrain_stats['n_episodes']} episodes, "
          f"{pretrain_stats['n_env_steps']} env steps, converged={pretrain_stats['converged']}, "
          f"final windowed win rate={pretrain_stats['final_windowed_win_rate']}")

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
            # Agent is NOT reconstructed here -- already warm from
            # pretraining, and never reset at config boundaries during the
            # scored protocol either (no paradigm in this project gets a
            # privileged "the environment just changed" reset).
        result = run_opsrl_episode(env, agent, max_steps=max_steps, csv_writer=csv_writer,
                                    seed=current_seed, episode_num=episode, config_idx=config_idx)
        results.append(result)

        if episode % 10 == 0:
            recent = results[-10:]
            sr = sum(r['success'] for r in recent) / len(recent)
            ms = np.mean([r['steps'] for r in recent])
            print(f"  [seed {current_seed}] ep {episode}/{num_episodes}  "
                  f"last10 success={sr:.2f} mean_steps={ms:.1f}", flush=True)

    return results, pretrain_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--episodes-per-config', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--log-dir', type=Path, default=None)
    parser.add_argument('--pretrain-window', type=int, default=50)
    parser.add_argument('--pretrain-patience', type=int, default=3)
    parser.add_argument('--pretrain-min-delta', type=float, default=0.02)
    parser.add_argument('--pretrain-max-episodes', type=int, default=3000)
    parser.add_argument('--pretrain-min-episodes', type=int, default=200)
    args = parser.parse_args()

    pretrain_kwargs = dict(window=args.pretrain_window, patience=args.pretrain_patience,
                            min_delta=args.pretrain_min_delta, max_episodes=args.pretrain_max_episodes,
                            min_episodes=args.pretrain_min_episodes)
    opsrl_kwargs = dict(gamma=0.95, thompson_samples=10, bernoullized_reward=True,
                         scale_prior_reward=1.0, prior_transition='uniform', stage_dependent=False)

    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    all_pretrain_stats = []
    for seed_idx in range(args.seeds):
        print(f"Running seed_idx={seed_idx} (seed={BASE_SEED + seed_idx})...")
        csv_file = writer = None
        if args.log_dir is not None:
            csv_path = args.log_dir / (
                f"nine_agents_comparison_ep{args.episodes}_step{args.max_steps}_"
                f"seed{seed_idx}_{timestamp}.csv"
            )
            csv_file, writer = open_csv_writer(csv_path)
        results, pretrain_stats = run_seed(
            seed_idx, args.episodes, args.episodes_per_config, args.max_steps,
            opsrl_kwargs, pretrain_kwargs, csv_writer=writer,
        )
        if csv_file is not None:
            csv_file.close()
        all_results.extend(results)
        all_pretrain_stats.append(pretrain_stats)

    n = len(all_results)
    successes = sum(r['success'] for r in all_results)
    deadlocks = sum(r['deadlock'] for r in all_results)
    mean_steps_win = np.mean([r['steps'] for r in all_results if r['success']]) if successes else float('nan')

    print()
    print("=" * 70)
    print(f"OPSRL (pretrained-to-convergence): {args.seeds} seeds x {args.episodes} episodes = {n} scored episodes")
    print(f"  success_rate       = {successes/n:.4f}")
    print(f"  deadlock_rate      = {deadlocks/n:.4f}")
    print(f"  mean_steps | win   = {mean_steps_win:.2f}")
    print(f"  mean_pretrain_episodes = {np.mean([s['n_episodes'] for s in all_pretrain_stats]):.1f}")
    print(f"  mean_pretrain_env_steps = {np.mean([s['n_env_steps'] for s in all_pretrain_stats]):.1f}")
    print(f"  pretrain_convergence_rate = {np.mean([s['converged'] for s in all_pretrain_stats]):.2f}")
    print("=" * 70)

    if args.log_dir is not None:
        stats_path = args.log_dir / f"stats_pretrained_convergence_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'success_rate': successes / n, 'deadlock_rate': deadlocks / n,
                'mean_steps_win': float(mean_steps_win),
                'mean_pretrain_episodes': float(np.mean([s['n_episodes'] for s in all_pretrain_stats])),
                'mean_pretrain_env_steps': float(np.mean([s['n_env_steps'] for s in all_pretrain_stats])),
                'pretrain_convergence_rate': float(np.mean([s['converged'] for s in all_pretrain_stats])),
                'per_seed_pretrain_stats': all_pretrain_stats,
            }, f, indent=2)
        print(f"Stats written to {stats_path}")


if __name__ == '__main__':
    main()
