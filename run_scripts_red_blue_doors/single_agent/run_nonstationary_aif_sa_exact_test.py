"""
Verification harness for the isolated agents/ActiveInferenceSAExact + generative_models/
SA_ActiveInference/RedBlueButtonExact packages.

Purpose: this is NOT a production/cluster script and does not touch, import, or
in any way depend on the original agents/ActiveInference or generative_models/
SA_ActiveInference/RedBlueButton packages that back the already-reported Stage 1
nine-agent results (ai/02-debug.md, ai/defense_story_for_james.md). It exists
purely to answer one question: does Stage 1's 100% success / ~7.86-mean-step
result survive once the SA agent gets (1) the same mathematically-exact
per-group info-gain decomposition already verified for MA Red-Blue-Button's
Exact package, and (2) the same zeroed movement noise the 2026-08-13 MA
Independent redesign applied -- or does it reproduce the "flat-EFE tie" total
policy-space deadlock documented for MA in ai/02-debug.md?

Protocol mirrors run_scripts_red_blue_doors/compare_agents/compare_nine_agents.py's
AIF row exactly (200 episodes, config relocated every 25 episodes via a fresh
random draw avoiding the agent's start cell, max 50 steps/episode, policy_len=2,
gamma=2.0, alpha=1.0, num_iter=16) -- same protocol parameters, not a byte-identical
RNG replay of the original 30-seed dataset (that dataset's own seed draws are not
required for this comparison; only a like-for-like protocol is).
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButtonExact import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils,
)
from agents.ActiveInferenceSAExact.agent import Agent

NUM_EPISODES = 200
EPISODES_PER_CONFIG = 25
MAX_STEPS = 50
BASE_SEED = 42


def set_protocol(num_episodes, episodes_per_config):
    global NUM_EPISODES, EPISODES_PER_CONFIG
    NUM_EPISODES = num_episodes
    EPISODES_PER_CONFIG = episodes_per_config


def make_agent():
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    agent = Agent(
        A_fn=A_fn, B_fn=B_fn, C_fn=C_fn, D_fn=D_fn,
        state_factors=state_factors, state_sizes=state_sizes,
        observation_labels=model_init.observations,
        env_params={'width': model_init.n, 'height': model_init.m},
        actions=list(range(6)),
        policy_len=2, gamma=2.0, alpha=1.0,
        num_iter=16,
    )
    agent.reset()
    return agent


ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}


def run_episode(env, agent, max_steps=50, csv_writer=None, seed=None, episode_num=None, agent_label='AIF_Exact'):
    env_obs, _ = env.reset()

    agent.qs['agent_pos'] = np.zeros(9)
    agent.qs['agent_pos'][0] = 1.0
    agent.qs['red_button_state'] = np.array([1.0, 0.0])
    agent.qs['blue_button_state'] = np.array([1.0, 0.0])
    agent.action = 5
    agent.prev_actions = []
    agent.curr_timestep = 0

    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    episode_reward = 0.0
    outcome = 'timeout'
    positions = []

    step = 0
    for step in range(1, max_steps + 1):
        action = agent.step(obs_dict)
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        positions.append(env.agent_position)

        if csv_writer is not None:
            csv_writer.writerow({
                'seed': seed,
                'agent': agent_label,
                'episode': episode_num,
                'step': step,
                'action': action,
                'action_name': ACTION_NAMES[action],
                'reward': reward,
            })

        obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        if done:
            outcome = info.get('result', 'neutral')
            break

    # Deadlock signature (same detector used for the MA Red-Blue-Button audit):
    # last 20 steps of a timeout loss touch <=4 distinct positions.
    deadlock = False
    if outcome != 'win' and step >= max_steps:
        tail = positions[-20:]
        if len(set(tail)) <= 4:
            deadlock = True

    return {'outcome': outcome, 'reward': episode_reward, 'steps': step,
            'success': outcome == 'win', 'deadlock': deadlock}


def run_seed(seed_idx, csv_writer=None, agent_label='AIF_Exact'):
    current_seed = BASE_SEED + seed_idx
    rng = np.random.RandomState(current_seed)

    agent = make_agent()

    num_configs = NUM_EPISODES // EPISODES_PER_CONFIG
    configs = []
    for _ in range(num_configs):
        available = list(range(1, 9))
        rng.shuffle(available)
        red_idx, blue_idx = available[0], available[1]
        configs.append({
            'red_pos': (red_idx // 3, red_idx % 3),
            'blue_pos': (blue_idx // 3, blue_idx % 3),
        })

    results = []
    for episode in range(1, NUM_EPISODES + 1):
        config_idx = (episode - 1) // EPISODES_PER_CONFIG
        config = configs[config_idx]
        env = SingleAgentRedBlueButtonEnv(
            width=3, height=3,
            red_button_pos=config['red_pos'],
            blue_button_pos=config['blue_pos'],
            agent_start_pos=(0, 0),
            max_steps=MAX_STEPS,
        )
        result = run_episode(env, agent, max_steps=MAX_STEPS, csv_writer=csv_writer,
                              seed=current_seed, episode_num=episode, agent_label=agent_label)
        result['seed'] = current_seed
        result['episode'] = episode
        result['config_idx'] = config_idx
        results.append(result)

        if episode % 10 == 0:
            recent = results[-10:]
            sr = sum(r['success'] for r in recent) / len(recent)
            ms = np.mean([r['steps'] for r in recent])
            print(f"  [seed {current_seed}] ep {episode}/{NUM_EPISODES}  "
                  f"last10 success={sr:.2f} mean_steps={ms:.1f}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=10,
                         help='Number of seeds to run (BASE_SEED..BASE_SEED+seeds-1). Default 10.')
    parser.add_argument('--episodes', type=int, default=200,
                         help='Episodes per seed (default 200, matching the original protocol).')
    parser.add_argument('--episodes-per-config', type=int, default=25,
                         help='Episodes per button-position config (default 25).')
    parser.add_argument('--log-dir', type=Path, default=None,
                         help='If set, write one nine_agents_comparison_*.csv per seed here '
                              '(same schema plot_sa_redbluebuttons_nine.py expects), agent label "AIF_Exact".')
    args = parser.parse_args()
    set_protocol(args.episodes, args.episodes_per_config)

    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    for seed_idx in range(args.seeds):
        print(f"Running seed_idx={seed_idx} (seed={BASE_SEED + seed_idx})...")
        if args.log_dir is not None:
            csv_path = args.log_dir / (
                f"nine_agents_comparison_ep{args.episodes}_step{MAX_STEPS}_"
                f"seed{seed_idx}_{timestamp}.csv"
            )
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['seed', 'agent', 'episode', 'step',
                                                        'action', 'action_name', 'reward'])
                writer.writeheader()
                all_results.extend(run_seed(seed_idx, csv_writer=writer))
        else:
            all_results.extend(run_seed(seed_idx))

    n = len(all_results)
    successes = sum(r['success'] for r in all_results)
    timeouts = sum((not r['success']) and r['steps'] >= MAX_STEPS for r in all_results)
    deadlocks = sum(r['deadlock'] for r in all_results)
    mean_steps_win = np.mean([r['steps'] for r in all_results if r['success']])
    mean_steps_all = np.mean([r['steps'] for r in all_results])

    print()
    print("=" * 70)
    print(f"SA-Exact AIF: {args.seeds} seeds x {NUM_EPISODES} episodes = {n} episodes")
    print(f"  success_rate       = {successes/n:.4f}")
    print(f"  timeout_rate       = {timeouts/n:.4f}")
    print(f"  deadlock_rate      = {deadlocks/n:.4f}  (of ALL episodes, not just timeouts)")
    print(f"  mean_steps | win   = {mean_steps_win:.2f}")
    print(f"  mean_steps | all   = {mean_steps_all:.2f}")
    print("=" * 70)
    print("Original (unmodified) SA AIF baseline, for reference: "
          "success_rate=1.000, mean_steps=7.86 (ai/02-debug.md, 30-seed step50 audit)")


if __name__ == '__main__':
    main()
