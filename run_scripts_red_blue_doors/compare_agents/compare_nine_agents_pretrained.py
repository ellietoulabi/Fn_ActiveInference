"""
"Fair adaptation-speed" variant of compare_nine_agents.py (Stage 1, SA
Red-Blue-Button). Same 9 agents, same hyperparameters, same per-seed config
sequence convention -- the only difference is that the 8 non-AIF agents are
each PRETRAINED (against a domain-randomized stream of configs, until their
own windowed win rate plateaus) before the real, scored relocation protocol
begins. AIF is never pretrained -- it is deployed with a hand-specified,
already-correct generative model, which is exactly the asymmetry this script
is designed to control for on the RL/OPSRL side. See ai/02-debug.md,
2026-08-23 "fair adaptation framework" entry, and this file's own docstring
in pretrain_common.py for the important disclosed caveat about what tabular
pretraining can and cannot transfer.

Isolation: imports agent classes and generate_random_config-equivalent logic
but does not edit compare_nine_agents.py, any agent class, or any generative
model file. A completely new, separate script.
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import copy
import csv
import json
import random
import numpy as np
from datetime import datetime

from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils,
)
from agents.ActiveInference.agent import Agent
from agents.QLearning.qlearning_agent import QLearningAgent
from agents.QLearning.dynaq_agent import DynaQAgent as VanillaDynaQ
from agents.QLearning.dynaq_agent_with_recency_bias import DynaQAgent as RecencyDynaQ
from agents.QLearning.dynaq_agent_trajectory_sampling import DynaQAgent as TrajectorySamplingDynaQ
from agents.OPSRL import OPSRLAgent

from run_scripts_red_blue_doors.compare_agents.pretrain_common import (
    generate_random_config_np, run_ql_family_episode, pretrain_ql_agent_to_convergence,
)
from run_scripts_red_blue_doors.single_agent.opsrl_pretrained_common import (
    run_opsrl_episode as run_opsrl_episode_core,
)

ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
BASE_SEED = 42
PLANNING_STEPS = 2
RECENCY_DECAYS = [0.99, 0.95, 0.90, 0.85]
AGENT_NAMES = ['AIF', 'QLearning', 'Vanilla', 'Recency0.99', 'Recency0.95',
               'Recency0.9', 'Recency0.85', 'TrajSampling', 'OPSRL']


def create_all_agents(current_seed, max_steps):
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    aif_agent = Agent(
        A_fn=A_fn, B_fn=B_fn, C_fn=C_fn, D_fn=D_fn,
        state_factors=state_factors, state_sizes=state_sizes,
        observation_labels=model_init.observations,
        env_params={'width': model_init.n, 'height': model_init.m},
        actions=list(range(6)), policy_len=2, gamma=2.0, alpha=1.0, num_iter=16,
    )
    aif_agent.reset()

    ql_agent = QLearningAgent(action_space_size=6, learning_rate=0.1, discount_factor=0.95,
                               epsilon=1.0, epsilon_decay=0.95, min_epsilon=0.05, load_existing=False)
    vanilla_agent = VanillaDynaQ(action_space_size=6, planning_steps=PLANNING_STEPS,
                                  learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
                                  epsilon_decay=0.95, min_epsilon=0.05, load_existing=False)
    recency_agents = [
        RecencyDynaQ(action_space_size=6, planning_steps=PLANNING_STEPS, recency_decay=decay,
                      learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
                      epsilon_decay=0.95, min_epsilon=0.05, load_existing=False)
        for decay in RECENCY_DECAYS
    ]
    traj_agent = TrajectorySamplingDynaQ(action_space_size=6, planning_steps=PLANNING_STEPS,
                                          use_trajectory_sampling=True, n_trajectories=10,
                                          rollout_length=5, planning_epsilon=0.1,
                                          learning_rate=0.1, discount_factor=0.95, epsilon=1.0,
                                          epsilon_decay=0.95, min_epsilon=0.05, load_existing=False)

    temp_env = SingleAgentRedBlueButtonEnv(width=3, height=3, red_button_pos=(0, 2),
                                            blue_button_pos=(2, 0), agent_start_pos=(0, 0),
                                            max_steps=max_steps)
    opsrl_agent = OPSRLAgent(env=temp_env, gamma=0.95, horizon=max_steps,
                              bernoullized_reward=True, scale_prior_reward=1.0,
                              thompson_samples=10, prior_transition='uniform',
                              reward_free=False, stage_dependent=False, seed=current_seed)

    return [aif_agent, ql_agent, vanilla_agent] + recency_agents + [traj_agent, opsrl_agent]


def pretrain_all(agents, pretrain_rng, max_steps, window, patience, min_delta,
                  max_episodes, min_episodes):
    """Pretrain every non-AIF agent (index 0 is always AIF, skipped) against
    its own independent domain-randomized config stream, drawn from the same
    pretrain_rng in agent order so results stay reproducible given a seed."""
    stats = {}
    for name, agent in zip(AGENT_NAMES, agents):
        if name == 'AIF':
            continue
        print(f"  Pretraining {name}...")
        if name == 'OPSRL':
            history = []
            episode = 0
            win_rate_windows = []
            plateaued_count = 0
            converged = False
            while episode < max_episodes:
                episode += 1
                config = generate_random_config_np(pretrain_rng)
                env = SingleAgentRedBlueButtonEnv(
                    width=3, height=3, red_button_pos=config['red_pos'],
                    blue_button_pos=config['blue_pos'], agent_start_pos=(0, 0), max_steps=max_steps,
                )
                result = run_opsrl_episode_core(env, agent, max_steps=max_steps)
                history.append(1 if result['success'] else 0)
                if episode % window == 0 and episode >= min_episodes:
                    recent_wr = sum(history[-window:]) / window
                    win_rate_windows.append(recent_wr)
                    if len(win_rate_windows) >= 2:
                        delta = abs(win_rate_windows[-1] - win_rate_windows[-2])
                        plateaued_count = plateaued_count + 1 if delta < min_delta else 0
                        if plateaued_count >= patience:
                            converged = True
                            break
            stats[name] = {'n_episodes': episode, 'converged': converged,
                            'final_windowed_win_rate': win_rate_windows[-1] if win_rate_windows else None}
        else:
            stats[name] = pretrain_ql_agent_to_convergence(
                agent, pretrain_rng, max_steps=max_steps, window=window, patience=patience,
                min_delta=min_delta, max_episodes=max_episodes, min_episodes=min_episodes,
                progress_desc=name,
            )
        print(f"    {name}: {stats[name]['n_episodes']} episodes, "
              f"converged={stats[name]['converged']}, "
              f"final_win_rate={stats[name]['final_windowed_win_rate']}")
    return stats


def run_scored_protocol(agents, current_seed, num_episodes, episodes_per_config, max_steps, csv_writer, configs):
    all_results = {name: [] for name in AGENT_NAMES}
    for episode in range(1, num_episodes + 1):
        config_idx = (episode - 1) // episodes_per_config
        config = configs[config_idx]
        env = SingleAgentRedBlueButtonEnv(
            width=3, height=3, red_button_pos=config['red_pos'], blue_button_pos=config['blue_pos'],
            agent_start_pos=(0, 0), max_steps=max_steps,
        )
        for name, agent in zip(AGENT_NAMES, agents):
            env.reset()
            if name == 'AIF':
                result = run_aif_episode(env, agent, episode, max_steps, csv_writer, current_seed, name)
            elif name == 'OPSRL':
                r = run_opsrl_episode_core(env, agent, max_steps=max_steps, csv_writer=csv_writer,
                                            seed=current_seed, episode_num=episode, agent_label=name)
                result = r
            else:
                result = run_ql_episode_logged(env, agent, episode, max_steps, csv_writer, current_seed, name)
            all_results[name].append(result)
        if episode % 10 == 0:
            recents = {n: np.mean([r['success'] for r in all_results[n][-10:]]) for n in AGENT_NAMES}
            print(f"  ep {episode}/{num_episodes}  " +
                  "  ".join(f"{n}={recents[n]:.2f}" for n in AGENT_NAMES), flush=True)
    return all_results


def run_aif_episode(env, agent, episode_num, max_steps, csv_writer, seed, agent_label):
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
    step = 0
    for step in range(1, max_steps + 1):
        action = agent.step(obs_dict)
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        if csv_writer is not None:
            csv_writer.writerow({'seed': seed, 'agent': agent_label, 'episode': episode_num,
                                  'step': step, 'action': action, 'action_name': ACTION_NAMES[action],
                                  'reward': reward,
                                  'agent_pos': env_utils.xy_to_index(*env.agent_position, width=3)})
        obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        if done:
            outcome = info.get('result', 'neutral')
            break
    return {'outcome': outcome, 'reward': episode_reward, 'steps': step, 'success': outcome == 'win'}


def run_ql_episode_logged(env, agent, episode_num, max_steps, csv_writer, seed, agent_label):
    env_obs, _ = env.reset()
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    state = agent.get_state(obs_dict)
    episode_reward = 0.0
    outcome = 'timeout'
    step = 0
    for step in range(1, max_steps + 1):
        action = agent.choose_action(state)
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        if csv_writer is not None:
            csv_writer.writerow({'seed': seed, 'agent': agent_label, 'episode': episode_num,
                                  'step': step, 'action': action, 'action_name': ACTION_NAMES[action],
                                  'reward': reward,
                                  'agent_pos': env_utils.xy_to_index(*env.agent_position, width=3)})
        next_obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        next_state = agent.get_state(next_obs_dict) if not done else None
        agent.update_q_table(state, action, reward, next_state)
        if hasattr(agent, 'update_model'):
            agent.update_model(state, action, next_state, reward, terminated)
            agent.planning()
        state = next_state
        if done:
            outcome = info.get('result', 'neutral')
            break
    agent.decay_exploration()
    return {'outcome': outcome, 'reward': episode_reward, 'steps': step, 'success': outcome == 'win'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--episodes-per-config', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--pretrain-window', type=int, default=50)
    parser.add_argument('--pretrain-patience', type=int, default=3)
    parser.add_argument('--pretrain-min-delta', type=float, default=0.02)
    parser.add_argument('--pretrain-max-episodes', type=int, default=1500)
    parser.add_argument('--pretrain-min-episodes', type=int, default=150)
    parser.add_argument('--log-dir', type=Path, default=None)
    parser.add_argument('--seed-idx-offset', type=int, default=0,
                         help='Added to each seed_idx in range(--seeds); lets multiple '
                              'processes each run --seeds 1 at a different offset in parallel '
                              'without colliding on the same seed.')
    parser.add_argument('--configs-file', type=Path, default=None,
                        help='Path to a JSON file of this seed\'s button-position configs. If it '
                             'already exists, configs are LOADED from it (byte-identical map '
                             'sequence) instead of regenerated -- use this to run an additional '
                             'agent later against the exact same map/seed sequence as an earlier '
                             'run. If it does not exist (or this flag is omitted), configs are '
                             'generated as usual and always saved -- to this path if given, '
                             'otherwise to an auto-named file next to the CSV log. Only meaningful '
                             'with --seeds 1 (one seed per invocation, e.g. one SLURM array task); '
                             'combining it with --seeds > 1 is rejected since one file cannot hold '
                             'more than one seed\'s config sequence.')
    args = parser.parse_args()

    if args.configs_file is not None:
        assert args.seeds == 1, (
            "--configs-file only makes sense with --seeds 1 (one seed per invocation); "
            f"got --seeds {args.seeds}. Run once per seed (e.g. via --seed-idx-offset) instead."
        )

    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for seed_idx in range(args.seed_idx_offset, args.seed_idx_offset + args.seeds):
        current_seed = BASE_SEED + seed_idx
        print(f"\n{'='*80}\nSEED INDEX {seed_idx} (seed={current_seed}) -- PRETRAINED protocol\n{'='*80}")

        # Seed the GLOBAL random/np.random state, matching compare_nine_agents.py's
        # own pattern. Required for reproducibility of the QL-family agents
        # specifically: QLearningAgent.choose_action (inherited by every Dyna-Q
        # variant) draws from the unseeded global `random` module, and the
        # Dyna-Q planning functions draw from the unseeded global `np.random` --
        # neither is tied to any of this script's own np.random.default_rng(seed)
        # objects (those only ever controlled config generation). Without this,
        # "seed X" does not reproducibly determine RL-baseline exploration or
        # planning at all, only which button configs are used. Found 2026-08-23
        # while trying to replay a specific stuck (seed, agent, config) case for
        # extended-episode testing and finding it didn't reproduce -- see
        # ai/02-debug.md. OPSRL is unaffected (uses its own seeded agent.rng).
        random.seed(current_seed)
        np.random.seed(current_seed)

        agents = create_all_agents(current_seed, args.max_steps)
        pretrain_rng = np.random.default_rng(int(current_seed) * 1_000_003 + 987)

        print("\nPretraining all non-AIF agents against domain-randomized configs...")
        pretrain_stats = pretrain_all(
            agents, pretrain_rng, args.max_steps,
            args.pretrain_window, args.pretrain_patience, args.pretrain_min_delta,
            args.pretrain_max_episodes, args.pretrain_min_episodes,
        )

        csv_file = writer = None
        if args.log_dir is not None:
            # cfg{episodes_per_config} included alongside ep/step so filenames stay
            # self-describing and non-colliding if this log-dir is ever shared across
            # runs with the same (episodes, max_steps) but a different relocation
            # interval -- mirrors the identical fix in compare_nine_agents.py.
            csv_path = args.log_dir / (
                f"nine_agents_comparison_pretrained_ep{args.episodes}_cfg{args.episodes_per_config}_"
                f"step{args.max_steps}_seed{seed_idx}_{timestamp}.csv"
            )
            csv_file = open(csv_path, 'w', newline='')
            writer = csv.DictWriter(csv_file, fieldnames=['seed', 'agent', 'episode', 'step',
                                                            'action', 'action_name', 'reward', 'agent_pos'])
            writer.writeheader()

        # Load (or generate + save) this seed's scored-protocol config sequence.
        # Loading from a previously-saved --configs-file guarantees a byte-identical
        # map sequence for an agent added later, independent of the RNG state --
        # mirrors the identical mechanism in compare_nine_agents.py; see
        # ai/02-debug.md, 2026-08-24 entry.
        num_configs = (args.episodes + args.episodes_per_config - 1) // args.episodes_per_config
        default_configs_path = (
            args.log_dir / f"nine_agents_configs_pretrained_ep{args.episodes}_cfg{args.episodes_per_config}_"
                            f"step{args.max_steps}_seed{seed_idx}_{timestamp}.json"
            if args.log_dir is not None else None
        )
        configs_path = args.configs_file if args.configs_file is not None else default_configs_path

        if args.configs_file is not None and args.configs_file.exists():
            print(f"\nLoading environment configurations from: {configs_path}")
            with open(configs_path, 'r') as f:
                loaded = json.load(f)
            configs = [{'red_pos': tuple(c['red_pos']), 'blue_pos': tuple(c['blue_pos'])} for c in loaded]
            assert len(configs) == num_configs, (
                f"Loaded {len(configs)} configs from {configs_path}, but this run needs "
                f"{num_configs} (episodes={args.episodes}, episodes_per_config={args.episodes_per_config})."
            )
            print(f"✓ Loaded {len(configs)} configurations (RNG-based generation skipped)")
        else:
            config_rng = np.random.default_rng(current_seed)
            configs = [generate_random_config_np(config_rng) for _ in range(num_configs)]
            if configs_path is not None:
                with open(configs_path, 'w') as f:
                    json.dump(configs, f, indent=2)
                print(f"✓ Saved {num_configs} configurations to: {configs_path}")

        print("\nRunning scored relocation protocol...")
        results = run_scored_protocol(agents, current_seed, args.episodes, args.episodes_per_config,
                                       args.max_steps, writer, configs)

        if csv_file is not None:
            csv_file.close()
            stats_path = args.log_dir / (
                f"pretrain_stats_seed{seed_idx}_{timestamp}.json"
            )
            with open(stats_path, 'w') as f:
                json.dump(pretrain_stats, f, indent=2)

        print(f"\nSeed {current_seed} summary:")
        for name in AGENT_NAMES:
            sr = np.mean([r['success'] for r in results[name]])
            ms = np.mean([r['steps'] for r in results[name]])
            print(f"  {name:16s} success_rate={sr:.3f}  mean_steps={ms:.1f}")


if __name__ == '__main__':
    main()
