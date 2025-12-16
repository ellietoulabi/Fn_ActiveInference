"""
Run two Active Inference agents playing in the TwoAgentRedBlueButton environment
using the FullyCollective paradigm.

FullyCollective: One centralized agent sees the full joint state and outputs
a joint action (a1, a2) encoded as a single integer. This agent controls both agents.

Runs experiments across multiple seeds, episodes, and changing map configurations.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import csv
from datetime import datetime
import argparse
from tqdm import tqdm
from environments.RedBlueButton.TwoAgentRedBlueButton import TwoAgentRedBlueButtonEnv
from generative_models.MA_ActiveInference.RedBlueButton.FullyCollective import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)
from agents.ActiveInference.agent import Agent


def _validate_ma_model(env):
    """
    Sanity checks to catch common A/B/C/D interface mismatches early.
    Raises AssertionError with a helpful message if something is inconsistent.
    """
    # Basic key consistency
    state_factors = list(model_init.states.keys())
    obs_modalities = list(model_init.observations.keys())

    d_config = env_utils.get_D_config_from_env(env)
    D = D_fn(d_config)
    assert set(D.keys()) == set(state_factors), (
        f"D_fn keys mismatch.\nExpected: {state_factors}\nGot: {sorted(D.keys())}"
    )
    for f in state_factors:
        assert len(D[f]) == len(model_init.states[f]), f"D[{f}] has wrong length"
        s = float(np.sum(D[f]))
        assert np.isfinite(s) and abs(s - 1.0) < 1e-6, f"D[{f}] not normalized (sum={s})"

    # A_fn output shape / normalization
    map_state = {f: int(np.argmax(np.array(D[f]))) for f in state_factors}
    A = A_fn(map_state)
    assert set(A.keys()) == set(obs_modalities), (
        f"A_fn keys mismatch.\nExpected: {obs_modalities}\nGot: {sorted(A.keys())}"
    )
    for m in obs_modalities:
        assert len(A[m]) == len(model_init.observations[m]), f"A[{m}] has wrong length"
        s = float(np.sum(A[m]))
        assert np.isfinite(s) and abs(s - 1.0) < 1e-6, f"A[{m}] not normalized (sum={s})"

    # B_fn output keys / normalization (joint action)
    joint_action = env_utils.encode_joint_action(0, 0)
    qs_next = B_fn(D, action=joint_action, width=env.width, height=env.height)
    assert set(qs_next.keys()) == set(state_factors), (
        f"B_fn keys mismatch.\nExpected: {state_factors}\nGot: {sorted(qs_next.keys())}"
    )
    for f in state_factors:
        assert len(qs_next[f]) == len(D[f]), f"B[{f}] has wrong length"
        s = float(np.sum(qs_next[f]))
        assert np.isfinite(s) and abs(s - 1.0) < 1e-6, f"B[{f}] not normalized (sum={s})"

    # C_fn interface sanity
    dummy_obs = {m: 0 for m in obs_modalities}
    prefs = C_fn(dummy_obs)
    assert isinstance(prefs, dict), "C_fn must return a dict"


def create_centralized_agent(env):
    """Create a centralized Active Inference agent that controls both agents."""
    
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    
    # Joint action space: 36 actions (6x6)
    joint_actions = list(range(model_init.N_JOINT_ACTIONS))
    
    agent = Agent(
        A_fn=A_fn,
        B_fn=B_fn,
        C_fn=C_fn,
        D_fn=D_fn,
        state_factors=state_factors,
        state_sizes=state_sizes,
        observation_labels=model_init.observations,
        env_params={'width': 3, 'height': 3},
        observation_state_dependencies=model_init.observation_state_dependencies,
        actions=joint_actions,
        policy_len=3,  # Longer horizon for coordination
        gamma=2.0,  # Policy precision
        alpha=1.0,  # Action precision
        num_iter=16,
    )
    
    # Get config from environment
    d_config = env_utils.get_D_config_from_env(env)
    agent.reset(config=d_config)
    
    return agent


def reset_agent_beliefs(agent, env):
    """Reset agent beliefs for new episode (preserving button position beliefs)."""
    d_config = env_utils.get_D_config_from_env(env)
    # Preserve button-position beliefs across episodes within the same config.
    agent.reset(config=d_config, keep_factors=['red_button_pos', 'blue_button_pos'])


def run_episode(
    env,
    centralized_agent,
    episode_num,
    max_steps=50,
    verbose=False,
    csv_writer=None,
    episode_progress=False,
    show_beliefs=False,
    show_policies=False,
):
    """Run one episode with a centralized AIF agent controlling both agents."""
    
    # Reset environment
    obs, _ = env.reset()
    
    # Reset agent beliefs for new episode
    reset_agent_beliefs(centralized_agent, env)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}")
        print(f"{'='*80}")
        print(f"Environment: Red at {env.red_button}, Blue at {env.blue_button}")
        print("\nInitial state:")
        env.render()
    
    episode_reward = 0.0
    outcome = 'timeout'
    step = 0
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    
    step_iter = range(1, max_steps + 1)
    if episode_progress and not verbose:
        step_iter = tqdm(step_iter, desc=f"Ep {episode_num}", leave=False)

    for step in step_iter:
        # Convert observation to joint model format
        model_obs = env_utils.env_obs_to_model_obs(obs, width=env.width)
        
        # Get joint action from centralized agent
        joint_action_idx = int(centralized_agent.step(model_obs))
        
        # Decode joint action to (a1, a2)
        action1, action2 = env_utils.decode_joint_action(joint_action_idx)
        action1 = int(action1)
        action2 = int(action2)
        actions = (action1, action2)
        
        # Get map before action
        grid = env.render(mode="silent")
        map_str = '|'.join([''.join(row) for row in grid])
        
        if verbose:
            print(f"\n--- Step {step} ---")
            print(f"Joint action: {joint_action_idx} → Agent 1: {action_names.get(action1, str(action1))}, Agent 2: {action_names.get(action2, str(action2))}")
            if show_beliefs:
                qs = centralized_agent.get_state_beliefs()
                print("  Beliefs:")
                for factor, dist in qs.items():
                    dist = np.array(dist)
                    top_idx = dist.argsort()[::-1][:3]
                    pretty = ", ".join([f"{i}:{dist[i]:.2f}" for i in top_idx])
                    print(f"    {factor}: {pretty}")
            if show_policies:
                top = centralized_agent.get_top_policies(top_k=5)
                pretty = "; ".join([f"{p}:{prob:.2f}" for (p_idx, prob, p) in top])
                print(f"  Top policies: {pretty}")
        
        # Execute actions in environment
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        episode_reward += reward
        
        # Log to CSV
        if csv_writer is not None:
            csv_writer.writerow({
                'episode': episode_num,
                'step': step,
                'joint_action': joint_action_idx,
                'action1': action1,
                'action1_name': action_names.get(action1, str(action1)),
                'action2': action2,
                'action2_name': action_names.get(action2, str(action2)),
                'map': map_str,
                'reward': reward,
                'result': info.get('result', 'neutral'),
                'button_pressed': info.get('button_just_pressed', ''),
                'pressed_by': info.get('button_pressed_by', ''),
            })
        
        if verbose:
            env.render()
            print(f"Reward: {reward:+.2f}, Result: {info.get('result', 'neutral')}")
            if info.get('button_just_pressed'):
                print(f"Button pressed: {info['button_just_pressed']} by Agent {info['button_pressed_by']}")
        
        obs = next_obs
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    if verbose:
        status = "✅ WIN" if outcome == 'win' else "❌ FAIL"
        print(f"\nResult: {status} - {outcome} (steps: {step}, reward: {episode_reward:+.2f})")
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win',
    }


def generate_random_config(rng, grid_width=3, grid_height=3):
    """Generate random button positions (avoiding agent start positions)."""
    available_positions = []
    for x in range(grid_width):
        for y in range(grid_height):
            if (x, y) not in [(0, 0), (grid_width-1, grid_height-1)]:
                available_positions.append((x, y))
    
    rng.shuffle(available_positions)
    red_pos = available_positions[0]
    blue_pos = available_positions[1]
    
    return {
        'red_pos': red_pos,
        'blue_pos': blue_pos,
    }


def run_seed_experiment(seed, num_episodes, episodes_per_config, max_steps,
                        verbose=False, csv_writer=None, episode_progress=False,
                        show_beliefs=False, show_policies=False):
    """Run experiment for a single seed."""
    
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    
    results = []
    configs = []
    
    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    
    # Pre-generate all configs for this seed
    for _ in range(num_configs):
        configs.append(generate_random_config(rng))
    
    env = None
    centralized_agent = None
    
    episode_iter = range(1, num_episodes + 1)
    for episode in tqdm(episode_iter, disable=verbose, desc=f"Seed {seed}", leave=False):
        # Get current config
        config_idx = (episode - 1) // episodes_per_config
        config = configs[config_idx]
        
        # Create new environment and agent when config changes
        if (episode - 1) % episodes_per_config == 0 or env is None:
            env = TwoAgentRedBlueButtonEnv(
                width=3,
                height=3,
                red_button_pos=config['red_pos'],
                blue_button_pos=config['blue_pos'],
                agent1_start_pos=(0, 0),
                agent2_start_pos=(2, 2),
                max_steps=max_steps,
            )

            # One-time validation per new config
            _validate_ma_model(env)
            
            # Create fresh agent for new config
            centralized_agent = create_centralized_agent(env)
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"SEED {seed} - CONFIG {config_idx + 1}")
                print(f"Red at {config['red_pos']}, Blue at {config['blue_pos']}")
                print(f"Episodes {episode}-{min(episode + episodes_per_config - 1, num_episodes)}")
                print(f"{'='*80}")
        
        result = run_episode(
            env,
            centralized_agent,
            episode,
            max_steps=max_steps,
            verbose=verbose,
            csv_writer=csv_writer,
            episode_progress=episode_progress,
            show_beliefs=show_beliefs,
            show_policies=show_policies,
        )
        result['seed'] = seed
        result['config_idx'] = config_idx
        results.append(result)
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            recent = results[-100:]
            recent_wins = sum(1 for r in recent if r['success'])
            recent_rate = 100.0 * recent_wins / max(1, len(recent))
            print(f"  Seed {seed}, Episode {episode}/{num_episodes}: "
                  f"Last 100 win rate: {recent_rate:.1f}% ({recent_wins}/100)")
    
    return results, configs


def main():
    print("="*80)
    print("TWO ACTIVE INFERENCE AGENTS - FULLY COLLECTIVE PARADIGM")
    print("="*80)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episodes-per-config", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Generate and save plots at the end")
    parser.add_argument(
        "--episode-progress",
        action="store_true",
        help="Show per-episode progress over steps (tqdm); off by default to keep logs light",
    )
    parser.add_argument(
        "--show-beliefs",
        action="store_true",
        help="When verbose, print top beliefs per factor each step",
    )
    parser.add_argument(
        "--show-policies",
        action="store_true",
        help="When verbose, print top policy posterior entries each step",
    )
    args = parser.parse_args()

    # Parameters
    NUM_SEEDS = args.seeds
    NUM_EPISODES_PER_SEED = args.episodes
    EPISODES_PER_CONFIG = args.episodes_per_config
    MAX_STEPS = args.max_steps
    VERBOSE = args.verbose
    GENERATE_PLOTS = args.plots
    
    # Setup logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"two_aif_agents_fully_collective_seeds{NUM_SEEDS}_ep{NUM_EPISODES_PER_SEED}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'seed', 'episode', 'step', 'joint_action', 'action1', 'action1_name',
        'action2', 'action2_name', 'map', 'reward', 'result',
        'button_pressed', 'pressed_by'
    ])
    csv_writer.writeheader()
    
    print(f"\nExperiment Parameters:")
    print(f"  Number of seeds: {NUM_SEEDS}")
    print(f"  Episodes per seed: {NUM_EPISODES_PER_SEED}")
    print(f"  Episodes per config: {EPISODES_PER_CONFIG}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Logging to: {csv_path}")
    
    # Run experiments across all seeds
    all_results = []
    seed_summaries = []
    
    try:
        for seed_idx, seed in enumerate(range(NUM_SEEDS)):
            print(f"\n{'='*80}")
            print(f"SEED {seed} ({seed_idx + 1}/{NUM_SEEDS})")
            print(f"{'='*80}")
            
            # Add seed to CSV rows
            class SeedCSVWriter:
                def __init__(self, writer, seed):
                    self.writer = writer
                    self.seed = seed
                
                def writerow(self, row):
                    row['seed'] = self.seed
                    self.writer.writerow(row)
            
            seed_csv_writer = SeedCSVWriter(csv_writer, seed)
            
            results, configs = run_seed_experiment(
                seed=seed,
                num_episodes=NUM_EPISODES_PER_SEED,
                episodes_per_config=EPISODES_PER_CONFIG,
                max_steps=MAX_STEPS,
                verbose=VERBOSE,
                csv_writer=seed_csv_writer,
                episode_progress=args.episode_progress,
                show_beliefs=args.show_beliefs and VERBOSE,
                show_policies=args.show_policies and VERBOSE,
            )
            
            all_results.extend(results)
            
            # Compute seed summary
            successes = sum(1 for r in results if r['success'])
            success_rate = 100 * successes / len(results)
            avg_reward = np.mean([r['reward'] for r in results])
            avg_steps = np.mean([r['steps'] for r in results])
            
            # Learning curve: first half vs second half
            mid = len(results) // 2
            first_half_wins = sum(1 for r in results[:mid] if r['success'])
            second_half_wins = sum(1 for r in results[mid:] if r['success'])
            
            seed_summaries.append({
                'seed': seed,
                'successes': successes,
                'total': len(results),
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
                'first_half_wins': first_half_wins,
                'second_half_wins': second_half_wins,
            })
            
            print(f"\nSeed {seed} Summary:")
            print(f"  Success rate: {successes}/{len(results)} ({success_rate:.1f}%)")
            print(f"  Average reward: {avg_reward:+.2f}")
            print(f"  Average steps: {avg_steps:.1f}")
            print(f"  Learning: First half {first_half_wins}/{mid}, Second half {second_half_wins}/{mid}")
    
    finally:
        csv_file.close()
        print(f"\n✓ Log saved to: {csv_path}")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL RESULTS SUMMARY")
    print("="*80)
    
    total_successes = sum(s['successes'] for s in seed_summaries)
    total_episodes = sum(s['total'] for s in seed_summaries)
    overall_success_rate = 100 * total_successes / total_episodes
    overall_avg_reward = np.mean([r['reward'] for r in all_results])
    overall_avg_steps = np.mean([r['steps'] for r in all_results])
    
    print(f"\nTotal episodes: {total_episodes}")
    print(f"Total successes: {total_successes} ({overall_success_rate:.1f}%)")
    print(f"Overall average reward: {overall_avg_reward:+.2f}")
    print(f"Overall average steps: {overall_avg_steps:.1f}")
    
    print(f"\n{'Seed':<6} {'Success Rate':<18} {'Avg Reward':<12} {'1st Half':<10} {'2nd Half':<10}")
    print("-" * 65)
    for s in seed_summaries:
        mid = s['total'] // 2
        print(f"{s['seed']:<6} {s['successes']:>4}/{s['total']:<4} ({s['success_rate']:>5.1f}%)  "
              f"{s['avg_reward']:>+7.2f}     "
              f"{s['first_half_wins']:>3}/{mid:<5}   "
              f"{s['second_half_wins']:>3}/{mid:<5}")
    
    # Learning improvement
    total_first = sum(s['first_half_wins'] for s in seed_summaries)
    total_second = sum(s['second_half_wins'] for s in seed_summaries)
    mid = NUM_EPISODES_PER_SEED // 2
    print(f"\nLearning Progress:")
    print(f"  First half: {total_first}/{NUM_SEEDS * mid} ({100*total_first/(NUM_SEEDS*mid):.1f}%)")
    print(f"  Second half: {total_second}/{NUM_SEEDS * mid} ({100*total_second/(NUM_SEEDS*mid):.1f}%)")
    
    # Generate plots (optional)
    if not GENERATE_PLOTS:
        print("\n(plots disabled; pass --plots to generate figures)")
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        return

    # Make matplotlib robust in headless environments
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Two Active Inference Agents - Fully Collective Paradigm', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Success rate over episodes (learning curve)
    ax = axes[0, 0]
    episodes = list(range(1, NUM_EPISODES_PER_SEED + 1))
    success_per_episode = []
    for ep_idx in range(NUM_EPISODES_PER_SEED):
        ep_results = [all_results[seed_idx * NUM_EPISODES_PER_SEED + ep_idx]
                     for seed_idx in range(NUM_SEEDS)]
        success_per_episode.append(np.mean([1 if r['success'] else 0 for r in ep_results]))
    
    window = 50
    if len(success_per_episode) >= window:
        success_ma = np.convolve(success_per_episode, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], success_ma, '-', linewidth=2, color='blue', 
                label=f'MA-{window}')
    ax.scatter(episodes, success_per_episode, alpha=0.1, s=5, color='blue')
    
    num_configs = NUM_EPISODES_PER_SEED // EPISODES_PER_CONFIG
    for i in range(1, num_configs):
        ax.axvline(i * EPISODES_PER_CONFIG + 0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Learning Curve (Averaged Across Seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: Success rate by seed
    ax = axes[0, 1]
    seeds = [s['seed'] for s in seed_summaries]
    rates = [s['success_rate'] for s in seed_summaries]
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(seeds)))
    ax.bar(seeds, rates, color=colors)
    ax.axhline(np.mean(rates), color='red', linestyle='--', label=f'Mean: {np.mean(rates):.1f}%')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Seed')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Reward distribution
    ax = axes[1, 0]
    rewards = [r['reward'] for r in all_results]
    ax.hist(rewards, bins=20, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(rewards):+.2f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: First half vs second half comparison
    ax = axes[1, 1]
    first_half_rates = [100 * s['first_half_wins'] / (s['total'] // 2) for s in seed_summaries]
    second_half_rates = [100 * s['second_half_wins'] / (s['total'] // 2) for s in seed_summaries]
    
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, first_half_rates, width, label='First Half', color='lightblue', edgecolor='blue')
    ax.bar(x + width/2, second_half_rates, width, label='Second Half', color='darkblue', edgecolor='blue')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Learning Progress (First vs Second Half)')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"two_aif_agents_fully_collective_seeds{NUM_SEEDS}_ep{NUM_EPISODES_PER_SEED}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    plt.close(fig)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

