"""
Run two random agents playing in the TwoAgentRedBlueButton environment.

Runs experiments across multiple seeds, episodes, and changing map configurations.
Both agents take random actions from their action space.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from environments.RedBlueButton.TwoAgentRedBlueButton import TwoAgentRedBlueButtonEnv


class RandomAgent:
    """Simple agent that takes random actions."""
    
    def __init__(self, action_space_size=6, seed=None):
        self.action_space_size = action_space_size
        self.rng = np.random.default_rng(seed)
    
    def choose_action(self, observation=None):
        """Choose a random action."""
        return self.rng.integers(0, self.action_space_size)
    
    def reset(self, seed=None):
        """Reset the agent's random state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)


def run_episode(env, agent1, agent2, episode_num, max_steps=50, verbose=False, csv_writer=None):
    """Run one episode with two agents."""
    
    # Reset environment
    obs, _ = env.reset()
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}")
        print(f"{'='*80}")
        print(f"Environment: Red at {env.red_button}, Blue at {env.blue_button}")
        print(f"Agent 1 start: {env.agent1_start_pos}, Agent 2 start: {env.agent2_start_pos}")
        print("\nInitial state:")
        env.render()
    
    episode_reward = 0.0
    outcome = 'timeout'
    step = 0
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    
    for step in range(1, max_steps + 1):
        # Get actions from both agents
        action1 = agent1.choose_action(obs)
        action2 = agent2.choose_action(obs)
        actions = (action1, action2)
        
        # Get map before action
        grid = env.render(mode="silent")
        map_str = '|'.join([''.join(row) for row in grid])
        
        if verbose:
            print(f"\n--- Step {step} ---")
            print(f"Agent 1: {action_names[action1]}, Agent 2: {action_names[action2]}")
        
        # Execute actions in environment
        obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        episode_reward += reward
        
        # Log to CSV
        if csv_writer is not None:
            csv_writer.writerow({
                'episode': episode_num,
                'step': step,
                'action1': action1,
                'action1_name': action_names[action1],
                'action2': action2,
                'action2_name': action_names[action2],
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
    # Agent start positions: agent1 at (0,0), agent2 at (2,2)
    # Avoid these positions for buttons
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


def run_seed_experiment(seed, num_episodes, episodes_per_config, max_steps, verbose=False, csv_writer=None):
    """Run experiment for a single seed."""
    
    rng = np.random.default_rng(seed)
    
    # Create agents with different seeds
    agent1 = RandomAgent(action_space_size=6, seed=seed)
    agent2 = RandomAgent(action_space_size=6, seed=seed + 1000)  # Different seed for agent 2
    
    results = []
    configs = []
    
    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    
    # Pre-generate all configs for this seed
    for _ in range(num_configs):
        configs.append(generate_random_config(rng))
    
    env = None
    
    for episode in range(1, num_episodes + 1):
        # Get current config
        config_idx = (episode - 1) // episodes_per_config
        config = configs[config_idx]
        
        # Create environment with current config
        if (episode - 1) % episodes_per_config == 0 or env is None:
            env = TwoAgentRedBlueButtonEnv(
                width=3,
                height=3,
                red_button_pos=config['red_pos'],
                blue_button_pos=config['blue_pos'],
                agent1_start_pos=(0, 0),
                agent2_start_pos=(2, 2),
                max_steps=max_steps * 2,  # Allow more steps in env than our limit
            )
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"SEED {seed} - CONFIG {config_idx + 1}")
                print(f"Red at {config['red_pos']}, Blue at {config['blue_pos']}")
                print(f"Episodes {episode}-{min(episode + episodes_per_config - 1, num_episodes)}")
                print(f"{'='*80}")
        
        result = run_episode(env, agent1, agent2, episode, max_steps=max_steps, 
                           verbose=verbose, csv_writer=csv_writer)
        result['seed'] = seed
        result['config_idx'] = config_idx
        results.append(result)
    
    return results, configs


def main():
    print("="*80)
    print("TWO RANDOM AGENTS - RED BLUE BUTTON ENVIRONMENT")
    print("="*80)
    
    # Parameters
    NUM_SEEDS = 5
    NUM_EPISODES_PER_SEED = 2000
    EPISODES_PER_CONFIG = 250  # Change environment config every N episodes
    MAX_STEPS = 50
    VERBOSE = False  # Set to True for detailed output
    
    # Setup logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"two_random_agents_seeds{NUM_SEEDS}_ep{NUM_EPISODES_PER_SEED}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'seed', 'episode', 'step', 'action1', 'action1_name', 
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
                csv_writer=seed_csv_writer
            )
            
            all_results.extend(results)
            
            # Compute seed summary
            successes = sum(1 for r in results if r['success'])
            success_rate = 100 * successes / len(results)
            avg_reward = np.mean([r['reward'] for r in results])
            avg_steps = np.mean([r['steps'] for r in results])
            
            seed_summaries.append({
                'seed': seed,
                'successes': successes,
                'total': len(results),
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
            })
            
            print(f"\nSeed {seed} Summary:")
            print(f"  Success rate: {successes}/{len(results)} ({success_rate:.1f}%)")
            print(f"  Average reward: {avg_reward:+.2f}")
            print(f"  Average steps: {avg_steps:.1f}")
            
            # Per-config breakdown
            num_configs = (NUM_EPISODES_PER_SEED + EPISODES_PER_CONFIG - 1) // EPISODES_PER_CONFIG
            print(f"\n  Per-config breakdown:")
            for config_idx in range(num_configs):
                config_results = [r for r in results if r['config_idx'] == config_idx]
                if config_results:
                    config_wins = sum(1 for r in config_results if r['success'])
                    config_rate = 100 * config_wins / len(config_results)
                    print(f"    Config {config_idx + 1}: {config_wins}/{len(config_results)} ({config_rate:.1f}%)")
    
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
    
    print(f"\n{'Seed':<8} {'Success Rate':<18} {'Avg Reward':<15} {'Avg Steps':<12}")
    print("-" * 55)
    for s in seed_summaries:
        print(f"{s['seed']:<8} {s['successes']:>4}/{s['total']:<4} ({s['success_rate']:>5.1f}%)  "
              f"{s['avg_reward']:>+8.2f}       {s['avg_steps']:>6.1f}")
    
    # Standard deviation across seeds
    success_rates = [s['success_rate'] for s in seed_summaries]
    avg_rewards = [s['avg_reward'] for s in seed_summaries]
    
    print(f"\nAcross seeds:")
    print(f"  Success rate: {np.mean(success_rates):.1f}% ± {np.std(success_rates):.1f}%")
    print(f"  Average reward: {np.mean(avg_rewards):+.2f} ± {np.std(avg_rewards):.2f}")
    
    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Two Random Agents - Red Blue Button Environment', fontsize=14, fontweight='bold')
    
    # Plot 1: Success rate over episodes (averaged across seeds)
    ax = axes[0, 0]
    episodes = list(range(1, NUM_EPISODES_PER_SEED + 1))
    
    # Compute success rate per episode across seeds
    success_per_episode = []
    for ep in range(NUM_EPISODES_PER_SEED):
        ep_results = [r for r in all_results if all_results.index(r) % NUM_EPISODES_PER_SEED == ep]
        if ep_results:
            success_per_episode.append(np.mean([1 if r['success'] else 0 for r in ep_results]))
        else:
            success_per_episode.append(0)
    
    # Moving average
    window = 10
    if len(success_per_episode) >= window:
        success_ma = np.convolve(success_per_episode, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], success_ma, '-', linewidth=2, color='blue', label=f'MA-{window}')
    ax.scatter(episodes, success_per_episode, alpha=0.3, s=10, color='blue')
    
    # Mark config changes
    num_configs = NUM_EPISODES_PER_SEED // EPISODES_PER_CONFIG
    for i in range(1, num_configs):
        ax.axvline(i * EPISODES_PER_CONFIG + 0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Over Episodes (Averaged Across Seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: Success rate by seed
    ax = axes[0, 1]
    seeds = [s['seed'] for s in seed_summaries]
    rates = [s['success_rate'] for s in seed_summaries]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(seeds)))
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
    ax.hist(rewards, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):+.2f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Steps distribution (wins vs losses)
    ax = axes[1, 1]
    win_steps = [r['steps'] for r in all_results if r['success']]
    lose_steps = [r['steps'] for r in all_results if not r['success']]
    
    if win_steps:
        ax.hist(win_steps, bins=20, alpha=0.6, label=f'Wins (n={len(win_steps)})', color='green')
    if lose_steps:
        ax.hist(lose_steps, bins=20, alpha=0.6, label=f'Losses (n={len(lose_steps)})', color='red')
    ax.set_xlabel('Steps to Completion')
    ax.set_ylabel('Frequency')
    ax.set_title('Steps Distribution (Wins vs Losses)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"two_random_agents_seeds{NUM_SEEDS}_ep{NUM_EPISODES_PER_SEED}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

