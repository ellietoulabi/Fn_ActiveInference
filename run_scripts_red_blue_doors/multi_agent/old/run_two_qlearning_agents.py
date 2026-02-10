"""
Run two Q-Learning agents playing in the TwoAgentRedBlueButton environment.

Runs experiments across multiple seeds, episodes, and changing map configurations.
Each agent learns its own Q-table based on its observations.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import csv
import json
import os
import ast
from datetime import datetime
from environments.RedBlueButton.TwoAgentRedBlueButton import TwoAgentRedBlueButtonEnv


class MultiAgentQLearning:
    """
    Q-Learning agent adapted for multi-agent environment.
    
    Each agent observes:
    - Its own position
    - Other agent's position  
    - Whether it's on red/blue button
    - Button states (pressed or not)
    """
    
    def __init__(
        self,
        agent_id,
        action_space_size=6,
        q_table_path=None,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        load_existing=False
    ):
        self.agent_id = agent_id
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table_path = q_table_path
        
        self.q_table = {}
        if load_existing:
            self.load_q_table()
    
    def get_state(self, obs):
        """
        Convert observation to state representation for this agent.
        
        State: (my_pos, other_pos, my_on_red, my_on_blue, red_pressed, blue_pressed)
        """
        if obs is None:
            return None
        
        if self.agent_id == 1:
            my_pos = tuple(obs['agent1_position'])
            other_pos = tuple(obs['agent2_position'])
            my_on_red = obs['agent1_on_red_button']
            my_on_blue = obs['agent1_on_blue_button']
        else:
            my_pos = tuple(obs['agent2_position'])
            other_pos = tuple(obs['agent1_position'])
            my_on_red = obs['agent2_on_red_button']
            my_on_blue = obs['agent2_on_blue_button']
        
        red_pressed = obs['red_button_pressed']
        blue_pressed = obs['blue_button_pressed']
        
        return (my_pos, other_pos, my_on_red, my_on_blue, red_pressed, blue_pressed)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if state is None or state not in self.q_table:
            return np.random.randint(0, self.action_space_size)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        
        return int(np.argmax(self.q_table[state]))
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule."""
        if state is None:
            return
        
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        if next_state is not None and next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space_size)
        
        # Q-learning update
        current_q = self.q_table[state][action]
        
        if next_state is not None:
            max_next_q = np.max(self.q_table[next_state])
        else:
            max_next_q = 0  # Terminal state
        
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[state][action] = new_q
    
    def decay_exploration(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self):
        """Save Q-table to file."""
        if self.q_table_path:
            serializable_q = {
                str(k): v.tolist() for k, v in self.q_table.items()
            }
            try:
                with open(self.q_table_path, 'w') as file:
                    json.dump(serializable_q, file, indent=2)
            except (IOError, OSError) as e:
                print(f"Error saving Q-table: {e}")
    
    def load_q_table(self):
        """Load Q-table from file."""
        if self.q_table_path and os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as file:
                    raw = json.load(file)
                    self.q_table = {
                        ast.literal_eval(k): np.array(v) for k, v in raw.items()
                    }
                print(f"Agent {self.agent_id}: Loaded Q-table with {len(self.q_table)} states")
            except Exception as e:
                print(f"Agent {self.agent_id}: Error loading Q-table: {e}. Starting fresh.")
                self.q_table = {}
    
    def reset(self):
        """Reset agent for new experiment (clear Q-table, reset epsilon)."""
        self.q_table = {}
        self.epsilon = 1.0


def run_episode(env, agent1, agent2, episode_num, max_steps=50, verbose=False, csv_writer=None):
    """Run one episode with two Q-learning agents."""
    
    # Reset environment
    obs, _ = env.reset()
    
    # Get initial states for both agents
    state1 = agent1.get_state(obs)
    state2 = agent2.get_state(obs)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}")
        print(f"{'='*80}")
        print(f"Environment: Red at {env.red_button}, Blue at {env.blue_button}")
        print(f"Agent 1 Q-table: {len(agent1.q_table)} states, ε={agent1.epsilon:.3f}")
        print(f"Agent 2 Q-table: {len(agent2.q_table)} states, ε={agent2.epsilon:.3f}")
        print("\nInitial state:")
        env.render()
    
    episode_reward = 0.0
    outcome = 'timeout'
    step = 0
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    
    for step in range(1, max_steps + 1):
        # Get actions from both agents
        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)
        actions = (action1, action2)
        
        # Get map before action
        grid = env.render(mode="silent")
        map_str = '|'.join([''.join(row) for row in grid])
        
        if verbose:
            print(f"\n--- Step {step} ---")
            print(f"Agent 1: {action_names[action1]}, Agent 2: {action_names[action2]}")
        
        # Execute actions in environment
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        episode_reward += reward
        
        # Get next states
        next_state1 = agent1.get_state(next_obs) if not done else None
        next_state2 = agent2.get_state(next_obs) if not done else None
        
        # Update Q-tables (both agents receive the same team reward)
        agent1.update_q_table(state1, action1, reward, next_state1)
        agent2.update_q_table(state2, action2, reward, next_state2)
        
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
                'agent1_q_size': len(agent1.q_table),
                'agent2_q_size': len(agent2.q_table),
                'agent1_epsilon': agent1.epsilon,
                'agent2_epsilon': agent2.epsilon,
            })
        
        if verbose:
            env.render()
            print(f"Reward: {reward:+.2f}, Result: {info.get('result', 'neutral')}")
            if info.get('button_just_pressed'):
                print(f"Button pressed: {info['button_just_pressed']} by Agent {info['button_pressed_by']}")
        
        # Update states
        state1 = next_state1
        state2 = next_state2
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    # Decay exploration after each episode
    agent1.decay_exploration()
    agent2.decay_exploration()
    
    if verbose:
        status = "✅ WIN" if outcome == 'win' else "❌ FAIL"
        print(f"\nResult: {status} - {outcome} (steps: {step}, reward: {episode_reward:+.2f})")
        print(f"Agent 1: Q-table={len(agent1.q_table)} states, ε={agent1.epsilon:.3f}")
        print(f"Agent 2: Q-table={len(agent2.q_table)} states, ε={agent2.epsilon:.3f}")
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win',
        'agent1_q_size': len(agent1.q_table),
        'agent2_q_size': len(agent2.q_table),
        'agent1_epsilon': agent1.epsilon,
        'agent2_epsilon': agent2.epsilon,
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
                        log_dir, verbose=False, csv_writer=None):
    """Run experiment for a single seed."""
    
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # For agents
    
    # Create Q-learning agents
    agent1 = MultiAgentQLearning(
        agent_id=1,
        action_space_size=6,
        q_table_path=str(log_dir / f"agent1_qtable_seed{seed}.json"),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        load_existing=False
    )
    
    agent2 = MultiAgentQLearning(
        agent_id=2,
        action_space_size=6,
        q_table_path=str(log_dir / f"agent2_qtable_seed{seed}.json"),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        load_existing=False
    )
    
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
                max_steps=max_steps * 2,
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
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            recent = results[-100:]
            recent_wins = sum(1 for r in recent if r['success'])
            print(f"  Seed {seed}, Episode {episode}/{num_episodes}: "
                  f"Last 100 win rate: {recent_wins}% "
                  f"Q1={len(agent1.q_table)}, Q2={len(agent2.q_table)}, "
                  f"ε={agent1.epsilon:.3f}")
    
    # Save Q-tables
    agent1.save_q_table()
    agent2.save_q_table()
    
    return results, configs, agent1, agent2


def main():
    print("="*80)
    print("TWO Q-LEARNING AGENTS - RED BLUE BUTTON ENVIRONMENT")
    print("="*80)
    
    # Parameters
    NUM_SEEDS = 5
    NUM_EPISODES_PER_SEED = 2000
    EPISODES_PER_CONFIG = 250  # Change environment config every N episodes
    MAX_STEPS = 50
    VERBOSE = False
    
    # Setup logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"two_qlearning_agents_seeds{NUM_SEEDS}_ep{NUM_EPISODES_PER_SEED}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'seed', 'episode', 'step', 'action1', 'action1_name',
        'action2', 'action2_name', 'map', 'reward', 'result',
        'button_pressed', 'pressed_by', 'agent1_q_size', 'agent2_q_size',
        'agent1_epsilon', 'agent2_epsilon'
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
            
            results, configs, agent1, agent2 = run_seed_experiment(
                seed=seed,
                num_episodes=NUM_EPISODES_PER_SEED,
                episodes_per_config=EPISODES_PER_CONFIG,
                max_steps=MAX_STEPS,
                log_dir=log_dir,
                verbose=VERBOSE,
                csv_writer=seed_csv_writer
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
                'final_q1_size': len(agent1.q_table),
                'final_q2_size': len(agent2.q_table),
            })
            
            print(f"\nSeed {seed} Summary:")
            print(f"  Success rate: {successes}/{len(results)} ({success_rate:.1f}%)")
            print(f"  Average reward: {avg_reward:+.2f}")
            print(f"  Average steps: {avg_steps:.1f}")
            print(f"  Learning: First half {first_half_wins}/{mid}, Second half {second_half_wins}/{mid}")
            print(f"  Final Q-tables: Agent1={len(agent1.q_table)}, Agent2={len(agent2.q_table)}")
    
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
    
    print(f"\n{'Seed':<6} {'Success Rate':<18} {'Avg Reward':<12} {'1st Half':<10} {'2nd Half':<10} {'Q1 Size':<10} {'Q2 Size':<10}")
    print("-" * 85)
    for s in seed_summaries:
        mid = s['total'] // 2
        print(f"{s['seed']:<6} {s['successes']:>4}/{s['total']:<4} ({s['success_rate']:>5.1f}%)  "
              f"{s['avg_reward']:>+7.2f}     "
              f"{s['first_half_wins']:>3}/{mid:<5}   "
              f"{s['second_half_wins']:>3}/{mid:<5}   "
              f"{s['final_q1_size']:<10} {s['final_q2_size']:<10}")
    
    # Learning improvement
    total_first = sum(s['first_half_wins'] for s in seed_summaries)
    total_second = sum(s['second_half_wins'] for s in seed_summaries)
    mid = NUM_EPISODES_PER_SEED // 2
    print(f"\nLearning Progress:")
    print(f"  First half (episodes 1-{mid}): {total_first}/{NUM_SEEDS * mid} ({100*total_first/(NUM_SEEDS*mid):.1f}%)")
    print(f"  Second half (episodes {mid+1}-{NUM_EPISODES_PER_SEED}): {total_second}/{NUM_SEEDS * mid} ({100*total_second/(NUM_SEEDS*mid):.1f}%)")
    
    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Two Q-Learning Agents - Red Blue Button Environment', fontsize=14, fontweight='bold')
    
    # Plot 1: Success rate over episodes (learning curve)
    ax = axes[0, 0]
    
    # Compute average success rate per episode across seeds
    episodes = list(range(1, NUM_EPISODES_PER_SEED + 1))
    success_per_episode = []
    for ep_idx in range(NUM_EPISODES_PER_SEED):
        ep_results = [all_results[seed_idx * NUM_EPISODES_PER_SEED + ep_idx] 
                     for seed_idx in range(NUM_SEEDS)]
        success_per_episode.append(np.mean([1 if r['success'] else 0 for r in ep_results]))
    
    # Moving average
    window = 20
    if len(success_per_episode) >= window:
        success_ma = np.convolve(success_per_episode, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], success_ma, '-', linewidth=2, color='blue', label=f'MA-{window}')
    ax.scatter(episodes, success_per_episode, alpha=0.2, s=10, color='blue')
    
    # Mark config changes
    num_configs = NUM_EPISODES_PER_SEED // EPISODES_PER_CONFIG
    for i in range(1, num_configs):
        ax.axvline(i * EPISODES_PER_CONFIG + 0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Learning Curve (Averaged Across Seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: Q-table growth
    ax = axes[0, 1]
    
    # Get Q-table sizes over episodes
    q1_sizes = []
    q2_sizes = []
    for ep_idx in range(NUM_EPISODES_PER_SEED):
        ep_results = [all_results[seed_idx * NUM_EPISODES_PER_SEED + ep_idx]
                     for seed_idx in range(NUM_SEEDS)]
        q1_sizes.append(np.mean([r['agent1_q_size'] for r in ep_results]))
        q2_sizes.append(np.mean([r['agent2_q_size'] for r in ep_results]))
    
    ax.plot(episodes, q1_sizes, '-', linewidth=2, color='blue', label='Agent 1')
    ax.plot(episodes, q2_sizes, '-', linewidth=2, color='orange', label='Agent 2')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-table Size (states)')
    ax.set_title('Q-table Growth Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Success rate by seed (bar chart)
    ax = axes[1, 0]
    seeds = [s['seed'] for s in seed_summaries]
    rates = [s['success_rate'] for s in seed_summaries]
    first_half_rates = [100 * s['first_half_wins'] / (s['total'] // 2) for s in seed_summaries]
    second_half_rates = [100 * s['second_half_wins'] / (s['total'] // 2) for s in seed_summaries]
    
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, first_half_rates, width, label='First Half', color='lightblue', edgecolor='blue')
    ax.bar(x + width/2, second_half_rates, width, label='Second Half', color='lightgreen', edgecolor='green')
    ax.axhline(np.mean(rates), color='red', linestyle='--', label=f'Overall Mean: {np.mean(rates):.1f}%')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Learning Progress by Seed (First vs Second Half)')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Epsilon decay
    ax = axes[1, 1]
    epsilon_values = []
    for ep_idx in range(NUM_EPISODES_PER_SEED):
        ep_results = [all_results[seed_idx * NUM_EPISODES_PER_SEED + ep_idx]
                     for seed_idx in range(NUM_SEEDS)]
        epsilon_values.append(np.mean([r['agent1_epsilon'] for r in ep_results]))
    
    ax.plot(episodes, epsilon_values, '-', linewidth=2, color='purple')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (Exploration Rate)')
    ax.set_title('Exploration Rate Decay')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"two_qlearning_agents_seeds{NUM_SEEDS}_ep{NUM_EPISODES_PER_SEED}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

