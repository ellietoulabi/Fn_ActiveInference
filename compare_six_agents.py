"""
Compare Q-Learning vs Vanilla Dyna-Q vs Dyna-Q with Multiple Recency Biases.

Compares 6 agents on the same environment configurations:
- Q-Learning (baseline, no planning)
- Vanilla Dyna-Q (uniform random sampling)
- Dyna-Q with recency_decay = 0.99 (slow forgetting)
- Dyna-Q with recency_decay = 0.95 (medium forgetting)
- Dyna-Q with recency_decay = 0.90 (fast forgetting)
- Dyna-Q with recency_decay = 0.85 (very fast forgetting)

All agents experience the EXACT SAME button position changes at the same episodes.
This ensures a fair comparison of their adaptation capabilities.
"""

import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import env_utils
from agents.QLearning.qlearning_agent import QLearningAgent
from agents.QLearning.dynaq_agent import DynaQAgent as VanillaDynaQ
from agents.QLearning.dynaq_agent_with_recency_bias import DynaQAgent as RecencyDynaQ


def grid_to_string(grid):
    """Convert grid array to string representation."""
    return '|'.join([''.join(row) for row in grid])


def run_episode(env, agent, agent_name, episode_num, max_steps=50, csv_writer=None):
    """Run one episode for an agent."""
    
    # Reset environment
    env_obs, _ = env.reset()
    
    # Convert to model observation format
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    state = agent.get_state(obs_dict)
    
    episode_reward = 0.0
    outcome = 'timeout'
    step = 0
    
    for step in range(1, max_steps + 1):
        # Get map before action
        grid = env.render(mode="array")
        map_str = grid_to_string(grid)
        
        # Choose action
        action = agent.choose_action(state)
        
        # Execute action in environment
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Log to CSV (note: csv_writer is passed from main, we need to get seed from somewhere)
        # We'll pass seed as a parameter to this function
        if csv_writer is not None:
            action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
            # Get seed from csv_writer if available (we'll add it as metadata)
            csv_writer.writerow({
                'seed': getattr(csv_writer, 'current_seed', 0),
                'agent': agent_name,
                'episode': episode_num,
                'step': step,
                'action': action,
                'action_name': action_names[action],
                'map': map_str,
                'reward': reward
            })
        
        # Convert to model observation format
        next_obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        next_state = agent.get_state(next_obs_dict) if not done else None
        
        # (1) Direct RL: Update Q-table from real experience
        agent.update_q_table(state, action, reward, next_state)
        
        # (2) For Dyna-Q agents: Model Learning + Planning
        if hasattr(agent, 'update_model'):  # Dyna-Q agents have this method
            agent.update_model(state, action, next_state, reward, terminated)
            agent.planning()
        
        # Update for next iteration
        state = next_state
        obs_dict = next_obs_dict
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    # Decay exploration after each episode
    agent.decay_exploration()
    
    # Get stats
    stats = agent.get_stats()
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win',
        'q_table_size': len(agent.q_table),
        'model_size': stats.get('model_size', 0),
        'visited_sa_pairs': stats.get('visited_state_actions', 0),
        'epsilon': agent.epsilon,
        'global_step': stats.get('global_step', 0),
        'avg_experience_age': stats.get('avg_experience_age', 0)
    }


def main():
    print("="*80)
    print("COMPARING: Q-Learning vs Vanilla Dyna-Q vs Dyna-Q with Multiple Recency Biases")
    print("6 Agents - Same Environment Configurations - Fair Comparison")
    print("Running with 5 different seeds for statistical reliability")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 1000
    EPISODES_PER_CONFIG = 20  # Change button positions every 20 episodes
    MAX_STEPS = 50
    PLANNING_STEPS = 10
    RECENCY_DECAYS = [0.99, 0.95, 0.90, 0.85]  # Different recency bias levels
    NUM_SEEDS = 5
    BASE_SEED = 42  # Base seed for reproducibility
    
    # Store results across all seeds
    all_seeds_results = []  # List of all_results for each seed
    
    # Setup CSV logging (will include seed column)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"six_agents_comparison_ep{NUM_EPISODES}_step{MAX_STEPS}_{NUM_SEEDS}seeds_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, 
                                fieldnames=['seed', 'agent', 'episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    print(f"\nLogging to: {csv_path}")
    
    # Agent names (consistent across all seeds)
    agent_names = ['QLearning', 'Vanilla', 'Recency0.99', 'Recency0.95', 'Recency0.9', 'Recency0.85']
    
    # Run experiments for each seed
    for seed_idx in range(NUM_SEEDS):
        current_seed = BASE_SEED + seed_idx
        np.random.seed(current_seed)
        
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx + 1}/{NUM_SEEDS} (seed={current_seed})")
        print(f"{'='*80}")
        
        # Setup all agents (1 QL + 1 vanilla DynaQ + 4 recency variants)
        print("\nSetting up agents for this seed...")
        
        agents = []
        
        # 1. Q-Learning (baseline, no planning)
        ql_agent = QLearningAgent(
            action_space_size=6,
            q_table_path=f"q_table_ql_seed{current_seed}.json",
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.95,
            min_epsilon=0.05,
            load_existing=False
        )
        agents.append(ql_agent)
        
        # 2. Vanilla Dyna-Q
        vanilla_agent = VanillaDynaQ(
            action_space_size=6,
            planning_steps=PLANNING_STEPS,
            q_table_path=f"q_table_vanilla_seed{current_seed}.json",
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.95,
            min_epsilon=0.05,
            load_existing=False
        )
        agents.append(vanilla_agent)
        
        # 3-6. Dyna-Q with different recency biases
        for decay in RECENCY_DECAYS:
            recency_agent = RecencyDynaQ(
                action_space_size=6,
                planning_steps=PLANNING_STEPS,
                recency_decay=decay,
                q_table_path=f"q_table_recency{decay}_seed{current_seed}.json",
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=1.0,
                epsilon_decay=0.95,
                min_epsilon=0.05,
                load_existing=False
            )
            agents.append(recency_agent)
        
        print(f"✓ All {len(agents)} agents initialized for seed {current_seed}")
        
        # Pre-generate all environment configurations for this seed
        print("\nGenerating environment configurations...")
        
        num_configs = NUM_EPISODES // EPISODES_PER_CONFIG
        configs = []
        
        for config_idx in range(num_configs):
            # Generate random button positions (avoid agent start position 0)
            available_positions = list(range(1, 9))
            np.random.shuffle(available_positions)
            red_pos_idx = available_positions[0]
            blue_pos_idx = available_positions[1]
            
            # Convert to (y, x) for environment (row, col)
            red_pos = (red_pos_idx // 3, red_pos_idx % 3)
            blue_pos = (blue_pos_idx // 3, blue_pos_idx % 3)
            
            configs.append({
                'red_pos': red_pos,
                'blue_pos': blue_pos,
                'red_idx': red_pos_idx,
                'blue_idx': blue_pos_idx
            })
        
        print(f"✓ Generated {num_configs} configurations")
        
        # Run experiments for this seed
        print(f"\nRunning {NUM_EPISODES} episodes...")
        
        # Results storage: list of lists (one per agent)
        all_results = [[] for _ in range(len(agents))]
        
        # Set current seed in csv_writer for logging
        csv_writer.current_seed = current_seed
        
        for episode in range(1, NUM_EPISODES + 1):
            # Determine current configuration
            config_idx = (episode - 1) // EPISODES_PER_CONFIG
            config = configs[config_idx]
            
            # Print configuration change (only for first seed)
            if seed_idx == 0 and (episode - 1) % EPISODES_PER_CONFIG == 0:
                print(f"\nConfig {config_idx+1}: Red at {config['red_pos']} (idx={config['red_idx']}), "
                      f"Blue at {config['blue_pos']} (idx={config['blue_idx']})")
            
            # Create environment with current configuration
            env = SingleAgentRedBlueButtonEnv(
                width=3,
                height=3,
                red_button_pos=config['red_pos'],
                blue_button_pos=config['blue_pos'],
                agent_start_pos=(0, 0),
                max_steps=100
            )
            
            # Run all agents on the same environment
            episode_results = []
            for agent_idx, (agent, agent_name) in enumerate(zip(agents, agent_names)):
                # Reset environment for each agent
                env.reset()
                
                # Run episode
                result = run_episode(env, agent, agent_name, episode, 
                                    max_steps=MAX_STEPS, csv_writer=csv_writer)
                all_results[agent_idx].append(result)
                episode_results.append(result)
            
            # Print episode summary every 100 episodes
            if episode % 100 == 0:
                print(f"  Ep {episode}/{NUM_EPISODES} completed for seed {current_seed}")
        
        # Store results for this seed
        all_seeds_results.append(all_results)
        print(f"\n✓ Seed {current_seed} completed")
    
    # Close CSV file
    csv_file.close()
    print(f"\n✓ Log saved to: {csv_path}")
    
    # Aggregate results across all seeds
    print("\n" + "="*80)
    print(f"AGGREGATED RESULTS ACROSS {NUM_SEEDS} SEEDS")
    print("="*80)
    
    # Calculate mean and std for each agent across all seeds
    agent_agg_stats = []
    for agent_idx, agent_name in enumerate(agent_names):
        # Collect metrics from all seeds for this agent
        success_rates = []
        avg_rewards = []
        avg_steps = []
        
        for seed_results in all_seeds_results:
            results = seed_results[agent_idx]
            successes = sum(1 for r in results if r['success'])
            success_rate = 100 * successes / len(results)
            avg_reward = np.mean([r['reward'] for r in results])
            avg_step = np.mean([r['steps'] for r in results])
            
            success_rates.append(success_rate)
            avg_rewards.append(avg_reward)
            avg_steps.append(avg_step)
        
        agent_agg_stats.append({
            'name': agent_name,
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'mean_reward': np.mean(avg_rewards),
            'std_reward': np.std(avg_rewards),
            'mean_steps': np.mean(avg_steps),
            'std_steps': np.std(avg_steps)
        })
    
    # Print aggregated statistics table
    print(f"\n{'Agent':<15} {'Success Rate (%)':<25} {'Avg Reward':<25} {'Avg Steps':<20}")
    print(f"{'':< 15} {'(mean ± std)':<25} {'(mean ± std)':<25} {'(mean ± std)':<20}")
    print(f"{'-'*85}")
    for stats in agent_agg_stats:
        print(f"{stats['name']:<15} "
              f"{stats['mean_success_rate']:5.1f} ± {stats['std_success_rate']:4.1f}{'':<14} "
              f"{stats['mean_reward']:+6.3f} ± {stats['std_reward']:5.3f}{'':<10} "
              f"{stats['mean_steps']:5.1f} ± {stats['std_steps']:4.1f}")
    
    # Find best performers (by mean)
    best_success_rate = max(s['mean_success_rate'] for s in agent_agg_stats)
    best_avg_reward = max(s['mean_reward'] for s in agent_agg_stats)
    best_avg_steps = min(s['mean_steps'] for s in agent_agg_stats)
    
    print(f"\n{'='*85}")
    print("BEST PERFORMERS (by mean across seeds):")
    print(f"{'-'*85}")
    best_success_agents = [s['name'] for s in agent_agg_stats if s['mean_success_rate'] == best_success_rate]
    print(f"Best Success Rate: {', '.join(best_success_agents)} ({best_success_rate:.1f}% ± "
          f"{[s['std_success_rate'] for s in agent_agg_stats if s['name'] == best_success_agents[0]][0]:.1f}%)")
    
    best_reward_agents = [s['name'] for s in agent_agg_stats if s['mean_reward'] == best_avg_reward]
    print(f"Best Avg Reward:   {', '.join(best_reward_agents)} ({best_avg_reward:+.3f} ± "
          f"{[s['std_reward'] for s in agent_agg_stats if s['name'] == best_reward_agents[0]][0]:.3f})")
    
    best_steps_agents = [s['name'] for s in agent_agg_stats if s['mean_steps'] == best_avg_steps]
    print(f"Best Avg Steps:    {', '.join(best_steps_agents)} ({best_avg_steps:.1f} ± "
          f"{[s['std_steps'] for s in agent_agg_stats if s['name'] == best_steps_agents[0]][0]:.1f} steps)")
    
    print("\n" + "="*85)
    print(f"\n✓ Comparison complete with {NUM_SEEDS} seeds for statistical reliability!")
    print(f"✓ Log saved to: {csv_path}")
    print(f"\nUse plot_six_agents_comparison.py to visualize results:")
    print(f"  python plot_six_agents_comparison.py {csv_path} --episodes_per_config {EPISODES_PER_CONFIG}")


if __name__ == "__main__":
    main()

