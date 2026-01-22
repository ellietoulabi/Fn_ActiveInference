"""
Compare Vanilla Dyna-Q vs Dyna-Q with Multiple Recency Biases.

Compares 5 agents on the same environment configurations:
- Vanilla Dyna-Q (uniform random sampling)
- Dyna-Q with recency_decay = 0.99 (slow forgetting)
- Dyna-Q with recency_decay = 0.95 (medium forgetting)
- Dyna-Q with recency_decay = 0.90 (fast forgetting)
- Dyna-Q with recency_decay = 0.85 (very fast forgetting)

All agents experience the EXACT SAME button position changes at the same episodes.
This ensures a fair comparison of their adaptation capabilities.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import csv
from datetime import datetime
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import env_utils
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
        
        # Log to CSV
        if csv_writer is not None:
            action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
            csv_writer.writerow({
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
        
        # (2) Model Learning: Store transition in world model
        agent.update_model(state, action, next_state, reward, terminated)
        
        # (3) Planning: Learn from simulated experience
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
        'model_size': len(agent.model),
        'visited_sa_pairs': len(agent.visited_state_actions),
        'epsilon': agent.epsilon,
        'global_step': stats.get('global_step', 0),
        'avg_experience_age': stats.get('avg_experience_age', 0)
    }


def main():
    print("="*80)
    print("COMPARING: Vanilla Dyna-Q vs Dyna-Q with Multiple Recency Biases")
    print("5 Agents - Same Environment Configurations - Fair Comparison")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 1000
    EPISODES_PER_CONFIG = 20  # Change button positions every 200 episodes
    MAX_STEPS = 50
    PLANNING_STEPS = 10
    RECENCY_DECAYS = [0.99, 0.95, 0.90, 0.85]  # Different recency bias levels
    SEED = 42  # For reproducibility
    
    np.random.seed(SEED)
    
    # Setup CSV logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"dynaq_comparison_ep{NUM_EPISODES}_step{MAX_STEPS}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    # Base filename for Q-tables (without extension)
    base_qtable_name = csv_filename.replace("_comparison_", "_qtable_").replace(".csv", "")
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, 
                                fieldnames=['agent', 'episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    print(f"\nLogging to: {csv_path}")
    
    # Setup all agents (1 vanilla + 4 recency variants)
    print("\n" + "="*80)
    print("SETTING UP AGENTS")
    print("="*80)
    
    agents = []
    agent_names = []
    
    # 1. Vanilla Dyna-Q
    print("\n1. Vanilla Dyna-Q (uniform random sampling)...")
    q_table_path_vanilla = log_dir / f"{base_qtable_name}_vanilla.json"
    vanilla_agent = VanillaDynaQ(
        action_space_size=6,
        planning_steps=PLANNING_STEPS,
        q_table_path=str(q_table_path_vanilla),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.95,
        min_epsilon=0.05,
        load_existing=False
    )
    agents.append(vanilla_agent)
    agent_names.append("Vanilla")
    print("   ✓ Vanilla Dyna-Q ready")
    print(f"   Planning: Uniform random sampling from all (s,a) pairs")
    
    # 2-5. Dyna-Q with different recency biases
    for i, decay in enumerate(RECENCY_DECAYS, start=2):
        print(f"\n{i}. Dyna-Q with Recency Bias (decay={decay})...")
        q_table_path_recency = log_dir / f"{base_qtable_name}_recency{decay}.json"
        recency_agent = RecencyDynaQ(
            action_space_size=6,
            planning_steps=PLANNING_STEPS,
            recency_decay=decay,
            q_table_path=str(q_table_path_recency),
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.95,
            min_epsilon=0.05,
            load_existing=False
        )
        agents.append(recency_agent)
        agent_names.append(f"Recency{decay}")
        print(f"   ✓ Dyna-Q with Recency Bias ready")
        print(f"   Planning: Recency-weighted sampling (decay={decay})")
    
    print(f"\nAll {len(agents)} agents have identical hyperparameters except for planning strategy")
    
    # Pre-generate all environment configurations
    print("\n" + "="*80)
    print("PRE-GENERATING ENVIRONMENT CONFIGURATIONS")
    print("="*80)
    
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
        
        print(f"Config {config_idx+1}: Red at {red_pos} (idx={red_pos_idx}), Blue at {blue_pos} (idx={blue_pos_idx})")
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    
    # Results storage: list of lists (one per agent)
    all_results = [[] for _ in range(len(agents))]
    
    try:
        for episode in range(1, NUM_EPISODES + 1):
            # Determine current configuration
            config_idx = (episode - 1) // EPISODES_PER_CONFIG
            config = configs[config_idx]
            
            # Print configuration change
            if (episode - 1) % EPISODES_PER_CONFIG == 0:
                print(f"\n{'='*80}")
                print(f"CONFIG {config_idx+1}: Episodes {episode}-{min(episode+EPISODES_PER_CONFIG-1, NUM_EPISODES)}")
                print(f"{'='*80}")
                print(f"Red at {config['red_pos']} (idx={config['red_idx']}), "
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
            
            # Print episode summary for all agents
            summary_parts = []
            for agent_name, result in zip(agent_names, episode_results):
                status = "✅" if result['success'] else "❌"
                summary_parts.append(f"{agent_name:11s} {status} (r={result['reward']:+.2f}, s={result['steps']:2d})")
            
            print(f"Ep {episode:4d}: " + " | ".join(summary_parts))
            
    finally:
        csv_file.close()
        print(f"\n✓ Log saved to: {csv_path}")
    
    # Save Q-tables for all agents
    print("\nSaving Q-tables...")
    for agent in agents:
        agent.save_q_table()
    print("✓ All Q-tables saved")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Calculate statistics for all agents
    agent_stats = []
    for agent_name, results in zip(agent_names, all_results):
        successes = sum(1 for r in results if r['success'])
        success_rate = 100 * successes / len(results)
        avg_reward = np.mean([r['reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        agent_stats.append({
            'name': agent_name,
            'successes': successes,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'final_q_size': results[-1]['q_table_size'],
            'final_model_size': results[-1]['model_size']
        })
    
    # Print overall statistics table
    print(f"\n{'Agent':<15} {'Success Rate':<18} {'Avg Reward':<15} {'Avg Steps':<12}")
    print(f"{'-'*60}")
    for stats in agent_stats:
        print(f"{stats['name']:<15} "
              f"{stats['successes']}/{len(all_results[0])} ({stats['success_rate']:.1f}%){'':<5} "
              f"{stats['avg_reward']:+.3f}{'':<10} "
              f"{stats['avg_steps']:.1f}")
    
    # Find best performers
    best_success_rate = max(s['success_rate'] for s in agent_stats)
    best_avg_reward = max(s['avg_reward'] for s in agent_stats)
    best_avg_steps = min(s['avg_steps'] for s in agent_stats)
    
    print(f"\n{'='*60}")
    print("WINNERS:")
    print(f"{'-'*60}")
    best_success_agents = [s['name'] for s in agent_stats if s['success_rate'] == best_success_rate]
    print(f"Best Success Rate: {', '.join(best_success_agents)} ({best_success_rate:.1f}%)")
    
    best_reward_agents = [s['name'] for s in agent_stats if s['avg_reward'] == best_avg_reward]
    print(f"Best Avg Reward:   {', '.join(best_reward_agents)} ({best_avg_reward:+.3f})")
    
    best_steps_agents = [s['name'] for s in agent_stats if s['avg_steps'] == best_avg_steps]
    print(f"Best Avg Steps:    {', '.join(best_steps_agents)} ({best_avg_steps:.1f} steps)")
    
    # Per-configuration analysis
    print(f"\n{'='*80}")
    print(f"PER-CONFIGURATION ANALYSIS (every {EPISODES_PER_CONFIG} episodes)")
    print(f"{'='*80}")
    
    # Header
    header = f"\n{'Config':<8} {'Episodes':<15} "
    for agent_name in agent_names:
        header += f"{agent_name:<15} "
    header += "Best"
    print(header)
    print(f"{'-'*80}")
    
    for config_idx in range(num_configs):
        start = config_idx * EPISODES_PER_CONFIG
        end = start + EPISODES_PER_CONFIG
        
        config_rates = []
        for results in all_results:
            config_results = results[start:end]
            successes = sum(1 for r in config_results if r['success'])
            rate = 100 * successes / len(config_results)
            config_rates.append((successes, len(config_results), rate))
        
        # Find best for this config
        best_rate = max(r[2] for r in config_rates)
        best_agents = [agent_names[i] for i, r in enumerate(config_rates) if r[2] == best_rate]
        
        row = f"{config_idx+1:<8} {start+1}-{end:<11} "
        for successes, total, rate in config_rates:
            row += f"{successes}/{total} ({rate:.0f}%){'':<5} "
        row += f"{', '.join(best_agents)}"
        print(row)
    
    print("\n" + "="*80)
    print(f"\n✓ Comparison complete! Use plot_dynaq_comparison.py to visualize results")
    print(f"  python utils/plotting/plot_dynaq_comparison.py {csv_path} --episodes_per_config {EPISODES_PER_CONFIG}")


if __name__ == "__main__":
    main()

