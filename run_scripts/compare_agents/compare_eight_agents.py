"""
Compare Active Inference vs Q-Learning vs Vanilla Dyna-Q vs Dyna-Q with Recency Biases vs Trajectory Sampling.

Compares 8 agents on the same environment configurations:
- Active Inference (model-based, variational inference)
- Q-Learning (baseline, no planning)
- Vanilla Dyna-Q (uniform random sampling)
- Dyna-Q with recency_decay = 0.99 (slow forgetting)
- Dyna-Q with recency_decay = 0.95 (medium forgetting)
- Dyna-Q with recency_decay = 0.90 (fast forgetting)
- Dyna-Q with recency_decay = 0.85 (very fast forgetting)
- Trajectory Sampling Dyna-Q (multi-step rollouts for fast Q-value propagation)

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
from tqdm import tqdm
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)
from agents.ActiveInference.agent import Agent
from agents.QLearning.qlearning_agent import QLearningAgent
from agents.QLearning.dynaq_agent import DynaQAgent as VanillaDynaQ
from agents.QLearning.dynaq_agent_with_recency_bias import DynaQAgent as RecencyDynaQ
from agents.QLearning.dynaq_agent_trajectory_sampling import DynaQAgent as TrajectorySamplingDynaQ


def grid_to_string(grid):
    """Convert grid array to string representation."""
    return '|'.join([''.join(row) for row in grid])


def run_aif_episode(env, agent, agent_name, episode_num, max_steps=50, csv_writer=None):
    """Run one Active Inference episode."""
    # Reset environment
    env_obs, _ = env.reset()
    
    # Manually update agent's beliefs (don't call reset!)
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
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    
    for step in range(1, max_steps + 1):
        # Get map before action
        grid = env.render(mode="array")
        map_str = grid_to_string(grid)
        
        # Agent perceives, infers, plans, and acts
        action = agent.step(obs_dict)
        
        # Execute action in environment
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Log to CSV
        if csv_writer is not None:
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
        
        obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win',
    }


def run_episode(env, agent, agent_name, episode_num, max_steps=50, csv_writer=None):
    """Run one episode for a Q-learning based agent."""
    
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
        'avg_experience_age': stats.get('avg_experience_age', 0),
        'synthetic_updates': stats.get('synthetic_updates_count', 0)
    }


def main():
    print("="*80)
    print("COMPARING: Active Inference vs Q-Learning vs Vanilla Dyna-Q vs Recency Dyna-Q vs Trajectory Sampling")
    print("8 Agents - Same Environment Configurations - Fair Comparison")
    print("Running with 5 different seeds for statistical reliability")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 1000
    EPISODES_PER_CONFIG = 20  # Change button positions every 20 episodes
    MAX_STEPS = 50
    PLANNING_STEPS = 2
    RECENCY_DECAYS = [0.99, 0.95, 0.90, 0.85]  # Different recency bias levels
    NUM_SEEDS = 5
    BASE_SEED = 42  # Base seed for reproducibility
    
    # Store results across all seeds
    all_seeds_results = []  # List of all_results for each seed
    
    # Setup CSV logging (will include seed column)
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"eight_agents_comparison_ep{NUM_EPISODES}_step{MAX_STEPS}_{NUM_SEEDS}seeds_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    # Base filename for Q-tables (without extension)
    base_qtable_name = csv_filename.replace("_comparison_", "_qtable_").replace(".csv", "")
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, 
                                fieldnames=['seed', 'agent', 'episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    print(f"\nLogging to: {csv_path}")
    
    # Agent names (consistent across all seeds)
    agent_names = ['AIF', 'QLearning', 'Vanilla', 'Recency0.99', 'Recency0.95', 'Recency0.9', 'Recency0.85', 'TrajSampling']
    
    # Run experiments for each seed
    for seed_idx in tqdm(range(NUM_SEEDS), desc="Seeds", unit="seed", position=0, leave=True):
        current_seed = BASE_SEED + seed_idx
        np.random.seed(current_seed)
        
        print(f"\n{'='*80}")
        print(f"SEED {seed_idx + 1}/{NUM_SEEDS} (seed={current_seed})")
        print(f"{'='*80}")
        
        # Setup all agents (1 AIF + 1 QL + 1 vanilla DynaQ + 4 recency variants + 1 trajectory sampling)
        print("\nSetting up agents for this seed...")
        
        agents = []
        
        # 1. Active Inference agent
        print("\n1. Setting up Active Inference agent...")
        state_factors = list(model_init.states.keys())
        state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
        
        aif_agent = Agent(
            A_fn=A_fn,
            B_fn=B_fn,
            C_fn=C_fn,
            D_fn=D_fn,
            state_factors=state_factors,
            state_sizes=state_sizes,
            observation_labels=model_init.observations,
            env_params={'width': model_init.n, 'height': model_init.m},
            actions=list(range(6)),
            policy_len=2,
            gamma=2.0,
            alpha=1.0,
            num_iter=16,
        )
        aif_agent.reset()
        agents.append(aif_agent)
        print("   ✓ Active Inference agent ready")
        
        # 2. Q-Learning (baseline, no planning)
        print("\n2. Q-Learning (baseline, no planning)...")
        q_table_path_ql = log_dir / f"{base_qtable_name}_ql_seed{current_seed}.json"
        ql_agent = QLearningAgent(
            action_space_size=6,
            q_table_path=str(q_table_path_ql),
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.95,
            min_epsilon=0.05,
            load_existing=False
        )
        agents.append(ql_agent)
        print("   ✓ Q-Learning agent ready")
        
        # 3. Vanilla Dyna-Q
        print("\n3. Vanilla Dyna-Q...")
        q_table_path_vanilla = log_dir / f"{base_qtable_name}_vanilla_seed{current_seed}.json"
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
        print("   ✓ Vanilla Dyna-Q agent ready")
        
        # 4-7. Dyna-Q with different recency biases
        for i, decay in enumerate(RECENCY_DECAYS, start=4):
            print(f"\n{i}. Dyna-Q with Recency Bias (decay={decay})...")
            q_table_path_recency = log_dir / f"{base_qtable_name}_recency{decay}_seed{current_seed}.json"
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
            print(f"   ✓ Dyna-Q with Recency Bias (decay={decay}) ready")
        
        # 8. Trajectory Sampling Dyna-Q (long trajectories)
        print("\n8. Trajectory Sampling Dyna-Q...")
        q_table_path_traj = log_dir / f"{base_qtable_name}_traj_seed{current_seed}.json"
        traj_agent = TrajectorySamplingDynaQ(
            action_space_size=6,
            planning_steps=PLANNING_STEPS,
            use_trajectory_sampling=True,
            n_trajectories=10,
            rollout_length=5,
            planning_epsilon=0.1,
            q_table_path=str(q_table_path_traj),
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.95,
            min_epsilon=0.05,
            load_existing=False
        )
        agents.append(traj_agent)
        print("   ✓ Trajectory Sampling Dyna-Q agent ready")
        
        print(f"\n✓ All {len(agents)} agents initialized for seed {current_seed}")
        
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
        
        # Progress bar for episodes
        episode_pbar = tqdm(range(1, NUM_EPISODES + 1), 
                           desc=f"Seed {seed_idx+1}/{NUM_SEEDS} Episodes", 
                           unit="ep", 
                           position=1, 
                           leave=False,
                           ncols=100)
        
        for episode in episode_pbar:
            # Determine current configuration
            config_idx = (episode - 1) // EPISODES_PER_CONFIG
            config = configs[config_idx]
            
            # Print configuration change (only for first seed and first episode of config)
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
            
            # Progress bar for agents (nested, only shows during episode execution)
            agents_iter = zip(agents, agent_names)
            for agent_idx, (agent, agent_name) in enumerate(agents_iter):
                # Reset environment for each agent
                env.reset()
                
                # Run episode (use AIF function for AIF agent, Q-learning function for others)
                if agent_name == 'AIF':
                    result = run_aif_episode(env, agent, agent_name, episode, 
                                           max_steps=MAX_STEPS, csv_writer=csv_writer)
                else:
                    result = run_episode(env, agent, agent_name, episode, 
                                       max_steps=MAX_STEPS, csv_writer=csv_writer)
                all_results[agent_idx].append(result)
                episode_results.append(result)
            
            # Update progress bar with current success rate
            successes = sum(1 for r in episode_results if r['success'])
            success_rate = 100 * successes / len(episode_results)
            episode_pbar.set_postfix({
                'success_rate': f'{success_rate:.1f}%',
                'ep': episode
            })
            
            # Print episode summary every 100 episodes (outside progress bar)
            if episode % 100 == 0:
                episode_pbar.write(f"  Ep {episode}/{NUM_EPISODES} completed for seed {current_seed} - Success rate: {success_rate:.1f}%")
        
        # Close episode progress bar
        episode_pbar.close()
        
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
    print(f"\nUse plot_eight_agents_aggregated.py to visualize results:")
    print(f"  python utils/plotting/plot_eight_agents_aggregated.py {csv_path} --episodes_per_config {EPISODES_PER_CONFIG}")


if __name__ == "__main__":
    main()

