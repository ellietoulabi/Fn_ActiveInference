"""
Compare Active Inference, Q-Learning, and Dyna-Q agents in non-stationary environments.

Runs all three agents on the same sequence of environment configurations
and compares their performance.
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
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)
from agents.ActiveInference.agent import Agent
from agents.QLearning import QLearningAgent, DynaQAgent


def run_aif_episode(env, agent, episode_num, max_steps=50, verbose=False, csv_writer=None):
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
        action = agent.step(obs_dict)
        env_obs, reward, done, _, info = env.step(action)
        episode_reward += reward
        
        # Log step to CSV
        if csv_writer:
            grid = env.render(mode="array")
            map_str = '|'.join([''.join(row) for row in grid])
            csv_writer.writerow({
                'agent': 'AIF',
                'episode': episode_num,
                'step': step,
                'action': action,
                'action_name': action_names.get(action, 'UNKNOWN'),
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


def run_ql_episode(env, agent, episode_num, max_steps=50, verbose=False, csv_writer=None):
    """Run one Q-Learning episode."""
    env_obs, _ = env.reset()
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    state = agent.get_state(obs_dict)
    
    episode_reward = 0.0
    outcome = 'timeout'
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    
    for step in range(1, max_steps + 1):
        action = agent.choose_action(state)
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Log step to CSV
        if csv_writer:
            grid = env.render(mode="array")
            map_str = '|'.join([''.join(row) for row in grid])
            csv_writer.writerow({
                'agent': 'QL',
                'episode': episode_num,
                'step': step,
                'action': action,
                'action_name': action_names.get(action, 'UNKNOWN'),
                'map': map_str,
                'reward': reward
            })
        
        next_obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        next_state = agent.get_state(next_obs_dict) if not done else None
        
        agent.update_q_table(state, action, reward, next_state)
        
        state = next_state
        obs_dict = next_obs_dict
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    agent.decay_exploration()
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win',
        'q_table_size': len(agent.q_table),
        'epsilon': agent.epsilon
    }


def run_dynaq_episode(env, agent, episode_num, max_steps=50, verbose=False, csv_writer=None):
    """Run one Dyna-Q episode."""
    env_obs, _ = env.reset()
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    state = agent.get_state(obs_dict)
    
    episode_reward = 0.0
    outcome = 'timeout'
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    
    for step in range(1, max_steps + 1):
        action = agent.choose_action(state)
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Log step to CSV
        if csv_writer:
            grid = env.render(mode="array")
            map_str = '|'.join([''.join(row) for row in grid])
            csv_writer.writerow({
                'agent': 'DynaQ',
                'episode': episode_num,
                'step': step,
                'action': action,
                'action_name': action_names.get(action, 'UNKNOWN'),
                'map': map_str,
                'reward': reward
            })
        
        next_obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        next_state = agent.get_state(next_obs_dict) if not done else None
        
        # Dyna-Q: Direct RL + Model Learning + Planning
        agent.update_q_table(state, action, reward, next_state)
        agent.update_model(state, action, next_state, reward, terminated)
        agent.planning()
        
        state = next_state
        obs_dict = next_obs_dict
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    agent.decay_exploration()
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win',
        'q_table_size': len(agent.q_table),
        'model_size': len(agent.model),
        'epsilon': agent.epsilon
    }


def main():
    print("="*80)
    print("COMPARING ACTIVE INFERENCE vs Q-LEARNING vs DYNA-Q")
    print("Non-Stationary Environment - Button Positions Change")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 80
    EPISODES_PER_CONFIG = 20  # Change environment every 20 episodes
    MAX_STEPS = 50
    PLANNING_STEPS = 2  # Dyna-Q planning steps
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)
    
    # Setup CSV logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"comparison_3agents_ep{NUM_EPISODES}_step{MAX_STEPS}_plan{PLANNING_STEPS}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    # Base filename for Q-tables (without extension)
    base_qtable_name = csv_filename.replace("_ep", "_qtable_ep").replace(".csv", "")
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['agent', 'episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    print(f"\nExperiment Parameters:")
    print(f"  Total episodes: {NUM_EPISODES}")
    print(f"  Episodes per config: {EPISODES_PER_CONFIG}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Dyna-Q planning steps: {PLANNING_STEPS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Logging to: {csv_path}")
    
    # Setup Active Inference agent
    print("\n" + "-"*80)
    print("Setting up Active Inference agent...")
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
    print("✓ Active Inference agent ready")
    
    # Setup Q-Learning agent
    print("\nSetting up Q-Learning agent...")
    q_table_path_ql = log_dir / f"{base_qtable_name}_ql.json"
    ql_agent = QLearningAgent(
        action_space_size=6,
        q_table_path=str(q_table_path_ql),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,  # Slower decay for 200 episodes (was 0.95)
        min_epsilon=0.05,
        load_existing=False
    )
    print("✓ Q-Learning agent ready")
    print(f"  Q-table will be saved to: {q_table_path_ql}")
    
    # Setup Dyna-Q agent
    print("\nSetting up Dyna-Q agent...")
    q_table_path_dynaq = log_dir / f"{base_qtable_name}_dynaq.json"
    dynaq_agent = DynaQAgent(
        action_space_size=6,
        planning_steps=PLANNING_STEPS,
        q_table_path=str(q_table_path_dynaq),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,  # Slower decay for 200 episodes (was 0.95)
        min_epsilon=0.05,
        load_existing=False
    )
    print(f"✓ Dyna-Q agent ready (planning_steps={PLANNING_STEPS})")
    print(f"  Q-table will be saved to: {q_table_path_dynaq}")
    
    # Generate environment configurations
    print("\n" + "-"*80)
    print("Generating environment configurations...")
    num_configs = NUM_EPISODES // EPISODES_PER_CONFIG
    configs = []
    
    for config_idx in range(num_configs):
        available_positions = list(range(1, 9))
        np.random.shuffle(available_positions)
        red_pos_idx = available_positions[0]
        blue_pos_idx = available_positions[1]
        red_pos = (red_pos_idx // 3, red_pos_idx % 3)
        blue_pos = (blue_pos_idx // 3, blue_pos_idx % 3)
        configs.append({
            'red_pos': red_pos,
            'blue_pos': blue_pos,
            'red_idx': red_pos_idx,
            'blue_idx': blue_pos_idx
        })
        print(f"  Config {config_idx+1}: Red at {red_pos} (idx={red_pos_idx}), "
              f"Blue at {blue_pos} (idx={blue_pos_idx})")
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    
    aif_results = []
    ql_results = []
    dynaq_results = []
    
    try:
        for episode in range(1, NUM_EPISODES + 1):
            # Get current config
            config_idx = (episode - 1) // EPISODES_PER_CONFIG
            config = configs[config_idx]
            
            # Create environment
            env = SingleAgentRedBlueButtonEnv(
                width=3,
                height=3,
                red_button_pos=config['red_pos'],
                blue_button_pos=config['blue_pos'],
                agent_start_pos=(0, 0),
                max_steps=100
            )
            
            # Print episode header
            if (episode - 1) % EPISODES_PER_CONFIG == 0:
                print(f"\n{'='*80}")
                print(f"CONFIG {config_idx + 1}: Episodes {episode}-{min(episode+EPISODES_PER_CONFIG-1, NUM_EPISODES)}")
                print(f"Red at {config['red_pos']}, Blue at {config['blue_pos']}")
                print(f"{'='*80}")
            
            # Progress indicator
            progress_pct = (episode / NUM_EPISODES) * 100
            print(f"\n[{progress_pct:5.1f}%] Episode {episode}/{NUM_EPISODES}:")
            
            # Run all three agents on same environment
            aif_result = run_aif_episode(env, aif_agent, episode, MAX_STEPS, verbose=False, csv_writer=csv_writer)
            aif_results.append(aif_result)
            print(f"  AIF:    {'✅ WIN' if aif_result['success'] else '❌ FAIL'} "
                  f"- {aif_result['steps']:2d} steps, reward: {aif_result['reward']:+.2f}")
            
            ql_result = run_ql_episode(env, ql_agent, episode, MAX_STEPS, verbose=False, csv_writer=csv_writer)
            ql_results.append(ql_result)
            print(f"  QL:     {'✅ WIN' if ql_result['success'] else '❌ FAIL'} "
                  f"- {ql_result['steps']:2d} steps, reward: {ql_result['reward']:+.2f}, "
                  f"ε={ql_result['epsilon']:.3f}, Q-size={ql_result['q_table_size']}")
            
            dynaq_result = run_dynaq_episode(env, dynaq_agent, episode, MAX_STEPS, verbose=False, csv_writer=csv_writer)
            dynaq_results.append(dynaq_result)
            print(f"  Dyna-Q: {'✅ WIN' if dynaq_result['success'] else '❌ FAIL'} "
                  f"- {dynaq_result['steps']:2d} steps, reward: {dynaq_result['reward']:+.2f}, "
                  f"ε={dynaq_result['epsilon']:.3f}, Q-size={dynaq_result['q_table_size']}, Model={dynaq_result['model_size']}")
            
            # Show running statistics every 20 episodes
            if episode % 20 == 0:
                aif_wins = sum(1 for r in aif_results if r['success'])
                ql_wins = sum(1 for r in ql_results if r['success'])
                dynaq_wins = sum(1 for r in dynaq_results if r['success'])
                
                aif_rate = 100 * aif_wins / len(aif_results)
                ql_rate = 100 * ql_wins / len(ql_results)
                dynaq_rate = 100 * dynaq_wins / len(dynaq_results)
                
                aif_avg_r = np.mean([r['reward'] for r in aif_results])
                ql_avg_r = np.mean([r['reward'] for r in ql_results])
                dynaq_avg_r = np.mean([r['reward'] for r in dynaq_results])
                
                print(f"\n  📊 PROGRESS SUMMARY (Episodes 1-{episode}):")
                print(f"     {'Agent':<15} {'Success Rate':<15} {'Avg Reward':<15}")
                print(f"     {'-'*45}")
                print(f"     {'AIF':<15} {aif_wins}/{episode} ({aif_rate:>5.1f}%)   {aif_avg_r:>+7.2f}")
                print(f"     {'Q-Learning':<15} {ql_wins}/{episode} ({ql_rate:>5.1f}%)   {ql_avg_r:>+7.2f}")
                print(f"     {'Dyna-Q':<15} {dynaq_wins}/{episode} ({dynaq_rate:>5.1f}%)   {dynaq_avg_r:>+7.2f}")
    finally:
        csv_file.close()
        print(f"\n✓ Log saved to: {csv_path}")
    
    # Analysis
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    aif_successes = sum(1 for r in aif_results if r['success'])
    ql_successes = sum(1 for r in ql_results if r['success'])
    dynaq_successes = sum(1 for r in dynaq_results if r['success'])
    
    aif_success_rate = 100 * aif_successes / len(aif_results)
    ql_success_rate = 100 * ql_successes / len(ql_results)
    dynaq_success_rate = 100 * dynaq_successes / len(dynaq_results)
    
    aif_avg_reward = np.mean([r['reward'] for r in aif_results])
    ql_avg_reward = np.mean([r['reward'] for r in ql_results])
    dynaq_avg_reward = np.mean([r['reward'] for r in dynaq_results])
    
    aif_avg_steps = np.mean([r['steps'] for r in aif_results])
    ql_avg_steps = np.mean([r['steps'] for r in ql_results])
    dynaq_avg_steps = np.mean([r['steps'] for r in dynaq_results])
    
    print(f"\n{'Metric':<25} {'Active Inference':<20} {'Q-Learning':<20} {'Dyna-Q':<20}")
    print("-" * 85)
    print(f"{'Success Rate':<25} {aif_success_rate:>6.1f}%             {ql_success_rate:>6.1f}%             {dynaq_success_rate:>6.1f}%")
    print(f"{'Total Wins':<25} {aif_successes:>6}/{len(aif_results):<13} {ql_successes:>6}/{len(ql_results):<13} {dynaq_successes:>6}/{len(dynaq_results)}")
    print(f"{'Average Reward':<25} {aif_avg_reward:>+7.2f}             {ql_avg_reward:>+7.2f}             {dynaq_avg_reward:>+7.2f}")
    print(f"{'Average Steps':<25} {aif_avg_steps:>7.1f}             {ql_avg_steps:>7.1f}             {dynaq_avg_steps:>7.1f}")
    
    # Per-configuration analysis
    print(f"\n\nPer-Configuration Performance:")
    print(f"{'Config':<8} {'Episodes':<15} {'AIF':<15} {'QL':<15} {'Dyna-Q':<15}")
    print("-" * 75)
    
    for config_idx in range(num_configs):
        start = config_idx * EPISODES_PER_CONFIG
        end = start + EPISODES_PER_CONFIG
        
        aif_config = aif_results[start:end]
        ql_config = ql_results[start:end]
        dynaq_config = dynaq_results[start:end]
        
        aif_wins = sum(1 for r in aif_config if r['success'])
        ql_wins = sum(1 for r in ql_config if r['success'])
        dynaq_wins = sum(1 for r in dynaq_config if r['success'])
        
        aif_rate = 100 * aif_wins / len(aif_config)
        ql_rate = 100 * ql_wins / len(ql_config)
        dynaq_rate = 100 * dynaq_wins / len(dynaq_config)
        
        print(f"{config_idx+1:<8} {start+1:>3}-{end:<11} "
              f"{aif_wins:>2}/{len(aif_config)} ({aif_rate:>5.1f}%)  "
              f"{ql_wins:>2}/{len(ql_config)} ({ql_rate:>5.1f}%)  "
              f"{dynaq_wins:>2}/{len(dynaq_config)} ({dynaq_rate:>5.1f}%)")
    
    # Plot results
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'Active Inference vs Q-Learning vs Dyna-Q (planning={PLANNING_STEPS}): Non-Stationary Environment', 
                 fontsize=14, fontweight='bold')
    
    episodes = list(range(1, NUM_EPISODES + 1))
    
    # Plot 1: Success rate over time
    ax = axes[0, 0]
    aif_success = [1 if r['success'] else 0 for r in aif_results]
    ql_success = [1 if r['success'] else 0 for r in ql_results]
    dynaq_success = [1 if r['success'] else 0 for r in dynaq_results]
    
    # Moving average
    window = 10
    aif_ma = np.convolve(aif_success, np.ones(window)/window, mode='valid')
    ql_ma = np.convolve(ql_success, np.ones(window)/window, mode='valid')
    dynaq_ma = np.convolve(dynaq_success, np.ones(window)/window, mode='valid')
    
    ax.plot(episodes[window-1:], aif_ma, '-', linewidth=2.5, color='blue', label=f'AIF (MA-{window})')
    ax.plot(episodes[window-1:], ql_ma, '-', linewidth=2.5, color='orange', label=f'QL (MA-{window})')
    ax.plot(episodes[window-1:], dynaq_ma, '-', linewidth=2.5, color='green', label=f'Dyna-Q (MA-{window})')
    
    # Mark config changes
    for i in range(1, num_configs):
        ax.axvline(i * EPISODES_PER_CONFIG + 0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Over Time (Moving Average)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: Cumulative reward
    ax = axes[0, 1]
    aif_cumulative = np.cumsum([r['reward'] for r in aif_results])
    ql_cumulative = np.cumsum([r['reward'] for r in ql_results])
    dynaq_cumulative = np.cumsum([r['reward'] for r in dynaq_results])
    
    ax.plot(episodes, aif_cumulative, '-', linewidth=2.5, color='blue', label='Active Inference')
    ax.plot(episodes, ql_cumulative, '-', linewidth=2.5, color='orange', label='Q-Learning')
    ax.plot(episodes, dynaq_cumulative, '-', linewidth=2.5, color='green', label='Dyna-Q')
    
    for i in range(1, num_configs):
        ax.axvline(i * EPISODES_PER_CONFIG + 0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Steps per episode (moving average)
    ax = axes[1, 0]
    aif_steps = [r['steps'] for r in aif_results]
    ql_steps = [r['steps'] for r in ql_results]
    dynaq_steps = [r['steps'] for r in dynaq_results]
    
    # Moving average for steps
    aif_steps_ma = np.convolve(aif_steps, np.ones(window)/window, mode='valid')
    ql_steps_ma = np.convolve(ql_steps, np.ones(window)/window, mode='valid')
    dynaq_steps_ma = np.convolve(dynaq_steps, np.ones(window)/window, mode='valid')
    
    ax.plot(episodes[window-1:], aif_steps_ma, '-', linewidth=2.5, color='blue', label='Active Inference')
    ax.plot(episodes[window-1:], ql_steps_ma, '-', linewidth=2.5, color='orange', label='Q-Learning')
    ax.plot(episodes[window-1:], dynaq_steps_ma, '-', linewidth=2.5, color='green', label='Dyna-Q')
    
    for i in range(1, num_configs):
        ax.axvline(i * EPISODES_PER_CONFIG + 0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Completion')
    ax.set_title(f'Efficiency (Moving Average, window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Success rate per configuration
    ax = axes[1, 1]
    config_labels = [f'Config {i+1}' for i in range(num_configs)]
    aif_config_rates = []
    ql_config_rates = []
    dynaq_config_rates = []
    
    for config_idx in range(num_configs):
        start = config_idx * EPISODES_PER_CONFIG
        end = start + EPISODES_PER_CONFIG
        aif_wins = sum(1 for r in aif_results[start:end] if r['success'])
        ql_wins = sum(1 for r in ql_results[start:end] if r['success'])
        dynaq_wins = sum(1 for r in dynaq_results[start:end] if r['success'])
        aif_config_rates.append(100 * aif_wins / EPISODES_PER_CONFIG)
        ql_config_rates.append(100 * ql_wins / EPISODES_PER_CONFIG)
        dynaq_config_rates.append(100 * dynaq_wins / EPISODES_PER_CONFIG)
    
    x = np.arange(num_configs)
    width = 0.25
    ax.bar(x - width, aif_config_rates, width, label='Active Inference', color='blue', alpha=0.7)
    ax.bar(x, ql_config_rates, width, label='Q-Learning', color='orange', alpha=0.7)
    ax.bar(x + width, dynaq_config_rates, width, label='Dyna-Q', color='green', alpha=0.7)
    
    ax.set_xlabel('Environment Configuration')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"comparison_3agents_ep{NUM_EPISODES}_plan{PLANNING_STEPS}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    plt.show()
    
    # Save Q-tables
    print("\n" + "="*80)
    print("SAVING Q-TABLES")
    print("="*80)
    print(f"Saving Q-Learning Q-table to '{q_table_path_ql}'...")
    ql_agent.save_q_table()
    print(f"Saving Dyna-Q Q-table to '{q_table_path_dynaq}'...")
    dynaq_agent.save_q_table()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    # Print winner
    winner = "Active Inference" if aif_success_rate > max(ql_success_rate, dynaq_success_rate) else \
             "Dyna-Q" if dynaq_success_rate > ql_success_rate else "Q-Learning"
    print(f"\n🏆 WINNER: {winner} ({max(aif_success_rate, ql_success_rate, dynaq_success_rate):.1f}% success rate)")


if __name__ == "__main__":
    main()

