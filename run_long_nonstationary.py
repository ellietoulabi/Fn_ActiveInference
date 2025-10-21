"""
Run 500 episodes with non-stationary button positions.
Log all data to CSV and create learning analysis plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)
from agents.ActiveInference.agent import Agent


def run_episode(env, agent, episode_num, max_steps=50, log_data=None):
    """Run one episode and log step data."""
    
    # Reset environment
    env_obs, _ = env.reset()
    
    # Get environment config for logging
    env_red_idx = env.red_button[1] * 3 + env.red_button[0]
    env_blue_idx = env.blue_button[1] * 3 + env.blue_button[0]
    
    # Manually update agent's beliefs
    agent.qs['agent_pos'] = np.zeros(9)
    agent.qs['agent_pos'][0] = 1.0
    agent.qs['red_button_state'] = np.array([1.0, 0.0])
    agent.qs['blue_button_state'] = np.array([1.0, 0.0])
    
    # Keep button position beliefs from previous episode
    agent.action = 5  # NOOP to preserve beliefs
    agent.prev_actions = []
    agent.curr_timestep = 0
    
    # Convert initial observation
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    
    episode_reward = 0.0
    outcome = 'timeout'
    
    for step in range(1, max_steps + 1):
        # Agent acts
        action = agent.step(obs_dict)
        qs = agent.get_state_beliefs()
        
        # Execute action
        env_obs, reward, done, _, info = env.step(action)
        episode_reward += reward
        
        # Log step data
        if log_data is not None:
            log_data.append({
                'episode': episode_num,
                'step': step,
                'action': action,
                'reward': reward,
                'cumulative_reward': episode_reward,
                'agent_pos': obs_dict['agent_pos'],
                'env_red_pos': env_red_idx,
                'env_blue_pos': env_blue_idx,
                'belief_red_pos': np.argmax(qs['red_button_pos']),
                'belief_red_conf': np.max(qs['red_button_pos']),
                'belief_blue_pos': np.argmax(qs['blue_button_pos']),
                'belief_blue_conf': np.max(qs['blue_button_pos']),
                'on_red_button': obs_dict['on_red_button'],
                'on_blue_button': obs_dict['on_blue_button'],
                'red_pressed': obs_dict['red_button_state'],
                'blue_pressed': obs_dict['blue_button_state'],
                'result': info.get('result', 'pending')
            })
        
        # Update observation
        obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win'
    }


def plot_learning(df, output_dir='plots'):
    """Create learning analysis plots."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get episode-level stats
    episode_stats = df.groupby('episode').agg({
        'reward': 'sum',
        'step': 'max',
        'result': lambda x: (x == 'win').any()
    }).reset_index()
    episode_stats.columns = ['episode', 'total_reward', 'steps', 'success']
    
    # Plot 1: Total reward per episode
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_stats['episode'], episode_stats['total_reward'], alpha=0.3, color='blue')
    # Rolling average
    window = 10
    plt.plot(episode_stats['episode'], 
             episode_stats['total_reward'].rolling(window).mean(), 
             color='blue', linewidth=2, label=f'{window}-episode avg')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve: Total Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines where map changes (every 10 episodes)
    for i in range(10, 100, 10):
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # Plot 2: Success rate over time
    plt.subplot(1, 3, 2)
    success_rate = episode_stats['success'].rolling(window=10).mean() * 100
    plt.plot(episode_stats['episode'], success_rate, color='green', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate (10-episode rolling average)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines where map changes
    for i in range(10, 100, 10):
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # Plot 3: Steps to completion
    plt.subplot(1, 3, 3)
    plt.plot(episode_stats['episode'], episode_stats['steps'], alpha=0.3, color='purple')
    plt.plot(episode_stats['episode'],
             episode_stats['steps'].rolling(window).mean(),
             color='purple', linewidth=2, label=f'{window}-episode avg')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines where map changes
    for i in range(10, 100, 10):
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curve.png', dpi=300)
    print(f"Saved: {output_dir}/learning_curve.png")
    
    # Plot 4: Step-level rewards across all episodes
    plt.figure(figsize=(12, 6))
    
    # Sample episodes to plot (too many to plot all)
    episodes_to_plot = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for ep in episodes_to_plot:
        ep_data = df[df['episode'] == ep]
        if len(ep_data) > 0:
            plt.plot(ep_data['step'], ep_data['reward'], marker='o', alpha=0.7, label=f'Episode {ep}')
    
    plt.xlabel('Step within Episode')
    plt.ylabel('Reward')
    plt.title('Step Rewards Across Selected Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/step_rewards.png', dpi=300)
    print(f"Saved: {output_dir}/step_rewards.png")
    
    # Plot 5: Belief confidence over time
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df['belief_red_conf'], alpha=0.1, color='red')
    plt.plot(df.index, df['belief_red_conf'].rolling(100).mean(), color='red', linewidth=2, label='100-step avg')
    plt.xlabel('Total Steps')
    plt.ylabel('Confidence')
    plt.title('Red Button Position Belief Confidence')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df.index, df['belief_blue_conf'], alpha=0.1, color='blue')
    plt.plot(df.index, df['belief_blue_conf'].rolling(100).mean(), color='blue', linewidth=2, label='100-step avg')
    plt.xlabel('Total Steps')
    plt.ylabel('Confidence')
    plt.title('Blue Button Position Belief Confidence')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/belief_confidence.png', dpi=300)
    print(f"Saved: {output_dir}/belief_confidence.png")
    
    # Plot 6: Performance per map configuration
    episode_stats['config'] = (episode_stats['episode'] - 1) // 50
    config_performance = episode_stats.groupby('config').agg({
        'total_reward': 'mean',
        'success': 'mean',
        'steps': 'mean'
    }).reset_index()
    config_performance['success'] *= 100
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].bar(config_performance['config'], config_performance['total_reward'], color='skyblue')
    axes[0].set_xlabel('Map Configuration')
    axes[0].set_ylabel('Average Total Reward')
    axes[0].set_title('Avg Reward per Map Config (10 episodes each)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(config_performance['config'], config_performance['success'], color='lightgreen')
    axes[1].set_xlabel('Map Configuration')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_title('Success Rate per Map Config')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].bar(config_performance['config'], config_performance['steps'], color='lightcoral')
    axes[2].set_xlabel('Map Configuration')
    axes[2].set_ylabel('Average Steps')
    axes[2].set_title('Avg Steps per Map Config')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/config_performance.png', dpi=300)
    print(f"Saved: {output_dir}/config_performance.png")
    
    plt.close('all')


def main():
    print("="*80)
    print("NON-STATIONARY ENVIRONMENT - 100 EPISODES")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 100
    EPISODES_PER_CONFIG = 10
    MAX_STEPS = 30  # Reduced from 50 for speed
    
    # Setup agent (only once!)
    print("\nSetting up agent...")
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    
    agent = Agent(
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
        gamma=1.0,
        alpha=1.0,
        num_iter=16,
    )
    
    # Initial reset
    agent.reset()
    print("✓ Agent ready\n")
    
    # Data logging
    log_data = []
    results = []
    env = None
    
    print(f"Running {NUM_EPISODES} episodes...")
    print(f"Map changes every {EPISODES_PER_CONFIG} episodes\n")
    
    for episode in range(1, NUM_EPISODES + 1):
        # Create new environment every k episodes
        if (episode - 1) % EPISODES_PER_CONFIG == 0:
            config_num = (episode - 1) // EPISODES_PER_CONFIG + 1
            
            # Generate random button positions
            available_positions = list(range(1, 9))
            np.random.shuffle(available_positions)
            red_pos_idx = available_positions[0]
            blue_pos_idx = available_positions[1]
            
            red_pos = (red_pos_idx // 3, red_pos_idx % 3)
            blue_pos = (blue_pos_idx // 3, blue_pos_idx % 3)
            
            print(f"Config {config_num} (Episodes {episode}-{min(episode+EPISODES_PER_CONFIG-1, NUM_EPISODES)}): "
                  f"Red@{red_pos_idx}, Blue@{blue_pos_idx}")
            
            env = SingleAgentRedBlueButtonEnv(
                width=3,
                height=3,
                red_button_pos=red_pos,
                blue_button_pos=blue_pos,
                agent_start_pos=(0, 0),
                max_steps=30
            )
        
        result = run_episode(env, agent, episode, max_steps=MAX_STEPS, log_data=log_data)
        results.append(result)
        
        # Print progress
        if episode % 10 == 0:
            recent_success = sum(1 for r in results[-10:] if r['success'])
            print(f"  Episode {episode}/{NUM_EPISODES} - Last 10: {recent_success}/10 wins ({100*recent_success/10:.1f}%)")
        elif episode % 1 == 0:  # Every episode
            status = "✓" if result['success'] else "✗"
            print(f"  Episode {episode}/{NUM_EPISODES} {status}", end='\r', flush=True)
    
    # Save data to CSV
    print("\nSaving data to CSV...")
    df = pd.DataFrame(log_data)
    df.to_csv('nonstationary_log.csv', index=False)
    print(f"✓ Saved: nonstationary_log.csv ({len(df)} rows)")
    
    # Create plots
    print("\nGenerating plots...")
    plot_learning(df)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total_successes = sum(1 for r in results if r['success'])
    print(f"Total episodes:  {len(results)}")
    print(f"Total successes: {total_successes} ({100*total_successes/len(results):.1f}%)")
    
    # Per-config performance
    print("\nPerformance per map configuration:")
    for config in range(NUM_EPISODES // EPISODES_PER_CONFIG):
        start = config * EPISODES_PER_CONFIG
        end = start + EPISODES_PER_CONFIG
        config_results = results[start:end]
        config_successes = sum(1 for r in config_results if r['success'])
        avg_reward = np.mean([r['reward'] for r in config_results])
        print(f"  Config {config+1}: {config_successes}/{len(config_results)} wins "
              f"({100*config_successes/len(config_results):.1f}%), "
              f"avg reward: {avg_reward:+.3f}")
    
    print("\n" + "="*80)
    print("Done! Check 'plots/' directory for visualizations")
    print("="*80)


if __name__ == "__main__":
    main()

