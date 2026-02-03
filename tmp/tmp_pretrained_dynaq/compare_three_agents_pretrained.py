"""
Compare Active Inference, Q-Learning, and Dyna-Q (with pre-trained model).

This version uses a pre-trained world model for Dyna-Q, giving it the same
world knowledge that Active Inference has in its generative model.

Usage:
    # First, create a pre-trained model:
    python pretrain_world_model.py --episodes 50 --planning_steps 2 --output pretrained_model.json
    
    # Then run this comparison:
    python compare_three_agents_pretrained.py --pretrained_model pretrained_model.json
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from pathlib import Path
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)
from agents.ActiveInference.agent import Agent
from agents.QLearning import QLearningAgent, TabularDynaQAgent  # Use TabularDynaQ


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
    parser = argparse.ArgumentParser(description='Compare three agents with pre-trained model')
    parser.add_argument('--pretrained_model', default='pretrained_world_model.json',
                       help='Path to pre-trained world model (default: pretrained_world_model.json)')
    parser.add_argument('--episodes', type=int, default=12,
                       help='Total episodes (default: 80)')
    parser.add_argument('--episodes_per_config', type=int, default=4,
                       help='Episodes per config (default: 20)')
    parser.add_argument('--planning_steps', type=int, default=2,
                       help='Dyna-Q planning steps (default: 2)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPARING: AIF vs QL vs Dyna-Q (WITH PRE-TRAINED MODEL)")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = args.episodes
    EPISODES_PER_CONFIG = args.episodes_per_config
    MAX_STEPS = 50
    PLANNING_STEPS = args.planning_steps
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)
    
    # Check if pre-trained model exists
    if not Path(args.pretrained_model).exists():
        print(f"\n❌ ERROR: Pre-trained model '{args.pretrained_model}' not found!")
        print("\nPlease run this first:")
        print(f"  python pretrain_world_model.py --episodes 50 --planning_steps {PLANNING_STEPS} --output {args.pretrained_model}")
        return
    
    print(f"\n✓ Using pre-trained model: {args.pretrained_model}")
    
    # Setup CSV logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"comparison_pretrained_ep{NUM_EPISODES}_step{MAX_STEPS}_plan{PLANNING_STEPS}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['agent', 'episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    print(f"Logging to: {csv_path}")
    
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
    print("✓ Active Inference agent ready (built-in generative model)")
    
    # Setup Q-Learning agent
    print("\nSetting up Q-Learning agent...")
    ql_agent = QLearningAgent(
        action_space_size=6,
        q_table_path=None,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        load_existing=False
    )
    print("✓ Q-Learning agent ready (no model)")
    
    # Setup Dyna-Q agent WITH PRE-TRAINED MODEL
    print("\nSetting up Dyna-Q agent...")
    dynaq_agent = TabularDynaQAgent(  # Use TabularDynaQ instead of DynaQ
        action_space_size=6,
        planning_steps=PLANNING_STEPS,
        q_table_path=None,
        model_path=args.pretrained_model,  # Load pre-trained model
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        load_existing_q_table=False,
        load_existing_model=True  # LOAD THE PRE-TRAINED MODEL
    )
    print(f"✓ Dyna-Q agent ready (planning_steps={PLANNING_STEPS}, PRE-TRAINED MODEL LOADED)")
    
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
                print(f"     {'Dyna-Q (pre)':<15} {dynaq_wins}/{episode} ({dynaq_rate:>5.1f}%)   {dynaq_avg_r:>+7.2f}")
    finally:
        csv_file.close()
        print(f"\n✓ Log saved to: {csv_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    aif_successes = sum(1 for r in aif_results if r['success'])
    ql_successes = sum(1 for r in ql_results if r['success'])
    dynaq_successes = sum(1 for r in dynaq_results if r['success'])
    
    aif_success_rate = 100 * aif_successes / len(aif_results)
    ql_success_rate = 100 * ql_successes / len(ql_results)
    dynaq_success_rate = 100 * dynaq_successes / len(dynaq_results)
    
    print(f"\n{'Agent':<20} {'Success Rate':<20} {'Wins':<15}")
    print("-" * 55)
    print(f"{'AIF (generative)':<20} {aif_success_rate:>6.1f}%             {aif_successes}/{len(aif_results)}")
    print(f"{'Q-Learning (none)':<20} {ql_success_rate:>6.1f}%             {ql_successes}/{len(ql_results)}")
    print(f"{'Dyna-Q (pre-trained)':<20} {dynaq_success_rate:>6.1f}%             {dynaq_successes}/{len(dynaq_results)}")
    
    print("\n" + "="*80)
    print("✓ Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()

