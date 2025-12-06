"""
Run episodes with non-stationary button positions.

Every k episodes, button positions change in the environment.
Agent keeps its beliefs about button positions from previous episode,
but resets position to 0 and button states to not_pressed.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)
from agents.ActiveInference.agent import Agent


def print_step_info(step, obs_dict, qs, action, reward, info, agent, env):
    """Print detailed step information with map, full beliefs, and policies."""
    print(f"\n{'='*80}")
    print(f"STEP {step}")
    print(f"{'='*80}")
    
    # Grid map
    print("\n🗺️  GRID MAP:")
    grid = env.render(mode="array")  # Get grid without printing
    for row in grid:
        print(f"     {' '.join(row)}")
    print(f"     (A=agent, r/R=red button, b/B=blue button, capitals=pressed)")
    
    # Observations
    print("\n📊 OBSERVATIONS:")
    print(f"  Position:             {obs_dict['agent_pos']}")
    print(f"  On red button:        {['FALSE', 'TRUE'][obs_dict['on_red_button']]}")
    print(f"  On blue button:       {['FALSE', 'TRUE'][obs_dict['on_blue_button']]}")
    print(f"  Red button state:     {['not_pressed', 'pressed'][obs_dict['red_button_state']]}")
    print(f"  Blue button state:    {['not_pressed', 'pressed'][obs_dict['blue_button_state']]}")
    
    # Full belief distributions
    print("\n🧠 AGENT BELIEFS (Full Distributions):")
    
    # Agent position belief (grid format)
    print("  Agent position belief:")
    agent_belief = qs['agent_pos']
    for row in range(3):
        row_str = "    "
        for col in range(3):
            idx = row * 3 + col
            row_str += f"{agent_belief[idx]:.2f} "
        print(row_str)
    
    # Red button position belief (grid format)
    print("  Red button position belief:")
    red_belief = qs['red_button_pos']
    for row in range(3):
        row_str = "    "
        for col in range(3):
            idx = row * 3 + col
            row_str += f"{red_belief[idx]:.2f} "
        print(row_str)
    
    # Blue button position belief (grid format)
    print("  Blue button position belief:")
    blue_belief = qs['blue_button_pos']
    for row in range(3):
        row_str = "    "
        for col in range(3):
            idx = row * 3 + col
            row_str += f"{blue_belief[idx]:.2f} "
        print(row_str)
    
    # Button states
    print(f"  Red button state:     {qs['red_button_state']} (not_pressed, pressed)")
    print(f"  Blue button state:    {qs['blue_button_state']} (not_pressed, pressed)")
    
    # All policies and their probabilities
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'OPEN', 5: 'NOOP'}
    print("\n📋 POLICY BELIEFS (All Policies):")
    
    q_pi = agent.q_pi
    sorted_indices = np.argsort(q_pi)[::-1]  # All policies in descending order
    
    # Print in columns for compactness
    for i in range(0, len(sorted_indices), 3):
        line = "  "
        for j in range(3):
            if i + j < len(sorted_indices):
                idx = sorted_indices[i + j]
                policy = agent.policies[idx]
                prob = q_pi[idx]
                policy_str = '→'.join([action_names[a][:2] for a in policy])  # Abbreviate
                line += f"[{idx:2d}]{policy_str:8s}:{prob:.4f}  "
        print(line)
    
    # Action
    print(f"\n🎯 ACTION SELECTED:     {action} ({action_names[action]})")
    
    # Outcome
    print(f"\n📈 OUTCOME:")
    print(f"  Reward:               {reward:+.3f}")
    print(f"  Result:               {info.get('result', 'pending')}")


def grid_to_string(grid):
    """Convert grid array to string representation."""
    return '|'.join([''.join(row) for row in grid])


def run_episode(env, agent, episode_num, max_steps=50, verbose=True, csv_writer=None):
    """Run one episode."""
    
    # Reset environment
    env_obs, _ = env.reset()
    
    # Manually update agent's beliefs (don't call reset!)
    # Only reset: agent position and button states
    agent.qs['agent_pos'] = np.zeros(9)
    agent.qs['agent_pos'][0] = 1.0  # Certain at position 0
    
    agent.qs['red_button_state'] = np.array([1.0, 0.0])  # Certain not_pressed
    agent.qs['blue_button_state'] = np.array([1.0, 0.0])  # Certain not_pressed
    
    # Keep button position beliefs from previous episode!
    # Keep policy posterior q_pi!
    # Reset action tracking but keep action = NOOP to avoid using D_fn as prior
    agent.action = 5  # Set to NOOP instead of None to preserve button position beliefs
    agent.prev_actions = []
    agent.curr_timestep = 0
    
    if verbose:
        # Convert env button positions to indice
        env_red_idx = env.red_button[1] * 3 + env.red_button[0]
        env_blue_idx = env.blue_button[1] * 3 + env.blue_button[0]
        
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}")
        print(f"{'='*80}")
        print(f"Environment: Red at {env.red_button} (idx={env_red_idx}), "
              f"Blue at {env.blue_button} (idx={env_blue_idx})")
        print(f"Agent beliefs (from prev episode): Red at idx={np.argmax(agent.qs['red_button_pos'])} "
              f"(conf: {np.max(agent.qs['red_button_pos']):.3f}), "
              f"Blue at idx={np.argmax(agent.qs['blue_button_pos'])} "
              f"(conf: {np.max(agent.qs['blue_button_pos']):.3f})")
    
    # Convert initial observation (don't call infer_states - first step() will do it)
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    
    episode_reward = 0.0
    outcome = 'timeout'
    
    for step in range(1, max_steps + 1):
        # Get map before action
        grid = env.render(mode="array")
        map_str = grid_to_string(grid)
        
        # Agent perceives, infers, plans, and acts
        action = agent.step(obs_dict)
        
        # Get agent's current beliefs
        qs = agent.get_state_beliefs()
        
        # Print step info with full details
        if verbose:
            print_step_info(step, obs_dict, qs, action, 0.0, {'result': 'pending'}, agent, env)
        
        # Execute action in environment
        env_obs, reward, done, _, info = env.step(action)
        episode_reward += reward
        
        # Log to CSV
        if csv_writer is not None:
            action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
            csv_writer.writerow({
                'episode': episode_num,
                'step': step,
                'action': action,
                'action_name': action_names[action],
                'map': map_str,
                'reward': reward
            })
        
        if verbose:
            print(f"\n  → Environment response:")
            print(f"     Reward:  {reward:+.3f}")
            print(f"     Result:  {info.get('result', 'neutral')}")
        
        # Update observation
        obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    if verbose:
        status = "✅ WIN" if outcome == 'win' else "❌ FAIL"
        print(f"Result: {status} - {outcome} (steps: {step}, reward: {episode_reward:+.3f})")
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win',
        'final_beliefs': {
            'red_pos': (np.argmax(agent.qs['red_button_pos']), 
                       np.max(agent.qs['red_button_pos'])),
            'blue_pos': (np.argmax(agent.qs['blue_button_pos']), 
                        np.max(agent.qs['blue_button_pos']))
        }
    }


def main():
    print("="*80)
    print("NON-STATIONARY ENVIRONMENT - BUTTON POSITIONS CHANGE")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 10
    EPISODES_PER_CONFIG = 2  # Change button positions every 2 episodes
    MAX_STEPS = 50
    
    # Setup CSV logging with timestamp
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"active_inference_log_ep{NUM_EPISODES}_step{MAX_STEPS}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    print(f"Logging to: {csv_path}")
    
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
        gamma=2.0,
        alpha=1.0,
        num_iter=16,
    )
    
    # Initial reset (only once at the very beginning)
    agent.reset()
    print("✓ Agent ready\n")
    
    # Run episodes
    results = []
    env = None
    
    try:
        for episode in range(1, NUM_EPISODES + 1):
            # Create new environment every k episodes
            if (episode - 1) % EPISODES_PER_CONFIG == 0:
                # Generate random button positions (avoid agent start position 0)
                available_positions = list(range(1, 9))
                np.random.shuffle(available_positions)
                red_pos_idx = available_positions[0]
                blue_pos_idx = available_positions[1]
                
                # Convert to (row, col)
                red_pos = (red_pos_idx // 3, red_pos_idx % 3)
                blue_pos = (blue_pos_idx // 3, blue_pos_idx % 3)
                
                print(f"\n{'='*80}")
                print(f"NEW ENVIRONMENT CONFIGURATION (Episodes {episode}-{min(episode+EPISODES_PER_CONFIG-1, NUM_EPISODES)})")
                print(f"{'='*80}")
                print(f"Red button moving to position {red_pos_idx} {red_pos}")
                print(f"Blue button moving to position {blue_pos_idx} {blue_pos}")
                
                env = SingleAgentRedBlueButtonEnv(
                    width=3,
                    height=3,
                    red_button_pos=red_pos,
                    blue_button_pos=blue_pos,
                    agent_start_pos=(0, 0),
                    max_steps=100
                )
            
            result = run_episode(env, agent, episode, max_steps=MAX_STEPS, verbose=True, csv_writer=csv_writer)
            results.append(result)
    finally:
        csv_file.close()
        print(f"\nLog saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successes = sum(1 for r in results if r['success'])
    success_rate = 100 * successes / len(results)
    
    print(f"\nTotal episodes:  {len(results)}")
    print(f"Successes:       {successes} ({success_rate:.1f}%)")
    print(f"Failures:        {len(results) - successes} ({100-success_rate:.1f}%)")
    
    # Success rate per configuration
    print(f"\nSuccess rate per configuration (every {EPISODES_PER_CONFIG} episodes):")
    for config_idx in range(NUM_EPISODES // EPISODES_PER_CONFIG):
        start = config_idx * EPISODES_PER_CONFIG
        end = start + EPISODES_PER_CONFIG
        config_results = results[start:end]
        config_successes = sum(1 for r in config_results if r['success'])
        config_rate = 100 * config_successes / len(config_results)
        print(f"  Config {config_idx+1} (Episodes {start+1}-{end}): "
              f"{config_successes}/{len(config_results)} ({config_rate:.1f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

