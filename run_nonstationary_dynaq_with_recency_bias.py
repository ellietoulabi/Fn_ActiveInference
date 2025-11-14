"""
Run Dyna-Q with RECENCY BIAS for non-stationary button positions.

Every k episodes, button positions change in the environment.
Agent keeps its Q-table AND world model from previous episodes but explores new environment.
Dyna-Q combines direct RL with model-based planning for faster learning.

RECENCY BIAS: Planning prioritizes recently experienced state-action pairs,
helping the agent adapt faster when environment dynamics change.
"""

import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import env_utils
from agents.QLearning.dynaq_agent_with_recency_bias import DynaQAgent


def print_step_info(step, obs_dict, state, action, reward, info, agent, env, q_values=None):
    """Print detailed step information with map and Q-values."""
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
    
    # State
    print(f"\n🔢 STATE:")
    print(f"  {state}")
    state_labels = agent.state_to_labels(state)
    if state_labels:
        print(f"  Agent position:       {state_labels['agent_pos']}")
        print(f"  On red button:        {state_labels['on_red_button']}")
        print(f"  On blue button:       {state_labels['on_blue_button']}")
        print(f"  Red button state:     {state_labels['red_button_state']}")
        print(f"  Blue button state:    {state_labels['blue_button_state']}")
    
    # Q-values for current state
    if q_values is not None:
        action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
        print(f"\n📊 Q-VALUES FOR CURRENT STATE:")
        sorted_actions = np.argsort(q_values)[::-1]  # Actions in descending Q-value order
        for a in sorted_actions:
            marker = "★" if a == action else " "
            print(f"  {marker} {action_names[a]:6s}: {q_values[a]:+.4f}")
    
    # Action
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    print(f"\n🎯 ACTION SELECTED:     {action} ({action_names[action]})")
    print(f"   Epsilon (exploration): {agent.epsilon:.4f}")
    
    # Outcome
    print(f"\n📈 OUTCOME:")
    print(f"  Reward:               {reward:+.3f}")
    print(f"  Result:               {info.get('result', 'pending')}")


def grid_to_string(grid):
    """Convert grid array to string representation."""
    return '|'.join([''.join(row) for row in grid])


def run_episode(env, agent, episode_num, max_steps=50, verbose=True, csv_writer=None):
    """Run one Dyna-Q episode with recency bias."""
    
    # Reset environment
    env_obs, _ = env.reset()
    
    # Convert to model observation format (SAME as Active Inference)
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    state = agent.get_state(obs_dict)
    
    if verbose:
        # Convert env button positions to indices
        env_red_idx = env.red_button[1] * 3 + env.red_button[0]
        env_blue_idx = env.blue_button[1] * 3 + env.blue_button[0]
        
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}")
        print(f"{'='*80}")
        print(f"Environment: Red at {env.red_button} (idx={env_red_idx}), "
              f"Blue at {env.blue_button} (idx={env_blue_idx})")
        print(f"Q-table (from all previous episodes): {len(agent.q_table)} states learned")
        print(f"World model: {len(agent.model)} states, {len(agent.visited_state_actions)} (s,a) pairs")
        
        # Show recency bias stats
        stats = agent.get_stats()
        print(f"Recency decay: {agent.recency_decay:.3f} (lower = faster forgetting)")
        print(f"Global step: {stats['global_step']} (total steps across all episodes)")
        if stats['avg_experience_age'] > 0:
            print(f"Average experience age: {stats['avg_experience_age']:.1f} steps")
        
        print(f"Epsilon: {agent.epsilon:.4f} (exploration rate)")
        print(f"Planning steps per real step: {agent.planning_steps} (with recency bias)")
    
    episode_reward = 0.0
    outcome = 'timeout'
    step = 0
    
    for step in range(1, max_steps + 1):
        # Get map before action
        grid = env.render(mode="array")
        map_str = grid_to_string(grid)
        
        # Get Q-values for current state
        if state in agent.q_table:
            q_values = agent.q_table[state].copy()
        else:
            q_values = None
        
        # Choose action
        action = agent.choose_action(state)
        
        # Print step info with full details
        if verbose:
            print_step_info(step, obs_dict, state, action, 0.0, 
                          {'result': 'pending'}, agent, env, q_values)
        
        # Execute action in environment
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
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
        
        # Convert to model observation format
        next_obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        next_state = agent.get_state(next_obs_dict) if not done else None
        
        # (1) Direct RL: Update Q-table from real experience
        agent.update_q_table(state, action, reward, next_state)
        
        # (2) Model Learning: Store transition in world model (with timestamp)
        agent.update_model(state, action, next_state, reward, terminated)
        
        # (3) Planning: Learn from simulated experience (WITH RECENCY BIAS)
        agent.planning()
        
        if verbose:
            print(f"\n  → Environment response:")
            print(f"     Reward:  {reward:+.3f}")
            print(f"     Result:  {info.get('result', 'neutral')}")
            print(f"     Q-table updated from REAL experience")
            print(f"     Model updated (now {len(agent.model)} states, step={agent.global_step})")
            print(f"     Performed {agent.planning_steps} PLANNING updates (recency-weighted)")
        
        # Update for next iteration
        state = next_state
        obs_dict = next_obs_dict
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    # Decay exploration after each episode
    agent.decay_exploration()
    
    if verbose:
        status = "✅ WIN" if outcome == 'win' else "❌ FAIL"
        print(f"\nResult: {status} - {outcome} (steps: {step}, reward: {episode_reward:+.3f})")
        print(f"Epsilon decayed to: {agent.epsilon:.4f}")
        print(f"Q-table now has: {len(agent.q_table)} states (persists to next episode)")
        print(f"Model now has: {len(agent.model)} states, {len(agent.visited_state_actions)} (s,a) pairs")
    
    # Get final stats
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
        'global_step': stats['global_step'],
        'avg_experience_age': stats['avg_experience_age']
    }


def main():
    print("="*80)
    print("DYNA-Q WITH RECENCY BIAS: NON-STATIONARY ENVIRONMENT")
    print("Model-Based Planning + Direct RL + Recency-Weighted Sampling")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 2000
    EPISODES_PER_CONFIG = 200  # Change button positions every 200 episodes
    MAX_STEPS = 100
    PLANNING_STEPS = 10  # Number of planning steps per real step
    RECENCY_DECAY = 0.95  # Decay rate for recency bias (lower = faster forgetting)
    
    # Setup CSV logging with timestamp
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"dynaq_recency_log_ep{NUM_EPISODES}_step{MAX_STEPS}_plan{PLANNING_STEPS}_decay{RECENCY_DECAY}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    print(f"Logging to: {csv_path}")
    
    # Setup Dyna-Q agent WITH RECENCY BIAS (only once!)
    # Q-table AND world model persist across ALL episodes
    print("\nSetting up Dyna-Q agent with RECENCY BIAS...")
    agent = DynaQAgent(
        action_space_size=6,
        planning_steps=PLANNING_STEPS,  # Key Dyna-Q parameter
        recency_decay=RECENCY_DECAY,  # NEW: Recency bias parameter
        q_table_path="q_table_dynaq_recency_nonstationary.json",
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,  # Start with full exploration
        epsilon_decay=0.95,  # Faster decay to exploit learned Q-values sooner
        min_epsilon=0.05,  # Lower minimum for more exploitation
        load_existing=False  # Start fresh
    )
    print("✓ Agent ready")
    print(f"   Q-table will persist across all {NUM_EPISODES} episodes")
    print(f"   World model will persist and grow with experience")
    print(f"   {PLANNING_STEPS} planning updates per real environment step")
    print(f"   RECENCY DECAY: {RECENCY_DECAY} (prioritizes recent experiences)")
    print(f"   (Dyna-Q + Recency Bias = Direct RL + Model Learning + Recency-Weighted Planning)\n")
    
    # Run episodes
    results = []
    env = None
    
    try:
        for episode in range(1, NUM_EPISODES + 1):
            # NOTE: Agent's Q-table AND model are NOT reset between episodes
            # They accumulate knowledge across all episodes and environment changes
            
            # Create new environment every k episodes
            if (episode - 1) % EPISODES_PER_CONFIG == 0:
                # Generate random button positions (avoid agent start position 0)
                available_positions = list(range(1, 9))
                np.random.shuffle(available_positions)
                red_pos_idx = available_positions[0]
                blue_pos_idx = available_positions[1]
                
                # Convert to (y, x) for environment (row, col)
                red_pos = (red_pos_idx // 3, red_pos_idx % 3)
                blue_pos = (blue_pos_idx // 3, blue_pos_idx % 3)
                
                print(f"\n{'='*80}")
                print(f"NEW ENVIRONMENT CONFIGURATION (Episodes {episode}-{min(episode+EPISODES_PER_CONFIG-1, NUM_EPISODES)})")
                print(f"{'='*80}")
                print(f"Red button moving to position {red_pos_idx} (row={red_pos[0]}, col={red_pos[1]})")
                print(f"Blue button moving to position {blue_pos_idx} (row={blue_pos[0]}, col={blue_pos[1]})")
                print(f"Recency bias will prioritize learning from new button positions")
                
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
            
            # Print episode summary
            status = "✅ WIN" if result['success'] else "❌ FAIL"
            print(f"Episode {episode}: {status} - {result['steps']} steps, "
                  f"reward: {result['reward']:+.2f}, "
                  f"Q-table: {result['q_table_size']} states, "
                  f"Model: {result['model_size']} states, "
                  f"Avg exp age: {result['avg_experience_age']:.1f}, "
                  f"ε: {result['epsilon']:.3f}")
    finally:
        csv_file.close()
        print(f"\nLog saved to: {csv_path}")
    
    # Save final Q-table
    print(f"Saving Q-table to '{agent.q_table_path}'...")
    agent.save_q_table()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successes = sum(1 for r in results if r['success'])
    success_rate = 100 * successes / len(results)
    
    print(f"\nTotal episodes:  {len(results)}")
    print(f"Successes:       {successes} ({success_rate:.1f}%)")
    print(f"Failures:        {len(results) - successes} ({100-success_rate:.1f}%)")
    print(f"Final Q-table size: {results[-1]['q_table_size']} states")
    print(f"Final model size: {results[-1]['model_size']} states")
    print(f"Final (s,a) pairs: {results[-1]['visited_sa_pairs']}")
    print(f"Final epsilon: {results[-1]['epsilon']:.4f}")
    print(f"Planning steps per real step: {PLANNING_STEPS}")
    print(f"Recency decay: {RECENCY_DECAY}")
    print(f"Total steps (global): {results[-1]['global_step']}")
    print(f"Avg experience age: {results[-1]['avg_experience_age']:.1f} steps")
    
    # Success rate per configuration
    print(f"\nSuccess rate per configuration (every {EPISODES_PER_CONFIG} episodes):")
    for config_idx in range(NUM_EPISODES // EPISODES_PER_CONFIG):
        start = config_idx * EPISODES_PER_CONFIG
        end = start + EPISODES_PER_CONFIG
        config_results = results[start:end]
        config_successes = sum(1 for r in config_results if r['success'])
        config_rate = 100 * config_successes / len(config_results)
        avg_reward = np.mean([r['reward'] for r in config_results])
        avg_model_size = np.mean([r['model_size'] for r in config_results])
        avg_exp_age = np.mean([r['avg_experience_age'] for r in config_results])
        print(f"  Config {config_idx+1} (Episodes {start+1}-{end}): "
              f"{config_successes}/{len(config_results)} ({config_rate:.1f}%), "
              f"avg reward: {avg_reward:+.2f}, "
              f"avg model size: {avg_model_size:.1f}, "
              f"avg exp age: {avg_exp_age:.1f}")
    
    # Episode-by-episode results
    print(f"\nEpisode-by-episode results:")
    print(f"  {'Ep':<4} {'Outcome':<8} {'Steps':<6} {'Reward':<8} {'Q-size':<8} {'Model':<8} {'ExpAge':<8} {'Epsilon':<8}")
    print(f"  {'-'*70}")
    for i, r in enumerate(results, 1):
        outcome_short = "WIN" if r['success'] else "FAIL"
        print(f"  {i:<4} {outcome_short:<8} {r['steps']:<6} {r['reward']:+7.2f}  "
              f"{r['q_table_size']:<8} {r['model_size']:<8} {r['avg_experience_age']:<8.1f} {r['epsilon']:<8.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()


