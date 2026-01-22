"""
Run OPSRL episodes with non-stationary button positions.

Every k episodes, button positions change in the environment.
OPSRL uses posterior sampling for efficient exploration.
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
from agents.OPSRL import OPSRLAgent


def print_step_info(step, obs, state, action, reward, info, agent, env):
    """Print detailed step information with map."""
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
    position = obs["position"]
    if isinstance(position, np.ndarray):
        x, y = int(position[0]), int(position[1])
    else:
        x, y = position[0], position[1]
    print(f"  Position:             ({x}, {y})")
    print(f"  On red button:        {['FALSE', 'TRUE'][obs.get('on_red_button', 0)]}")
    print(f"  On blue button:       {['FALSE', 'TRUE'][obs.get('on_blue_button', 0)]}")
    print(f"  Red button pressed:   {['FALSE', 'TRUE'][obs.get('red_button_pressed', 0)]}")
    print(f"  Blue button pressed: {['FALSE', 'TRUE'][obs.get('blue_button_pressed', 0)]}")
    
    # State
    print(f"\n🔢 STATE:")
    print(f"  Discrete state index: {state}")
    
    # Action
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    print(f"\n🎯 ACTION SELECTED:     {action} ({action_names[action]})")
    
    # Outcome
    print(f"\n📈 OUTCOME:")
    print(f"  Reward:               {reward:+.3f}")
    print(f"  Result:               {info.get('result', 'pending')}")


def grid_to_string(grid):
    """Convert grid array to string representation."""
    return '|'.join([''.join(row) for row in grid])


def run_evaluation_episode(env, agent, episode_num, max_steps=50, verbose=True, csv_writer=None):
    """Run one evaluation episode using the trained policy."""
    
    # Reset environment
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    
    if verbose:
        # Convert env button positions to indices
        env_red_idx = env.red_button[1] * 3 + env.red_button[0]
        env_blue_idx = env.blue_button[1] * 3 + env.blue_button[0]
        
        print(f"\n{'='*80}")
        print(f"EVALUATION EPISODE {episode_num}")
        print(f"{'='*80}")
        print(f"Environment: Red at {env.red_button} (idx={env_red_idx}), "
              f"Blue at {env.blue_button} (idx={env_blue_idx})")
        print(f"Using trained OPSRL policy (episode {agent.episode})")
    
    episode_reward = 0.0
    outcome = 'timeout'
    step = 0
    
    for step in range(1, max_steps + 1):
        # Get map before action
        grid = env.render(mode="array")
        map_str = grid_to_string(grid)
        
        # Get state
        state = agent._obs_to_state(obs)
        
        # Choose action using trained policy
        action = agent.policy(obs)
        
        # Print step info with full details
        if verbose:
            print_step_info(step, obs, state, action, 0.0, 
                          {'result': 'pending'}, agent, env)
        
        # Execute action in environment
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result
            terminated = done
        
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
        
        # Update for next iteration
        obs = next_obs
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    if verbose:
        status = "✅ WIN" if outcome == 'win' else "❌ FAIL"
        print(f"\nResult: {status} - {outcome} (steps: {step}, reward: {episode_reward:+.3f})")
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win'
    }


def main():
    print("="*80)
    print("OPSRL: NON-STATIONARY ENVIRONMENT - BUTTON POSITIONS CHANGE")
    print("Optimistic Posterior Sampling for Reinforcement Learning")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 200  # Training episodes
    EVAL_EPISODES = 10  # Evaluation episodes after training
    EPISODES_PER_CONFIG = 20  # Change button positions every N episodes
    MAX_STEPS = 100
    HORIZON = 100  # OPSRL horizon
    
    # OPSRL parameters
    GAMMA = 0.95
    THOMPSON_SAMPLES = 1
    BERNOULLIZED_REWARD = True
    SCALE_PRIOR_REWARD = 1.0
    PRIOR_TRANSITION = 'uniform'
    
    # Setup CSV logging with timestamp
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"opsrl_log_ep{NUM_EPISODES}_step{MAX_STEPS}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    print(f"Logging to: {csv_path}")
    
    # Setup OPSRL agent
    print("\nSetting up OPSRL agent...")
    env = SingleAgentRedBlueButtonEnv(
        width=3,
        height=3,
        red_button_pos=(0, 2),
        blue_button_pos=(2, 0),
        agent_start_pos=(0, 0),
        max_steps=MAX_STEPS
    )
    
    agent = OPSRLAgent(
        env=env,
        gamma=GAMMA,
        horizon=HORIZON,
        bernoullized_reward=BERNOULLIZED_REWARD,
        scale_prior_reward=SCALE_PRIOR_REWARD,
        thompson_samples=THOMPSON_SAMPLES,
        prior_transition=PRIOR_TRANSITION,
        reward_free=False,
        stage_dependent=False,
        seed=42
    )
    print("✓ Agent ready")
    print(f"   Horizon: {HORIZON}")
    print(f"   Gamma: {GAMMA}")
    print(f"   Thompson samples: {THOMPSON_SAMPLES}")
    print(f"   Bernoullized reward: {BERNOULLIZED_REWARD}")
    print(f"   Prior transition: {PRIOR_TRANSITION}\n")
    
    # Training phase
    print("="*80)
    print("TRAINING PHASE")
    print("="*80)
    
    training_results = []
    current_env = env
    
    try:
        for episode in range(1, NUM_EPISODES + 1):
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
                
                current_env = SingleAgentRedBlueButtonEnv(
                    width=3,
                    height=3,
                    red_button_pos=red_pos,
                    blue_button_pos=blue_pos,
                    agent_start_pos=(0, 0),
                    max_steps=MAX_STEPS
                )
            
            # Update agent's environment reference before each episode
            agent.env = current_env
            
            # Train for one episode (OPSRL handles episode internally)
            episode_reward = agent._run_episode()
            
            training_results.append({
                'episode': episode,
                'reward': episode_reward,
                'success': episode_reward > 0  # Win gives reward 1
            })
            
            # Print episode summary
            if episode % 10 == 0 or episode == 1:
                status = "✅" if episode_reward > 0 else "❌"
                print(f"Episode {episode}/{NUM_EPISODES}: {status} reward: {episode_reward:+.2f}")
        
        # Compute final policy after training
        print("\nComputing final recommended policy...")
        agent.fit(budget=0)  # This computes the policy from current posteriors
        
        # Evaluation phase
        print("\n" + "="*80)
        print("EVALUATION PHASE")
        print("="*80)
        
        eval_results = []
        for eval_ep in range(1, EVAL_EPISODES + 1):
            result = run_evaluation_episode(
                current_env, agent, eval_ep, 
                max_steps=MAX_STEPS, 
                verbose=(eval_ep <= 3),  # Only verbose for first 3 episodes
                csv_writer=csv_writer
            )
            eval_results.append(result)
            
            status = "✅ WIN" if result['success'] else "❌ FAIL"
            print(f"Eval Episode {eval_ep}: {status} - {result['steps']} steps, "
                  f"reward: {result['reward']:+.2f}")
    
    finally:
        csv_file.close()
        print(f"\nLog saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Training summary
    training_successes = sum(1 for r in training_results if r['success'])
    training_success_rate = 100 * training_successes / len(training_results)
    avg_training_reward = np.mean([r['reward'] for r in training_results])
    
    print(f"\nTRAINING PHASE:")
    print(f"  Total episodes:  {len(training_results)}")
    print(f"  Successes:       {training_successes} ({training_success_rate:.1f}%)")
    print(f"  Failures:        {len(training_results) - training_successes} ({100-training_success_rate:.1f}%)")
    print(f"  Avg reward:      {avg_training_reward:+.3f}")
    
    # Evaluation summary
    if eval_results:
        eval_successes = sum(1 for r in eval_results if r['success'])
        eval_success_rate = 100 * eval_successes / len(eval_results)
        avg_eval_reward = np.mean([r['reward'] for r in eval_results])
        
        print(f"\nEVALUATION PHASE:")
        print(f"  Total episodes:  {len(eval_results)}")
        print(f"  Successes:       {eval_successes} ({eval_success_rate:.1f}%)")
        print(f"  Failures:        {len(eval_results) - eval_successes} ({100-eval_success_rate:.1f}%)")
        print(f"  Avg reward:      {avg_eval_reward:+.3f}")
    
    # Success rate per configuration
    print(f"\nTraining success rate per configuration (every {EPISODES_PER_CONFIG} episodes):")
    for config_idx in range(NUM_EPISODES // EPISODES_PER_CONFIG):
        start = config_idx * EPISODES_PER_CONFIG
        end = start + EPISODES_PER_CONFIG
        config_results = training_results[start:end]
        config_successes = sum(1 for r in config_results if r['success'])
        config_rate = 100 * config_successes / len(config_results)
        avg_reward = np.mean([r['reward'] for r in config_results])
        print(f"  Config {config_idx+1} (Episodes {start+1}-{end}): "
              f"{config_successes}/{len(config_results)} ({config_rate:.1f}%), "
              f"avg reward: {avg_reward:+.2f}")
    
    # Recent training performance
    print(f"\nRecent training performance (last 20 episodes):")
    recent_results = training_results[-20:]
    recent_successes = sum(1 for r in recent_results if r['success'])
    recent_rate = 100 * recent_successes / len(recent_results)
    recent_avg_reward = np.mean([r['reward'] for r in recent_results])
    print(f"  Success rate: {recent_rate:.1f}%")
    print(f"  Avg reward: {recent_avg_reward:+.3f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

