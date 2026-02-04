"""
Compare 9 agents on the same environment configurations:
- Active Inference (model-based, variational inference)
- Q-Learning (baseline, no planning)
- Vanilla Dyna-Q (uniform random sampling)
- Dyna-Q with recency_decay = 0.99 (slow forgetting)
- Dyna-Q with recency_decay = 0.95 (medium forgetting)
- Dyna-Q with recency_decay = 0.90 (fast forgetting)
- Dyna-Q with recency_decay = 0.85 (very fast forgetting)
- Trajectory Sampling Dyna-Q (multi-step rollouts for fast Q-value propagation)
- OPSRL (Optimistic Posterior Sampling for Reinforcement Learning)

All agents experience the EXACT SAME button position changes at the same episodes.
This ensures a fair comparison of their adaptation capabilities.

NOTE: This script uses relative paths based on script location. It can be run from any
directory and will work on any server as long as the project structure is maintained.
All paths are computed relative to the script's location.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
# Uses __file__ to find project root relative to script location (portable across servers)
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import random
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
from agents.OPSRL import OPSRLAgent


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


def run_opsrl_episode(env, agent, agent_name, episode_num, max_steps=50, csv_writer=None):
    """Run one OPSRL episode.
    
    OPSRL accumulates experience across all episodes (including environment changes),
    just like AIF and QL agents. This allows it to learn from all experience.
    """
    
    # Update agent's environment reference
    agent.env = env
    
    # STEP 1: Sample from posterior and plan BEFORE episode (this is critical!)
    # This is what _run_episode does at the start
    B = agent.thompson_samples
    
    if agent.stage_dependent:
        M_sab_zero = np.repeat(agent.M_sa[..., 0, np.newaxis], B, -1)
        M_sab_one = np.repeat(agent.M_sa[..., 1, np.newaxis], B, -1)
        N_sasb = np.repeat(agent.N_sas[..., np.newaxis], B, axis=-1)
    else:
        M_sab_zero = np.repeat(agent.M_sa[..., 0, np.newaxis], B, -1)
        M_sab_one = np.repeat(agent.M_sa[..., 1, np.newaxis], B, -1)
        N_sasb = np.repeat(agent.N_sas[..., np.newaxis], B, axis=-1)
    
    # Sample rewards from Beta distribution
    R_samples = agent.rng.beta(M_sab_zero, M_sab_one)
    
    # Sample transitions from Dirichlet (via Gamma)
    P_samples = agent.rng.gamma(N_sasb)
    P_samples = P_samples + 1e-10  # Add small epsilon to avoid zeros
    if agent.stage_dependent:
        sums = P_samples.sum(-2, keepdims=True)
        P_samples = P_samples / sums
    else:
        sums = P_samples.sum(-1, keepdims=True)
        P_samples = P_samples / sums
    
    # Denormalize rewards back to [-1, 1] range
    R_samples = 2.0 * R_samples - 1.0
    
    # Run backward induction to compute Q-values for this episode
    from agents.OPSRL.utils import backward_induction_in_place, backward_induction_sd
    
    if agent.stage_dependent:
        backward_induction_sd(
            agent.Q, agent.V, R_samples, P_samples, agent.gamma, agent.v_max[0]
        )
    else:
        backward_induction_in_place(
            agent.Q,
            agent.V,
            R_samples,
            P_samples,
            agent.horizon,
            agent.gamma,
            agent.v_max[0],
        )
    
    # STEP 2: Now run the episode using the sampled Q-values
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    
    episode_reward = 0.0
    outcome = 'timeout'
    step = 0
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}
    
    for step in range(1, max_steps + 1):
        # Get map before action
        grid = env.render(mode="array")
        map_str = grid_to_string(grid)
        
        # Get state
        state = agent._obs_to_state(obs)
        
        # Choose action using sampling policy (Q-values from backward induction above)
        action = agent._get_action(state, hh=0)
        
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
        
        # STEP 3: Update agent's posterior during episode
        next_state = agent._obs_to_state(next_obs) if not done else None
        agent._update(state, action, next_state, reward, hh=0)
        
        # Update for next iteration
        obs = next_obs
        state = next_state
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    # Update episode counter
    agent.episode += 1
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win',
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compare 9 agents on RedBlueButton environment')
    parser.add_argument('--seed_idx', type=int, required=True,
                        help='Seed index to run (0-based). Will use BASE_SEED + seed_idx as the actual seed.')
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Total number of seeds (for filename generation, default: 30)')
    args = parser.parse_args()
    
    seed_idx = args.seed_idx
    NUM_SEEDS = args.num_seeds
    
    print("="*80)
    print("COMPARING: 9 Agents - Active Inference, Q-Learning, Dyna-Q variants, Trajectory Sampling, OPSRL")
    print("Same Environment Configurations - Fair Comparison")
    print(f"Running seed index {seed_idx} (seed {42 + seed_idx})")
    print("="*80)
    
    # Parameters
    NUM_EPISODES = 200  # Changed to 100 episodes
    EPISODES_PER_CONFIG = 25  # Change button positions every 20 episodes
    MAX_STEPS = 50
    PLANNING_STEPS = 2
    RECENCY_DECAYS = [0.99, 0.95, 0.90, 0.85]  # Different recency bias levels
    BASE_SEED = 42  # Base seed for reproducibility
    
    current_seed = BASE_SEED + seed_idx
    random.seed(current_seed)
    np.random.seed(current_seed)

    # Setup CSV logging (will include seed column)
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"nine_agents_comparison_ep{NUM_EPISODES}_step{MAX_STEPS}_seed{seed_idx}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    
    # Base filename for Q-tables (without extension)
    base_qtable_name = csv_filename.replace("_comparison_", "_qtable_").replace(".csv", "")
    
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, 
                                fieldnames=['seed', 'agent', 'episode', 'step', 'action', 'action_name', 'map', 'reward'])
    csv_writer.writeheader()
    
    # Show relative path for portability
    csv_path_relative = csv_path.relative_to(project_root)
    print(f"\nLogging to: logs/{csv_path_relative.name}")
    print(f"  Full path: {csv_path}")
    
    # Agent names (consistent across all seeds) - 9 agents total
    agent_names = ['AIF', 'QLearning', 'Vanilla', 'Recency0.99', 'Recency0.95', 'Recency0.9', 'Recency0.85', 'TrajSampling', 'OPSRL']
    
    print(f"\n{'='*80}")
    print(f"SEED INDEX {seed_idx} (seed={current_seed})")
    print(f"{'='*80}")
    
    # Setup all agents (1 AIF + 1 QL + 1 vanilla DynaQ + 4 recency variants + 1 trajectory sampling + 1 OPSRL)
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
    
    # 9. OPSRL agent
    print("\n9. OPSRL agent...")
    # Create a temporary env for initialization (will be updated per episode)
    temp_env = SingleAgentRedBlueButtonEnv(
        width=3,
        height=3,
        red_button_pos=(0, 2),
        blue_button_pos=(2, 0),
        agent_start_pos=(0, 0),
        max_steps=MAX_STEPS
    )
    opsrl_agent = OPSRLAgent(
        env=temp_env,
        gamma=0.95,
        horizon=MAX_STEPS,
        bernoullized_reward=True,
        scale_prior_reward=1.0,
        thompson_samples=1,
        prior_transition='uniform',
        reward_free=False,
        stage_dependent=False,
        seed=current_seed
    )
    agents.append(opsrl_agent)
    print("   ✓ OPSRL agent ready")
    
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
                       desc=f"Seed {seed_idx} Episodes", 
                       unit="ep", 
                       position=0, 
                       leave=True,
                       ncols=100)
    
    for episode in episode_pbar:
        # Determine current configuration
        config_idx = (episode - 1) // EPISODES_PER_CONFIG
        config = configs[config_idx]
        
        # Print configuration change (only for first episode of config)
        if (episode - 1) % EPISODES_PER_CONFIG == 0:
            print(f"\nConfig {config_idx+1}: Red at {config['red_pos']} (idx={config['red_idx']}), "
                  f"Blue at {config['blue_pos']} (idx={config['blue_idx']})")
        
        # Create environment with current configuration
        env = SingleAgentRedBlueButtonEnv(
            width=3,
            height=3,
            red_button_pos=config['red_pos'],
            blue_button_pos=config['blue_pos'],
            agent_start_pos=(0, 0),
            max_steps=MAX_STEPS
        )
        
        # Run all agents on the same environment
        episode_results = []
        
        # Progress bar for agents (nested, only shows during episode execution)
        agents_iter = zip(agents, agent_names)
        for agent_idx, (agent, agent_name) in enumerate(agents_iter):
            # Reset environment for each agent
            env.reset()
            
            # Run episode (use appropriate function for each agent type)
            if agent_name == 'AIF':
                result = run_aif_episode(env, agent, agent_name, episode, 
                                       max_steps=MAX_STEPS, csv_writer=csv_writer)
            elif agent_name == 'OPSRL':
                result = run_opsrl_episode(env, agent, agent_name, episode, 
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
        
        # Print episode summary every 50 episodes (outside progress bar)
        if episode % 50 == 0:
            episode_pbar.write(f"  Ep {episode}/{NUM_EPISODES} completed for seed {current_seed} - Success rate: {success_rate:.1f}%")
    
    # Close episode progress bar
    episode_pbar.close()
    
    # Close CSV file
    csv_file.close()
    csv_path_relative = csv_path.relative_to(project_root)
    print(f"\n✓ Seed {current_seed} (index {seed_idx}) completed")
    print(f"✓ Log saved to: logs/{csv_path_relative.name}")
    print(f"  Full path: {csv_path}")
    
    # Print summary statistics for this seed
    print("\n" + "="*80)
    print(f"RESULTS FOR SEED {current_seed} (index {seed_idx})")
    print("="*80)
    
    print(f"\n{'Agent':<15} {'Success Rate (%)':<20} {'Avg Reward':<20} {'Avg Steps':<15}")
    print(f"{'-'*70}")
    for agent_idx, agent_name in enumerate(agent_names):
        results = all_results[agent_idx]
        successes = sum(1 for r in results if r['success'])
        success_rate = 100 * successes / len(results)
        avg_reward = np.mean([r['reward'] for r in results])
        avg_step = np.mean([r['steps'] for r in results])
        
        print(f"{agent_name:<15} {success_rate:5.1f}{'':<14} {avg_reward:+6.3f}{'':<13} {avg_step:5.1f}")
    
    print("\n" + "="*80)
    print(f"✓ Single seed run complete!")
    print(f"Note: Aggregate results across all seeds can be computed after all jobs complete.")


if __name__ == "__main__":
    main()

