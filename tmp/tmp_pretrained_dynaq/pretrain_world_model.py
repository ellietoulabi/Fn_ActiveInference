"""
Pre-train a world model for Dyna-Q agent.

This script explores the environment to build a world model that can be loaded
by Dyna-Q agents. This ensures fair comparison with Active Inference by giving
both agents the same initial world knowledge.

Usage:
    python pretrain_world_model.py --episodes 50 --planning_steps 2 --output pretrained_model.json
"""

import argparse
import numpy as np
from pathlib import Path
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import env_utils
from agents.QLearning import TabularDynaQAgent


def main():
    parser = argparse.ArgumentParser(description='Pre-train a world model for Dyna-Q')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes to explore (default: 50)')
    parser.add_argument('--planning_steps', type=int, default=2,
                       help='Planning steps per real step (default: 2)')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Max steps per episode (default: 50)')
    parser.add_argument('--output', default='pretrained_world_model.json',
                       help='Output model file (default: pretrained_world_model.json)')
    parser.add_argument('--configs', type=int, default=5,
                       help='Number of different button configurations (default: 5)')
    parser.add_argument('--episodes_per_config', type=int, default=10,
                       help='Episodes per configuration (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("="*80)
    print("PRE-TRAINING WORLD MODEL FOR DYNA-Q")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Planning steps: {args.planning_steps}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Configurations: {args.configs}")
    print(f"  Episodes per config: {args.episodes_per_config}")
    print(f"  Output: {args.output}")
    print(f"  Random seed: {args.seed}")
    
    # Create agent with model persistence
    agent = TabularDynaQAgent(
        action_space_size=6,
        planning_steps=args.planning_steps,
        q_table_path=None,  # Don't save Q-table
        model_path=args.output,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,  # High exploration to build good model
        epsilon_decay=0.99,
        min_epsilon=0.1,  # Keep some exploration
        load_existing_q_table=False,
        load_existing_model=False
    )
    
    print(f"\n✓ Agent created with {args.planning_steps} planning steps")
    
    # Generate environment configurations
    print(f"\nGenerating {args.configs} environment configurations...")
    configs = []
    
    for config_idx in range(args.configs):
        # Random button positions (avoid position 0 where agent starts)
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
        print(f"  Config {config_idx+1}: Red at {red_pos}, Blue at {blue_pos}")
    
    # Run exploration episodes
    print(f"\n{'='*80}")
    print("EXPLORING ENVIRONMENT")
    print(f"{'='*80}\n")
    
    episode_rewards = []
    episode_successes = []
    
    for episode in range(1, args.episodes + 1):
        # Get current config
        config_idx = (episode - 1) // args.episodes_per_config
        if config_idx >= len(configs):
            config_idx = len(configs) - 1
        config = configs[config_idx]
        
        # Create environment with current config
        env = SingleAgentRedBlueButtonEnv(
            width=3,
            height=3,
            red_button_pos=config['red_pos'],
            blue_button_pos=config['blue_pos'],
            agent_start_pos=(0, 0),
            max_steps=args.max_steps
        )
        
        # Print config header
        if (episode - 1) % args.episodes_per_config == 0:
            print(f"\nConfig {config_idx + 1}: Red at {config['red_pos']}, Blue at {config['blue_pos']}")
        
        # Run episode
        env_obs, _ = env.reset()
        obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        state = agent.get_state(obs_dict)
        
        total_reward = 0
        steps = 0
        
        for step in range(1, args.max_steps + 1):
            action = agent.choose_action(state)
            env_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_obs_dict = env_utils.env_obs_to_model_obs(env_obs)
            next_state = agent.get_state(next_obs_dict) if not done else None
            
            # Update Q-table and model
            agent.update_q_table(state, action, reward, next_state)
            agent.update_model(state, action, next_state, reward, terminated)
            agent.planning()
            
            total_reward += reward
            steps = step
            state = next_state
            
            if done:
                break
        
        agent.decay_exploration()
        
        outcome = info.get('result', 'neutral')
        success = outcome == 'win'
        episode_rewards.append(total_reward)
        episode_successes.append(success)
        
        status = "✅" if success else "❌"
        print(f"  Episode {episode:3d}: {status} reward={total_reward:+.2f}, steps={steps:2d}, "
              f"ε={agent.epsilon:.3f}, model={len(agent.model)} states")
        
        # Print summary every 10 episodes
        if episode % 10 == 0:
            recent_success_rate = 100 * np.mean(episode_successes[-10:])
            recent_avg_reward = np.mean(episode_rewards[-10:])
            print(f"    → Last 10 episodes: {recent_success_rate:.0f}% success, "
                  f"avg reward: {recent_avg_reward:+.2f}")
    
    # Save the model
    print(f"\n{'='*80}")
    print("SAVING WORLD MODEL")
    print(f"{'='*80}\n")
    
    agent.save_model()
    
    # Print final statistics
    stats = agent.get_stats()
    
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Training Performance:")
    print(f"  Total episodes: {args.episodes}")
    print(f"  Success rate: {100 * np.mean(episode_successes):.1f}%")
    print(f"  Average reward: {np.mean(episode_rewards):+.2f}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    
    print(f"\nWorld Model Statistics:")
    print(f"  States in model: {stats['model_size']}")
    print(f"  Total transitions: {stats['total_transitions']}")
    print(f"  (state, action) pairs: {stats['visited_state_actions']}")
    print(f"  Average transitions per state: {stats['total_transitions'] / max(stats['model_size'], 1):.1f}")
    
    print(f"\nModel saved to: {args.output}")
    print(f"✓ Pre-training complete!")
    
    print(f"\n{'='*80}")
    print("HOW TO USE THIS MODEL")
    print(f"{'='*80}\n")
    print("To use this pre-trained model in your comparison:")
    print("")
    print("from agents.QLearning import TabularDynaQAgent")
    print("")
    print("agent = TabularDynaQAgent(")
    print("    action_space_size=6,")
    print(f"    planning_steps={args.planning_steps},")
    print(f"    model_path='{args.output}',")
    print("    load_existing_model=True  # Load the pre-trained model")
    print(")")
    print("")
    print("This gives Dyna-Q the same world knowledge that Active Inference")
    print("has built into its generative model (A, B, C, D functions).")
    print("")


if __name__ == "__main__":
    main()


