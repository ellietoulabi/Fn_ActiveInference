"""
Run 10 episodes and track success rate.
"""

import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)
from agents.ActiveInference.agent import Agent


def run_episode(env, agent, max_steps=100, verbose=False):
    """Run one episode and return outcome."""
    
    # Reset environment and agent
    env_obs, _ = env.reset()
    d_config = env_utils.get_D_config_from_env(env)
    agent.reset(config=d_config)
    
    # Convert initial observation
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    agent.infer_states(obs_dict)
    
    episode_reward = 0.0
    outcome = 'timeout'
    
    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\rStep {step}/{max_steps}", end='', flush=True)
        
        # Select action
        action = agent.step(obs_dict)
        
        # Execute action in environment
        env_obs, reward, done, _, info = env.step(action)
        episode_reward += reward
        
        # Update observation
        obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        
        if done:
            outcome = info.get('result', 'neutral')
            break
    
    if verbose:
        print()
    
    return {
        'outcome': outcome,
        'reward': episode_reward,
        'steps': step,
        'success': outcome == 'win'
    }


def main():
    print("="*80)
    print("RUNNING 10 EPISODES - RED BLUE BUTTON TASK")
    print("="*80)
    
    # Setup environment
    print("\nSetting up environment...")
    env = SingleAgentRedBlueButtonEnv(
        width=3,
        height=3,
        red_button_pos=(0, 2),   # Position 6
        blue_button_pos=(2, 0),  # Position 2
        agent_start_pos=(0, 0),  # Position 0
        max_steps=100
    )
    print("✓ Environment ready")
    
    # Setup agent
    print("Setting up agent...")
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
    print("✓ Agent ready")
    print(f"  Planning horizon: {agent.policy_len} steps")
    print(f"  Number of policies: {len(agent.policies)}\n")
    
    # Run episodes
    print("="*80)
    print("RUNNING EPISODES")
    print("="*80)
    
    results = []
    for episode in range(1, 11):
        print(f"\nEpisode {episode}/10:", end=' ')
        result = run_episode(env, agent, max_steps=100, verbose=False)
        results.append(result)
        
        # Print result
        status = "✅ WIN" if result['success'] else "❌ FAIL"
        print(f"{status} - {result['outcome']} (steps: {result['steps']}, reward: {result['reward']:+.3f})")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successes = sum(1 for r in results if r['success'])
    failures = len(results) - successes
    success_rate = 100 * successes / len(results)
    
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"\nTotal episodes:  {len(results)}")
    print(f"Successes:       {successes} ({success_rate:.1f}%)")
    print(f"Failures:        {failures} ({100-success_rate:.1f}%)")
    print(f"\nAverage reward:  {avg_reward:+.3f}")
    print(f"Average steps:   {avg_steps:.1f}")
    
    # Breakdown by outcome
    outcomes = {}
    for r in results:
        outcome = r['outcome']
        if outcome not in outcomes:
            outcomes[outcome] = 0
        outcomes[outcome] += 1
    
    print("\nOutcome breakdown:")
    for outcome, count in sorted(outcomes.items()):
        pct = 100 * count / len(results)
        print(f"  {outcome:10s}: {count:2d} ({pct:5.1f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

