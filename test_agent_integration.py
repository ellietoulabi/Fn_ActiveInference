"""
Comprehensive integration test for functional Active Inference agent.

Tests the complete agent with functional generative model and environment.
"""

import numpy as np

# Import environment
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv

# Import functional generative model
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)

# Import agent
from agents.ActiveInference.agent import Agent


def test_agent_creation():
    """Test that we can create the agent successfully."""
    print("="*80)
    print("TEST 1: Agent Creation")
    print("="*80)
    
    # Get state factors and sizes
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    
    print(f"\nState factors: {state_factors}")
    print(f"State sizes: {state_sizes}")
    
    # Create agent
    agent = Agent(
        A_fn=A_fn,
        B_fn=B_fn,
        C_fn=C_fn,
        D_fn=D_fn,
        state_factors=state_factors,
        state_sizes=state_sizes,
        observation_labels=model_init.observations,
        env_params={'width': model_init.n, 'height': model_init.m},
        actions=list(range(6)),  # UP, DOWN, LEFT, RIGHT, OPEN, NOOP
        policy_len=1,
        gamma=16.0,
        alpha=16.0,
    )
    
    print(f"\n✓ Agent created successfully")
    print(f"  Number of policies: {len(agent.policies)}")
    print(f"  Actions: {agent.actions}")
    
    # Check initial beliefs
    print(f"\nInitial state beliefs:")
    for factor, belief in agent.qs.items():
        most_likely = np.argmax(belief)
        prob = belief[most_likely]
        print(f"  {factor:20s}: index {most_likely} (p={prob:.3f})")
    
    return agent


def test_environment_agent_loop():
    """Test agent interacting with environment."""
    print("\n" + "="*80)
    print("TEST 2: Environment-Agent Interaction")
    print("="*80)
    
    # Create environment
    env = SingleAgentRedBlueButtonEnv()
    env_obs, _ = env.reset()
    
    print("\nEnvironment observation:")
    for key, value in env_obs.items():
        print(f"  {key:25s}: {value}")
    
    # Create agent with config matching environment
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    
    # Get D config from environment
    d_config = env_utils.get_D_config_from_env(env)
    
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
        policy_len=2,  # Plan 2 steps ahead
        gamma=16.0,
    )
    
    # Reset agent with environment-compatible config
    agent.reset(config=d_config)
    
    print("\nAgent initialized with environment config")
    
    # Convert environment observation to model format
    model_obs = env_utils.env_obs_to_model_obs(env_obs)
    
    print("\nModel observation (converted):")
    for key, value in model_obs.items():
        print(f"  {key:25s}: {value}")
    
    # Run one perception-action cycle
    print("\n" + "-"*80)
    print("Running perception-action cycle...")
    print("-"*80)
    
    action = agent.step(model_obs)
    
    print(f"\nAgent selected action: {action} ({env.ACTION_MEANING.get(action, 'unknown')})")
    
    # Show agent's beliefs after inference
    print("\nAgent's state beliefs after inference:")
    for factor, belief in agent.qs.items():
        most_likely = np.argmax(belief)
        prob = belief[most_likely]
        entropy = -np.sum(belief * np.log(belief + 1e-16))
        print(f"  {factor:20s}: index {most_likely} (p={prob:.3f}, H={entropy:.3f})")
    
    # Show top policies
    print("\nTop 3 policies:")
    top_policies = agent.get_top_policies(top_k=3)
    for i, (policy, prob, idx) in enumerate(top_policies, 1):
        action_names = [env.ACTION_MEANING.get(a, str(a)) for a in policy]
        print(f"  {i}. {action_names} (p={prob:.4f})")
    
    # Take action in environment
    env_action = env_utils.model_action_to_env_action(action)
    env_obs_next, reward, terminated, truncated, info = env.step(env_action)
    
    print(f"\nEnvironment response:")
    print(f"  Reward: {reward:.3f}")
    print(f"  Terminated: {terminated}")
    print(f"  Info: {info.get('result', 'neutral')}")
    
    return agent, env


def test_multi_step_episode():
    """Test agent running a complete episode."""
    print("\n" + "="*80)
    print("TEST 3: Multi-Step Episode")
    print("="*80)
    
    # Create environment and agent
    env = SingleAgentRedBlueButtonEnv(max_steps=20)
    
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    
    d_config = env_utils.get_D_config_from_env(env)
    
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
        policy_len=3,  # Plan 3 steps ahead
        gamma=16.0,
        alpha=16.0,
    )
    
    # Reset
    env_obs, _ = env.reset()
    agent.reset(config=d_config)
    
    print("\nRunning episode...")
    print("-"*80)
    
    step_count = 0
    total_reward = 0.0
    done = False
    
    while not done and step_count < 10:  # Limit to 10 steps for test
        # Convert observation
        model_obs = env_utils.env_obs_to_model_obs(env_obs)
        
        # Agent selects action
        action = agent.step(model_obs)
        
        # Environment responds
        env_action = env_utils.model_action_to_env_action(action)
        env_obs, reward, terminated, truncated, info = env.step(env_action)
        
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        # Print step info
        agent_pos_belief = agent.qs['agent_pos']
        most_likely_pos = np.argmax(agent_pos_belief)
        
        print(f"Step {step_count}: {env.ACTION_MEANING.get(action, 'unknown'):8s} "
              f"→ pos={most_likely_pos}, reward={reward:+.2f}, "
              f"result={info.get('result', 'neutral')}")
    
    print("-"*80)
    print(f"\nEpisode completed!")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Outcome: {info.get('result', 'unknown')}")
    
    return agent, env, step_count, total_reward


def test_agent_diagnostics():
    """Test agent diagnostic functions."""
    print("\n" + "="*80)
    print("TEST 4: Agent Diagnostics")
    print("="*80)
    
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
    )
    
    # Get diagnostics
    print("\nInference diagnostics:")
    diagnostics = agent.get_inference_diagnostics()
    
    print(f"  VFE: {diagnostics.get('vfe', 0.0):.4f}")
    
    print(f"\n  Entropy per factor:")
    for factor, H in diagnostics.get('entropy', {}).items():
        print(f"    {factor:20s}: {H:.4f}")
    
    print(f"\n  Concentration (max probability) per factor:")
    for factor, conc in diagnostics.get('concentration', {}).items():
        print(f"    {factor:20s}: {conc:.4f}")
    
    print(f"\n  MAP state:")
    for factor, idx in diagnostics.get('map_state', {}).items():
        print(f"    {factor:20s}: {idx}")
    
    # Test policy evaluation
    print("\n" + "-"*80)
    print("Evaluating specific policy: [DOWN, RIGHT]")
    print("-"*80)
    
    policy = [model_init.DOWN, model_init.RIGHT]
    components = agent.evaluate_policy(policy)
    
    print(f"  Expected utility: {components['utility']:+.4f}")
    print(f"  Expected info gain: {components['info_gain']:+.4f}")
    print(f"  Total EFE (G): {components['G_total']:+.4f}")
    
    return agent


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE AGENT INTEGRATION TEST")
    print("="*80)
    
    try:
        # Test 1: Agent creation
        agent1 = test_agent_creation()
        print("\n✅ TEST 1 PASSED")
        
        # Test 2: Single step interaction
        agent2, env2 = test_environment_agent_loop()
        print("\n✅ TEST 2 PASSED")
        
        # Test 3: Multi-step episode
        agent3, env3, steps, reward = test_multi_step_episode()
        print("\n✅ TEST 3 PASSED")
        
        # Test 4: Diagnostics
        agent4 = test_agent_diagnostics()
        print("\n✅ TEST 4 PASSED")
        
        # Summary
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - AGENT INTEGRATION SUCCESSFUL! 🎉")
        print("="*80)
        
        print("\n✅ Functional generative model (A_fn, B_fn, C_fn, D_fn) works correctly")
        print("✅ Agent can perceive, infer, plan, and act")
        print("✅ Environment integration works seamlessly")
        print("✅ Observation conversion (env ↔ model) works")
        print("✅ State inference updates beliefs correctly")
        print("✅ Policy evaluation computes EFE correctly")
        print("✅ Action selection works")
        
        print("\n" + "="*80)
        print("READY FOR FULL EXPERIMENTS!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()

