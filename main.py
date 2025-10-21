"""
Main script: Active Inference agent interacting with RedBlueButton environment.

Runs one episode showing beliefs, observations, and actions at each step.
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


def print_header():
    """Print header with grid layout."""
    print("="*80)
    print("ACTIVE INFERENCE AGENT - RED BLUE BUTTON TASK")
    print("="*80)
    print("\n3×3 Grid Layout:")
    print("  ┌─────┬─────┬─────┐")
    print("  │  0  │  1  │  2  │  ← Blue button at 2")
    print("  ├─────┼─────┼─────┤")
    print("  │  3  │  4  │  5  │")
    print("  ├─────┼─────┼─────┤")
    print("  │  6  │  7  │  8  │  ← Red button at 6")
    print("  └─────┴─────┴─────┘")
    print("  ↑")
    print("  Agent starts here (position 0)")
    print("\nGoal: Press RED button first, then BLUE button")
    print("="*80)


def print_step_info(step, obs_dict, qs, action, reward, info, agent):
    """Print detailed step information."""
    print(f"\n{'='*80}")
    print(f"STEP {step}")
    print(f"{'='*80}")
    
    # Observations
    print("\n📊 OBSERVATIONS:")
    print(f"  Position:             {obs_dict['agent_pos']}")
    print(f"  On red button:        {['FALSE', 'TRUE'][obs_dict['on_red_button']]}")
    print(f"  On blue button:       {['FALSE', 'TRUE'][obs_dict['on_blue_button']]}")
    print(f"  Red button state:     {['not_pressed', 'pressed'][obs_dict['red_button_state']]}")
    print(f"  Blue button state:    {['not_pressed', 'pressed'][obs_dict['blue_button_state']]}")
    print(f"  Game result:          {['neutral', 'win', 'lose'][obs_dict['game_result']]}")
    print(f"  Button just pressed:  {['FALSE', 'TRUE'][obs_dict['button_just_pressed']]}")
    
    # Beliefs (show most likely state for each factor)
    print("\n🧠 AGENT BELIEFS (Most Likely States):")
    for factor, belief in qs.items():
        most_likely_idx = np.argmax(belief)
        confidence = belief[most_likely_idx]
        
        # Format based on factor type
        if factor == 'agent_pos':
            print(f"  Agent position:       {most_likely_idx} (confidence: {confidence:.3f})")
        elif factor == 'red_button_pos':
            print(f"  Red button at:        {most_likely_idx} (confidence: {confidence:.3f})")
        elif factor == 'blue_button_pos':
            print(f"  Blue button at:       {most_likely_idx} (confidence: {confidence:.3f})")
        elif factor == 'red_button_state':
            state_name = ['not_pressed', 'pressed'][most_likely_idx]
            print(f"  Red button:           {state_name} (confidence: {confidence:.3f})")
        elif factor == 'blue_button_state':
            state_name = ['not_pressed', 'pressed'][most_likely_idx]
            print(f"  Blue button:          {state_name} (confidence: {confidence:.3f})")
    
    # Policies and their probabilities
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'OPEN', 5: 'NOOP'}
    print("\n📋 POLICY POSTERIOR (Top 5):")
    
    # Get top 5 policies by probability
    q_pi = agent.q_pi
    top_indices = np.argsort(q_pi)[-5:][::-1]  # Top 5 in descending order
    
    for rank, idx in enumerate(top_indices, 1):
        policy = agent.policies[idx]
        prob = q_pi[idx]
        policy_str = ' → '.join([action_names[a] for a in policy])
        print(f"  #{rank} [{idx:2d}]: {policy_str:20s} (p={prob:.4f})")
    
    # Action
    print(f"\n🎯 ACTION SELECTED:     {action} ({action_names[action]})")
    
    # Outcome
    print(f"\n📈 OUTCOME:")
    print(f"  Reward:               {reward:+.3f}")
    print(f"  Result:               {info.get('result', 'neutral')}")


def print_summary(episode_reward, step_count, outcome):
    """Print episode summary."""
    print("\n" + "="*80)
    print("EPISODE SUMMARY")
    print("="*80)
    print(f"  Total steps:          {step_count}")
    print(f"  Total reward:         {episode_reward:+.3f}")
    print(f"  Outcome:              {outcome.upper()}")
    
    if outcome == 'win':
        print("\n  🎉 SUCCESS! Agent pressed red then blue!")
    elif outcome == 'lose':
        print("\n  ❌ FAILED! Wrong button order or timeout.")
    else:
        print("\n  ⏸️  Episode incomplete.")
    
    print("="*80)


def main():
    """Run one episode of the agent interacting with environment."""
    
    # Print header
    print_header()
    
    # =========================================================================
    # 1. Setup Environment
    # =========================================================================
    print("\n1. Setting up environment...")
    env = SingleAgentRedBlueButtonEnv(
        width=3,
        height=3,
        red_button_pos=(0, 2),   # Position 6
        blue_button_pos=(2, 0),  # Position 2
        agent_start_pos=(0, 0),  # Position 0
        max_steps=100
    )
    env_obs, _ = env.reset()
    print("   ✓ Environment ready")
    
    # =========================================================================
    # 2. Setup Agent
    # =========================================================================
    print("\n2. Setting up agent...")
    
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    
    # Get D config from environment to ensure alignment
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
        actions=list(range(6)),  # UP, DOWN, LEFT, RIGHT, OPEN, NOOP
        policy_len=2,            # Plan 2 steps ahead
        gamma=1.0,              # High precision for near-deterministic policy selection
        alpha=1.0,              # High precision for near-deterministic action selection
        num_iter=16,             # Inference iterations
    )
    
    # Reset agent with environment-compatible config
    agent.reset(config=d_config)
    print("   ✓ Agent ready")
    print(f"   Planning horizon: {agent.policy_len} steps")
    print(f"   Number of policies: {len(agent.policies)}")
    
    # =========================================================================
    # 3. Run Episode
    # =========================================================================
    print("\n3. Running episode ...")
    print("-"*80)
    
    step = 0
    episode_reward = 0.0
    done = False
    
    while not done and step < 100:
        step += 1
        
        # Convert environment observation to model format
        model_obs = env_utils.env_obs_to_model_obs(env_obs)
        
        # Agent perceives, infers, plans, and acts
        action = agent.step(model_obs)
        
        # Get agent's current beliefs
        qs = agent.get_state_beliefs()
        
        # Print step information BEFORE taking action in environment
        # (showing what agent believes and decides)
        print_step_info(step, model_obs, qs, action, 0.0, {'result': 'pending'}, agent)
        
        # Take action in environment
        env_action = env_utils.model_action_to_env_action(action)
        env_obs, reward, terminated, truncated, info = env.step(env_action)
        
        # Update with actual reward and result
        print(f"\n  → Environment response:")
        print(f"     Reward:  {reward:+.3f}")
        print(f"     Result:  {info.get('result', 'neutral')}")
        
        episode_reward += reward
        done = terminated or truncated
        
        # Show game state visualization
        print(f"\n  Grid state:")
        grid_vis = env.render(mode='silent')
        for row in grid_vis:
            print(f"     {' '.join(row)}")
        print(f"     (A=agent, r/R=red button, b/B=blue button)")
    
    # =========================================================================
    # 4. Print Summary
    # =========================================================================
    outcome = info.get('result', 'neutral')
    print_summary(episode_reward, step, outcome)
    
    # =========================================================================
    # 5. Agent Diagnostics
    # =========================================================================
    print("\n" + "="*80)
    print("AGENT DIAGNOSTICS")
    print("="*80)
    
    # Get final beliefs
    final_qs = agent.get_state_beliefs()
    
    print("\nFinal belief distribution entropies:")
    for factor, belief in final_qs.items():
        entropy = -np.sum(belief * np.log(belief + 1e-16))
        max_entropy = np.log(len(belief))
        certainty = 100 * (1 - entropy / max_entropy)
        print(f"  {factor:20s}: H={entropy:.3f} ({certainty:.1f}% certain)")
    
    # Get action history
    print(f"\nAction history: {agent.prev_actions}")
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'OPEN', 5: 'NOOP'}
    action_seq = [action_names[a] for a in agent.prev_actions]
    print(f"Action sequence: {' → '.join(action_seq)}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
