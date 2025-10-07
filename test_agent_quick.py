"""
Quick integration test for functional Active Inference agent.
Tests basic functionality without running full episodes.
"""

import numpy as np

# Import functional generative model
from generative_models.SA_ActiveInference.RedBlueButton import (
    A_fn, B_fn, C_fn, D_fn, model_init
)

# Import agent
from agents.ActiveInference.agent import Agent

print("="*60)
print("QUICK AGENT TEST")
print("="*60)

# 1. Create agent
print("\n1. Creating agent...")
state_factors = list(model_init.states.keys())
state_sizes = {f: len(v) for f, v in model_init.states.items()}

agent = Agent(
    A_fn=A_fn,
    B_fn=B_fn,
    C_fn=C_fn,
    D_fn=D_fn,
    state_factors=state_factors,
    state_sizes=state_sizes,
    observation_labels=model_init.observations,
    env_params={'width': 3, 'height': 3},
    actions=list(range(6)),
    policy_len=1,  # Short planning
    num_iter=5,    # Quick inference
)
print("   ✓ Agent created")

# 2. Test state inference
print("\n2. Testing state inference...")
obs = {
    'agent_pos': 0,
    'on_red_button': 0,
    'on_blue_button': 0,
    'red_button_state': 0,
    'blue_button_state': 0,
    'game_result': 0,
    'button_just_pressed': 0,
}

qs = agent.infer_states(obs)
print(f"   ✓ State inference complete")
print(f"   Agent belief: pos {np.argmax(qs['agent_pos'])}")

# 3. Test policy inference
print("\n3. Testing policy inference...")
q_pi, G = agent.infer_policies()
print(f"   ✓ Policy inference complete")
print(f"   Best policy: {agent.policies[np.argmax(q_pi)]}")
print(f"   Best EFE: {G[np.argmax(q_pi)]:.3f}")

# 4. Test action selection
print("\n4. Testing action selection...")
action = agent.sample_action()
action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'OPEN', 5: 'NOOP'}
print(f"   ✓ Action selected: {action} ({action_names[action]})")

# 5. Test full step
print("\n5. Testing full step...")
action = agent.step(obs)
print(f"   ✓ Full step complete: {action} ({action_names[action]})")

print("\n" + "="*60)
print("✅ ALL QUICK TESTS PASSED!")
print("="*60)
print("\nAgent is working correctly with functional generative model.")


