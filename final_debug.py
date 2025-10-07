"""Final debug: trace through one complete step."""

import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)
from agents.ActiveInference.agent import Agent

# Setup
env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=10)
env_obs, _ = env.reset()

d_config = env_utils.get_D_config_from_env(env)
D_init = D.D_fn(d_config)

state_factors = list(model_init.states.keys())
state_sizes = {k: len(v) for k, v in model_init.states.items()}

agent = Agent(
    A_fn=A.A_fn,
    B_fn=B.B_fn,
    C_fn=C.C_fn,
    D_fn=lambda config=None: D_init,
    state_factors=state_factors,
    state_sizes=state_sizes,
    observation_labels=model_init.observations,
    env_params={'width': 3, 'height': 3, 'B_NOISE_LEVEL': 0.0},
    policy_len=3,
    inference_horizon=3,
    action_selection="deterministic",
    num_iter=3,
)

agent.reset()

print("="*70)
print("STEP-BY-STEP TRACE")
print("="*70)

# Get observation
model_obs = env_utils.env_obs_to_model_obs(env_obs)
print(f"\n1. Observation: {model_obs}")

# Infer states
print(f"\n2. Infer states...")
agent.infer_states(model_obs)
print(f"   Agent position belief: most likely = {np.argmax(agent.qs['agent_pos'])}")

# Infer policies
print(f"\n3. Infer policies...")
q_pi, G = agent.infer_policies()
print(f"   q_pi type: {type(q_pi)}")
print(f"   q_pi shape: {q_pi.shape if hasattr(q_pi, 'shape') else 'N/A'}")
print(f"   Max q_pi: {np.max(q_pi)}")
best_policy_idx = np.argmax(q_pi)
print(f"   Best policy index: {best_policy_idx}")
best_policy = agent.policies[best_policy_idx]
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']
print(f"   Best policy: {' → '.join([action_names[a] for a in best_policy])}")
print(f"   First action of best policy: {best_policy[0]} ({action_names[best_policy[0]]})")

# Sample action
print(f"\n4. Sample action...")
action = agent.sample_action()
print(f"   Returned action: {action}")
print(f"   Action type: {type(action)}")
print(f"   Action name: {action_names[action]}")

print("\n" + "="*70)
if action == 1:
    print("SUCCESS! Agent selected DOWN (correct!)")
else:
    print(f"BUG! Agent selected {action_names[action]} instead of DOWN")
print("="*70)


