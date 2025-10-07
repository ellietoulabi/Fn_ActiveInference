"""Debug actual agent policy evaluation with policy_len=3."""

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

print(f"Agent has {len(agent.policies)} policies")
print(f"Policy length: {agent.policy_len}")

# First step
model_obs = env_utils.env_obs_to_model_obs(env_obs)
qs = agent.infer_states(model_obs)

# Run policy inference (returns just the selected action)
# We need to call the control function directly to get q_pi
print("\nRunning policy inference...")
from agents.ActiveInference import control

q_pi, G = control.vanilla_fpi_update_posterior_policies(
    qs,
    A.A_fn,
    B.B_fn,
    C.C_fn,
    agent.policies,
    agent.env_params,
    agent.state_factors,
    agent.state_sizes,
    agent.observation_labels,
    use_utility=True,
    use_states_info_gain=True,
    gamma=agent.gamma
)

# Find top policies
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']

top_indices = np.argsort(q_pi)[-10:][::-1]  # Top 10

print(f"\nTop 10 policies (out of {len(agent.policies)}):")
for rank, idx in enumerate(top_indices, 1):
    policy = agent.policies[idx]
    policy_str = ' → '.join([action_names[a] for a in policy])
    print(f"  {rank:2d}. Policy {idx:3d}: {policy_str:25s}  q(π)={q_pi[idx]:.6f}, G={G[idx]:+.4f}")

# Find DOWN-DOWN-OPEN
target_policy = [1, 1, 4]
for idx, policy in enumerate(agent.policies):
    if list(policy) == target_policy:
        print(f"\nDOWN-DOWN-OPEN is policy {idx}: q(π)={q_pi[idx]:.6f}")
        rank = np.sum(q_pi > q_pi[idx]) + 1
        print(f"  Rank: {rank}/{len(agent.policies)}")
        break

# Check which action will be selected
selected = agent.sample_action(q_pi, agent.policies, agent.action_selection, agent.alpha)
print(f"\nSelected action: {action_names[selected[0]]} (from policy: {' → '.join([action_names[a] for a in selected])})")

