"""Debug why agent always chooses UP."""

import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)
from agents.ActiveInference.agent import Agent
from agents.ActiveInference import control

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
    policy_len=1,
    inference_horizon=1,
    action_selection="deterministic",
    num_iter=3,
)

agent.reset()

# First step
model_obs = env_utils.env_obs_to_model_obs(env_obs)
qs = agent.infer_states(model_obs)

print("="*70)
print("DEBUGGING POLICY SELECTION")
print("="*70)

print("\nCurrent beliefs:")
for factor, belief in qs.items():
    print(f"  {factor}: {belief}")
    print(f"    → Most likely: index {np.argmax(belief)} with p={np.max(belief):.3f}")

print("\nEvaluating each action...")
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']

for action in range(6):
    policy = [action]
    
    # Get expected states
    qs_pi = control.get_expected_states(B.B_fn, qs, policy, agent.env_params)
    
    # Get expected obs and info gain
    qo_pi, info_gain = control.get_expected_obs_and_info_gain_unified(
        A.A_fn, qs_pi, state_factors, state_sizes, model_init.observations
    )
    
    # Calculate utility
    utility = control.calc_expected_utility(qo_pi, C.C_fn, model_init.observations)
    
    # Calculate EFE
    G = -utility - info_gain
    
    print(f"\n  Action {action} ({action_names[action]}):")
    print(f"    Expected utility:     {utility:+.4f}")
    print(f"    Expected info gain:   {info_gain:+.4f}")
    print(f"    Expected Free Energy: {G:+.4f} (lower is better)")
    
    # Show predicted next position
    if 'agent_pos' in qs_pi[0]:
        next_pos_belief = qs_pi[0]['agent_pos']
        most_likely_pos = np.argmax(next_pos_belief)
        print(f"    Predicted position:   {most_likely_pos} (p={next_pos_belief[most_likely_pos]:.3f})")
    
    # Show predicted observations
    print(f"    Predicted observations:")
    for mod, qo in qo_pi[0].items():
        most_likely_obs = np.argmax(qo)
        print(f"      {mod:25s}: {most_likely_obs} (p={qo[most_likely_obs]:.3f})")

# Run actual policy inference
print("\n" + "="*70)
print("ACTUAL POLICY INFERENCE")
print("="*70)

q_pi, G_all = control.vanilla_fpi_update_posterior_policies(
    qs,
    A.A_fn,
    B.B_fn,
    C.C_fn,
    agent.policies,
    agent.env_params,
    state_factors,
    state_sizes,
    model_init.observations,
    use_utility=True,
    use_states_info_gain=True,
    gamma=agent.gamma
)

print("\nPolicy posteriors (q_pi):")
for idx, (policy, prob) in enumerate(zip(agent.policies, q_pi)):
    action = policy[0]
    print(f"  Policy {idx} (Action={action_names[action]:5s}): q(π)={prob:.4f}, G={G_all[idx]:+.4f}")

selected_action = agent.sample_action(q_pi, agent.policies, agent.action_selection, agent.alpha)
print(f"\nSelected action: {action_names[selected_action]} (index {selected_action})")


