"""Compare EFE for UP-UP-UP vs DOWN-DOWN-OPEN."""

import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)
from agents.ActiveInference import control

# Setup
env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=10)
env.reset()

d_config = env_utils.get_D_config_from_env(env)
qs = D.D_fn(d_config)

state_factors = list(model_init.states.keys())
state_sizes = {k: len(v) for k, v in model_init.states.items()}
env_params = {'width': 3, 'height': 3, 'B_NOISE_LEVEL': 0.0}

action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']

policies_to_test = [
    ([0, 0, 0], "UP-UP-UP (stays at 0)"),
    ([1, 1, 4], "DOWN-DOWN-OPEN (reaches red button!)"),
]

print("="*80)
print("COMPARING POLICIES")
print("="*80)

for policy, description in policies_to_test:
    print(f"\nPolicy: {description}")
    print(f"  Actions: {' → '.join([action_names[a] for a in policy])}")
    
    # Get expected states
    qs_pi = control.get_expected_states(B.B_fn, qs, policy, env_params)
    
    print(f"\n  Predicted states over time:")
    for t, qs_t in enumerate(qs_pi):
        pos = np.argmax(qs_t['agent_pos'])
        red_state = np.argmax(qs_t['red_button_state'])
        blue_state = np.argmax(qs_t['blue_button_state'])
        print(f"    t={t+1}: pos={pos}, red={'pressed' if red_state else 'not_pressed'}, blue={'pressed' if blue_state else 'not_pressed'}")
    
    # Get expected obs and info gain
    qo_pi, info_gain = control.get_expected_obs_and_info_gain_unified(
        A.A_fn, qs_pi, state_factors, state_sizes, model_init.observations
    )
    
    # Calculate utility
    utility = control.calc_expected_utility(qo_pi, C.C_fn, model_init.observations)
    
    # EFE
    G = -utility - info_gain
    
    print(f"\n  Expected observations at each timestep:")
    for t, qo_t in enumerate(qo_pi):
        print(f"    t={t+1}:")
        for mod in ['on_red_button', 'red_button_state', 'button_just_pressed', 'game_result']:
            obs_idx = np.argmax(qo_t[mod])
            prob = qo_t[mod][obs_idx]
            obs_labels = model_init.observations[mod]
            obs_name = obs_labels[obs_idx] if isinstance(obs_labels[0], str) else obs_idx
            print(f"      {mod:25s}: {obs_name} (p={prob:.3f})")
    
    print(f"\n  Metrics:")
    print(f"    Utility:      {utility:+.4f}")
    print(f"    Info gain:    {info_gain:+.4f}")
    print(f"    EFE (G):      {G:+.4f} (LOWER is better)")

print("\n" + "="*80)
print("If DOWN-DOWN-OPEN doesn't have much lower EFE, there's a bug!")
print("="*80)


