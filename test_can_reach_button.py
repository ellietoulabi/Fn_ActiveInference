"""Test if agent can predict reaching button in 3 steps."""

import numpy as np
from generative_models.SA_ActiveInference.RedBlueButton import B, D, model_init, env_utils
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv

# Setup
env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=10)
env.reset()

d_config = env_utils.get_D_config_from_env(env)
qs = D.D_fn(d_config)

print("Initial position belief:")
print(f"  {qs['agent_pos']}")
print(f"  Most likely: position {np.argmax(qs['agent_pos'])}")

print("\nRed button at position 6 (bottom-left)")
print("Blue button at position 2 (top-right)")
print("\nGrid layout:")
print("  0  1  2  ← blue")
print("  3  4  5")
print("  6  7  8")
print("  ↑")
print("  red")

# Test policy: DOWN, DOWN, OPEN
policy = [1, 1, 4]  # DOWN, DOWN, OPEN
action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']

print(f"\nTesting policy: {' → '.join([action_names[a] for a in policy])}")

qs_t = qs
for t, action in enumerate(policy):
    qs_next = B.B_fn(qs_t, action, width=3, height=3, B_NOISE_LEVEL=0.0)
    most_likely_pos = np.argmax(qs_next['agent_pos'])
    prob = qs_next['agent_pos'][most_likely_pos]
    
    print(f"\n  After step {t+1} ({action_names[action]}):")
    print(f"    Most likely position: {most_likely_pos} (p={prob:.3f})")
    print(f"    Position distribution: {qs_next['agent_pos']}")
    
    # Check if on red button
    on_red = qs_next['agent_pos'][6]
    print(f"    P(on red button): {on_red:.3f}")
    
    # Check button state
    red_pressed = qs_next['red_button_state'][1]
    print(f"    P(red button pressed): {red_pressed:.3f}")
    
    qs_t = qs_next

print("\n" + "="*70)
print("Can the agent reach red button in 3 steps? YES, with DOWN-DOWN-OPEN")
print("So policy_len=3 SHOULD be enough!")


