"""Check which factors are being treated as dynamic."""

import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)

# Setup
env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=10)
env_obs, _ = env.reset()

d_config = env_utils.get_D_config_from_env(env)
D_init = D.D_fn(d_config)

# Check entropy of each factor
print("Initial beliefs (D_init):")
print("=" * 60)

ENTROPY_THRESHOLD = 0.01

for factor, q_f in D_init.items():
    entropy = -np.sum(q_f * np.log(q_f + 1e-16))
    is_dynamic = entropy > ENTROPY_THRESHOLD
    max_prob = np.max(q_f)
    map_idx = int(np.argmax(q_f))
    
    print(f"{factor:20s}: entropy={entropy:.4f}, max_prob={max_prob:.3f}, MAP={map_idx:2d}  {'[DYNAMIC]' if is_dynamic else '[STATIC]'}")

# Count dynamic states
dynamic_count = 1
for factor, q_f in D_init.items():
    entropy = -np.sum(q_f * np.log(q_f + 1e-16))
    if entropy > ENTROPY_THRESHOLD:
        dynamic_count *= len(q_f)

print(f"\nTotal state combos if enumerating all dynamic factors: {dynamic_count}")


