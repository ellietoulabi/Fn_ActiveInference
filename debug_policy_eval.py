"""Debug what happens during policy evaluation."""

import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)
from agents.ActiveInference import control

# Setup
env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=10)
env_obs, _ = env.reset()

d_config = env_utils.get_D_config_from_env(env)
qs = D.D_fn(d_config)

print("Initial qs (from D):")
for f, q in qs.items():
    ent = -np.sum(q * np.log(q + 1e-16))
    print(f"  {f:20s}: entropy={ent:.4f}")

# Simulate one action (DOWN) to see what happens
print("\nAfter simulating action=DOWN (1):")
qs_next = B.B_fn(qs, action=1, width=3, height=3, B_NOISE_LEVEL=0.0)
for f, q in qs_next.items():
    ent = -np.sum(q * np.log(q + 1e-16))
    print(f"  {f:20s}: entropy={ent:.4f}, non-zero states: {np.sum(q > 1e-10)}")

# Try get_expected_obs_from_beliefs with this uncertain qs
print("\nCalling get_expected_obs_from_beliefs with uncertain qs_next...")

state_factors = list(model_init.states.keys())
state_sizes = {k: len(v) for k, v in model_init.states.items()}

import time
start = time.time()
qo = control.get_expected_obs_from_beliefs(A.A_fn, qs_next, state_factors, state_sizes)
elapsed = time.time() - start

print(f"Time: {elapsed*1000:.1f}ms")
print("\nPredicted observations:")
for mod, q_o in qo.items():
    print(f"  {mod}: {q_o}")


