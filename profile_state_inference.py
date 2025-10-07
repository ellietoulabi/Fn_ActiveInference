"""Profile state inference."""

import time
import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)
from agents.ActiveInference.agent import Agent

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
model_obs = env_utils.env_obs_to_model_obs(env_obs)

print("Testing state inference performance...")

# Warm up
for _ in range(5):
    agent.infer_states(model_obs)

# Time it
num_trials = 10
start = time.time()
for _ in range(num_trials):
    agent.infer_states(model_obs)
elapsed = time.time() - start

print(f"{num_trials} inferences in {elapsed*1000:.1f}ms")
print(f"Per inference: {elapsed/num_trials*1000:.1f}ms")


