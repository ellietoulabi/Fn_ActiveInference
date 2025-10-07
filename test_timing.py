"""Quick timing test to identify bottlenecks."""

import time
import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)
from agents.ActiveInference.agent import Agent

# Setup
env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=10)
env.reset()

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

# Get first observation
env_obs, _ = env.reset()
model_obs = env_utils.env_obs_to_model_obs(env_obs)

print("Testing component timings...")
print("="*60)

# Time state inference
start = time.time()
agent.infer_states(model_obs)
t_infer_states = time.time() - start
print(f"State inference: {t_infer_states*1000:.1f}ms")

# Time policy inference (the slow one)
start = time.time()
agent.infer_policies()
t_infer_policies = time.time() - start
print(f"Policy inference: {t_infer_policies*1000:.1f}ms")

print(f"\nTotal step time: {(t_infer_states + t_infer_policies)*1000:.1f}ms")

