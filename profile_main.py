"""Profile main.py to find bottleneck."""

import time
import cProfile
import pstats
from io import StringIO

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
    policy_len=1,
    inference_horizon=1,
    action_selection="deterministic",
    num_iter=3,
)

agent.reset()
model_obs = env_utils.env_obs_to_model_obs(env_obs)

print("Profiling one agent step...")
print("=" * 70)

# Profile
profiler = cProfile.Profile()
profiler.enable()

agent.step(model_obs)

profiler.disable()

# Print stats
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Top 30 functions
print(s.getvalue())


