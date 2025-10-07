"""Minimal test - just evaluate 1 step with limited policies."""

import time
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

# Only use 4 directional actions (no OPEN, no NOOP)
limited_actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT

agent = Agent(
    A_fn=A.A_fn,
    B_fn=B.B_fn,
    C_fn=C.C_fn,
    D_fn=lambda config=None: D_init,
    state_factors=state_factors,
    state_sizes=state_sizes,
    observation_labels=model_init.observations,
    env_params={'width': 3, 'height': 3, 'B_NOISE_LEVEL': 0.0},
    actions=limited_actions,  # Only 4 actions
    policy_len=1,
    inference_horizon=1,
    action_selection="deterministic",
    num_iter=3,
)

agent.reset()
model_obs = env_utils.env_obs_to_model_obs(env_obs)

print(f"Limited test: 4 actions, policy_len=1")
print("="*60)

start = time.time()
action = agent.step(model_obs)
elapsed = time.time() - start

print(f"agent.step() took: {elapsed*1000:.1f}ms")
print(f"Selected action: {action}")


