"""Test just ONE step with policy_len=5 to see if it works."""

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

agent = Agent(
    A_fn=A.A_fn,
    B_fn=B.B_fn,
    C_fn=C.C_fn,
    D_fn=lambda config=None: D_init,
    state_factors=state_factors,
    state_sizes=state_sizes,
    observation_labels=model_init.observations,
    env_params={'width': 3, 'height': 3, 'B_NOISE_LEVEL': 0.0},
    policy_len=5,
    inference_horizon=5,
    gamma=2.0,
    action_selection="deterministic",
    num_iter=3,
)

agent.reset()

print(f"Agent initialized with policy_len=5")
print(f"Number of policies to evaluate: {len(agent.policies)} (6^5 = 7776)")
print("\nRunning ONE step...")

model_obs = env_utils.env_obs_to_model_obs(env_obs)

start = time.time()
action = agent.step(model_obs)
elapsed = time.time() - start

action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']
print(f"\nSelected action: {action_names[action]}")
print(f"Time: {elapsed:.1f}s")

if elapsed > 60:
    print("\nTOO SLOW! policy_len=5 is not practical.")
else:
    print(f"\nFeasible! ~{elapsed*10:.0f}s for 10 steps, ~{elapsed*150:.0f}s for 150 steps")


