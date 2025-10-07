"""Test 10-step episode with optimized agent."""

import time
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)
from agents.ActiveInference.agent import Agent

print("Initializing...")
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
    policy_len=3,
    inference_horizon=3,
    gamma=4.0,  # Lower for exploration
    action_selection="deterministic",
    num_iter=3,
)

agent.reset()
print("Running 10-step episode...\n")

action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']
total_time = 0

for step in range(10):
    model_obs = env_utils.env_obs_to_model_obs(env_obs)
    
    start = time.time()
    action = agent.step(model_obs)
    step_time = time.time() - start
    total_time += step_time
    
    print(f"Step {step+1}: Pos={env_obs['position']}, Action={action_names[action]}, Time={step_time*1000:.1f}ms")
    
    env_obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"\nEpisode ended: {info}")
        print(f"Reward: {reward}")
        break

print(f"\nTotal episode time: {total_time*1000:.1f}ms")
print(f"Average step time: {total_time/min(step+1, 10)*1000:.1f}ms")

