"""Run agent for 150 steps with compact output."""

import time
import numpy as np
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import (
    A, B, C, D, model_init, env_utils
)
from agents.ActiveInference.agent import Agent

print("="*80)
print("ACTIVE INFERENCE AGENT - 150 STEP TEST")
print("="*80)

# Setup
env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=150)
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
    policy_len=3,  # Need depth to plan route to buttons
    inference_horizon=3,
    gamma=4.0,  # Lower = more exploration (was 16.0)
    action_selection="deterministic",
    num_iter=3,
)

agent.reset()

action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'OPEN', 'NOOP']
step_times = []
episode_reward = 0.0

print("\nRunning episode...")
print("  (Showing every 10th step and key events)\n")

for step in range(1, 151):
    model_obs = env_utils.env_obs_to_model_obs(env_obs)
    
    # Agent step - TIMED
    start = time.time()
    action = agent.step(model_obs)
    step_time = time.time() - start
    step_times.append(step_time)
    
    # Take action
    env_obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    
    # Show progress every 10 steps or on important events
    show_step = (step % 10 == 0 or 
                 env_obs.get('button_just_pressed') is not None or
                 terminated or truncated)
    
    if show_step:
        pos = env_obs['position']
        red = '✓' if env_obs['red_button_pressed'] else '✗'
        blue = '✓' if env_obs['blue_button_pressed'] else '✗'
        just_pressed = env_obs.get('button_just_pressed')
        event = f" [PRESSED: {just_pressed}]" if just_pressed else ""
        
        print(f"  Step {step:3d}: Pos=({pos[0]},{pos[1]}), Action={action_names[action]:5s}, "
              f"Red={red}, Blue={blue}, Time={step_time*1000:5.1f}ms{event}")
    
    if terminated or truncated:
        print(f"\n  Episode ended: {info.get('result', 'unknown')}")
        break

# Summary
print("\n" + "="*80)
print("EPISODE SUMMARY")
print("="*80)
print(f"  Total steps:        {step}")
print(f"  Total reward:       {episode_reward:.3f}")
print(f"  Result:             {info.get('result', 'unknown')}")
print(f"  Red pressed:        {'Yes' if env_obs['red_button_pressed'] else 'No'}")
print(f"  Blue pressed:       {'Yes' if env_obs['blue_button_pressed'] else 'No'}")
print()
print(f"  Total time:         {sum(step_times)*1000:.1f}ms")
print(f"  Average step time:  {np.mean(step_times)*1000:.1f}ms")
print(f"  Min step time:      {np.min(step_times)*1000:.1f}ms")
print(f"  Max step time:      {np.max(step_times)*1000:.1f}ms")
print(f"  Median step time:   {np.median(step_times)*1000:.1f}ms")

# Show final grid
print("\nFinal grid state:")
grid_vis = env.render(mode='silent')
for row in grid_vis:
    print(f"  {' '.join(row)}")
print("  (A=agent, r/R=red button, b/B=blue button)")

print("\n" + "="*80)
print("DONE!")
print("="*80)

