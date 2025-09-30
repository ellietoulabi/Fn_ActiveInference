#!/usr/bin/env python3
"""
Main script to run Active Inference agent in RedBlueButton environment
"""

import numpy as np
import jax.numpy as jnp
from agents.ActiveInference import FunctionalAgent
from agents.ActiveInference.maths import build_decode_table
from generative_models.SA_ActiveInference.RedBlueButton.B import apply_B
from generative_models.SA_ActiveInference.RedBlueButton.A_noisy import A_funcs_noisy
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv

def main():
    print("=== Active Inference Agent in RedBlueButton Environment ===")
    print("Running for 50 steps with policy length 1")
    print("=" * 60)
    
    # Create environment
    env = SingleAgentRedBlueButtonEnv(width=3, height=3)
    
    # Create A functions (wrappers for the agent)
    def A_agent_pos(state_tuple, width, height):
        S = width * height
        return A_funcs_noisy["agent_pos"](state_tuple, S)
    
    def A_on_red_button(state_tuple, width, height):
        S = 2
        return A_funcs_noisy["on_red_button"](state_tuple, S)
    
    def A_on_blue_button(state_tuple, width, height):
        S = 2
        return A_funcs_noisy["on_blue_button"](state_tuple, S)
    
    def A_red_button_state(state_tuple, width, height):
        S = 2
        return A_funcs_noisy["red_button_state"](state_tuple, S)
    
    def A_blue_button_state(state_tuple, width, height):
        S = 2
        return A_funcs_noisy["blue_button_state"](state_tuple, S)
    
    def A_game_result(state_tuple, width, height):
        S = 3
        return A_funcs_noisy["game_result"](state_tuple, S)
    
    A_funcs = {
        'agent_pos': A_agent_pos,
        'on_red_button': A_on_red_button,
        'on_blue_button': A_on_blue_button,
        'red_button_state': A_red_button_state,
        'blue_button_state': A_blue_button_state,
        'game_result': A_game_result
    }
    
    # Create mock C functions that return preference vectors directly
    def C_agent_pos():
        # Prefer center positions
        prefs = np.ones(9) * 0.1
        prefs[4] = 1.0  # Prefer center
        return prefs
    
    def C_on_red_button():
        return np.array([0.1, 0.9])  # Prefer being on red button
    
    def C_on_blue_button():
        return np.array([0.1, 0.9])  # Prefer being on blue button
    
    def C_red_button_state():
        return np.array([0.1, 0.9])  # Prefer red button pressed
    
    def C_blue_button_state():
        return np.array([0.1, 0.9])  # Prefer blue button pressed
    
    def C_game_result():
        return np.array([0.1, 0.9, 0.1])  # Prefer winning
    
    C_funcs = {
        'agent_pos': C_agent_pos,
        'on_red_button': C_on_red_button,
        'on_blue_button': C_on_blue_button,
        'red_button_state': C_red_button_state,
        'blue_button_state': C_blue_button_state,
        'game_result': C_game_result
    }
    
    # Initial beliefs
    D_funcs = {
        'agent_pos': np.ones(9) / 9,
        'red_door_pos': np.ones(9) / 9,
        'blue_door_pos': np.ones(9) / 9,
        'red_door_state': np.array([0.8, 0.2]),
        'blue_door_state': np.array([0.8, 0.2]),
        'goal_context': np.array([0.5, 0.5])
    }
    
    # Create decode table
    decode_table = build_decode_table(3, 3)
    
    # Environment parameters
    env_params = {
        'width': 3,
        'height': 3,
        'obs_sizes': {
            'agent_pos': 9,
            'on_red_button': 2,
            'on_blue_button': 2,
            'red_button_state': 2,
            'blue_button_state': 2,
            'game_result': 3
        }
    }
    
    # Create agent
    agent = FunctionalAgent(
        A_funcs=A_funcs,
        B_func=apply_B,
        C_funcs=C_funcs,
        D_funcs=D_funcs,
        decode_table=decode_table,
        env_params=env_params,
        policy_len=1,  # Policy length 1 as requested
        gamma=16.0,
        alpha=16.0
    )
    
    print(f"Agent created with {len(agent.policies)} policies")
    print(f"Actions: UP(0), DOWN(1), LEFT(2), RIGHT(3), OPEN(4), NOOP(5)")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Environment reset. Initial observation keys: {list(obs.keys())}")
    
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "OPEN", "NOOP"]
    
    # Run for 50 steps
    for step in range(50):
        print(f"\n--- STEP {step + 1} ---")
        
        # Show current observations (convert to proper format)
        agent_pos = int(obs['position'][0] * 3 + obs['position'][1])  # Convert [row, col] to index
        on_red = int(obs['on_red_button'])
        on_blue = int(obs['on_blue_button'])
        red_pressed = int(obs['red_button_pressed'])
        blue_pressed = int(obs['blue_button_pressed'])
        game_result = int(obs['win_lose_neutral'])
        
        print(f"Agent at position: {agent_pos} (row {obs['position'][0]}, col {obs['position'][1]})")
        print(f"On red button: {bool(on_red)}")
        print(f"On blue button: {bool(on_blue)}")
        print(f"Red button pressed: {bool(red_pressed)}")
        print(f"Blue button pressed: {bool(blue_pressed)}")
        print(f"Game state: {['neutral', 'win', 'lose'][game_result]}")
        
        # Convert environment observations to agent format (one-hot arrays)
        def int_to_onehot(value, size):
            arr = np.zeros(size)
            arr[value] = 1.0
            return arr
        
        agent_obs = {
            'agent_pos': int_to_onehot(agent_pos, 9),
            'on_red_button': int_to_onehot(on_red, 2),
            'on_blue_button': int_to_onehot(on_blue, 2),
            'red_button_state': int_to_onehot(red_pressed, 2),
            'blue_button_state': int_to_onehot(blue_pressed, 2),
            'game_result': int_to_onehot(game_result, 3)
        }
        
        # Agent inference
        print("Agent processing...")
        
        # State inference
        qs = agent.infer_states(agent_obs)
        print(f"Agent beliefs: pos {np.argmax(qs['agent_pos'])} (confidence: {np.max(qs['agent_pos']):.3f})")
        
        # Policy inference
        q_pi, G = agent.infer_policies()
        best_policy = np.argmax(q_pi)
        print(f"Best policy: {best_policy} (confidence: {q_pi[best_policy]:.3f})")
        
        # Action selection
        action = agent.sample_action()
        print(f"Selected action: {action} ({action_names[action]})")
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        
        # Step agent time
        agent.step_time()
        
        # Check if episode ended
        if done:
            print(f"\n🎉 Episode ended at step {step + 1}!")
            break
    
    print(f"\n=== Simulation Complete ===")
    print(f"Final position: {np.argmax(agent_obs['agent_pos'])}")
    print(f"Final game state: {['neutral', 'win', 'lose'][np.argmax(agent_obs['game_result'])]}")

if __name__ == "__main__":
    main()