"""
Env <-> model utilities for FullyCollective paradigm.

This model sees the FULL joint observation and selects a JOINT action (a1, a2),
encoded as a single integer in [0, 35].
"""

from . import model_init


def xy_to_index(x, y, width=3):
    return y * width + x


def index_to_xy(index, width=3):
    y = index // width
    x = index % width
    return x, y


def encode_joint_action(a1, a2):
    return int(a1) * model_init.N_ACTIONS + int(a2)


def decode_joint_action(joint_action):
    a = int(joint_action)
    return a // model_init.N_ACTIONS, a % model_init.N_ACTIONS


def env_obs_to_model_obs(env_obs, width=3):
    """
    Convert TwoAgentRedBlueButton observation dict to joint-model observation indices.
    """
    a1_xy = env_obs["agent1_position"]
    a2_xy = env_obs["agent2_position"]

    a1_idx = xy_to_index(int(a1_xy[0]), int(a1_xy[1]), width)
    a2_idx = xy_to_index(int(a2_xy[0]), int(a2_xy[1]), width)

    button_just_pressed = 0 if env_obs.get("button_just_pressed") is None else 1

    return {
        "agent1_pos": a1_idx,
        "agent2_pos": a2_idx,
        "agent1_on_red_button": int(env_obs["agent1_on_red_button"]),
        "agent1_on_blue_button": int(env_obs["agent1_on_blue_button"]),
        "agent2_on_red_button": int(env_obs["agent2_on_red_button"]),
        "agent2_on_blue_button": int(env_obs["agent2_on_blue_button"]),
        "red_button_state": int(env_obs["red_button_pressed"]),
        "blue_button_state": int(env_obs["blue_button_pressed"]),
        "game_result": int(env_obs["win_lose_neutral"]),
        "button_just_pressed": int(button_just_pressed),
    }


def get_D_config_from_env(env):
    """
    Extract D_fn config from a TwoAgentRedBlueButtonEnv for the centralized model.
    """
    a1 = env.agent1_start_pos
    a2 = env.agent2_start_pos
    return {
        "agent1_start_pos": xy_to_index(int(a1[0]), int(a1[1]), env.width),
        "agent2_start_pos": xy_to_index(int(a2[0]), int(a2[1]), env.width),
        "button_pos_uncertainty": True,
        "button_state_uncertainty": False,
    }


def model_action_to_env_action(model_action):
    """
    Convert joint action index to (action1, action2) tuple for env.step(...).
    """
    return decode_joint_action(model_action)


def env_action_to_model_action(env_action):
    """
    Convert (action1, action2) tuple to joint action index.
    """
    a1, a2 = env_action
    return encode_joint_action(a1, a2)


def env_obs_to_model_obs_verbose(env_obs, width=3):
    """
    Convert environment observation to model format with verbose output.
    
    Same as env_obs_to_model_obs but prints conversion details.
    Useful for debugging.
    
    Parameters
    ----------
    env_obs : dict
        Environment observation
    width : int, optional
        Grid width (default: 3)
    
    Returns
    -------
    model_obs : dict
        Model observation indices
    """
    print("Converting Environment → Model Observation:")
    print("-" * 60)
    
    # Position conversion for joint model
    a1_xy = env_obs["agent1_position"]
    a2_xy = env_obs["agent2_position"]
    a1_idx = xy_to_index(int(a1_xy[0]), int(a1_xy[1]), width)
    a2_idx = xy_to_index(int(a2_xy[0]), int(a2_xy[1]), width)
    print(f"  agent1_position: {a1_xy} → agent1_pos: {a1_idx}")
    print(f"  agent2_position: {a2_xy} → agent2_pos: {a2_idx}")
    
    # Button just pressed conversion
    if env_obs.get("button_just_pressed") is None:
        button_just_pressed_idx = 0
        print(f"  button_just_pressed: None → 0 (FALSE)")
    else:
        button_just_pressed_idx = 1
        print(f"  button_just_pressed: '{env_obs['button_just_pressed']}' → 1 (TRUE)")
    
    # Direct mappings
    print(f"  agent1_on_red_button: {env_obs['agent1_on_red_button']}")
    print(f"  agent1_on_blue_button: {env_obs['agent1_on_blue_button']}")
    print(f"  agent2_on_red_button: {env_obs['agent2_on_red_button']}")
    print(f"  agent2_on_blue_button: {env_obs['agent2_on_blue_button']}")
    print(f"  red_button_pressed: {env_obs['red_button_pressed']} → red_button_state")
    print(f"  blue_button_pressed: {env_obs['blue_button_pressed']} → blue_button_state")
    print(f"  win_lose_neutral: {env_obs['win_lose_neutral']} → game_result")
    
    model_obs = env_obs_to_model_obs(env_obs, width)
    return model_obs
