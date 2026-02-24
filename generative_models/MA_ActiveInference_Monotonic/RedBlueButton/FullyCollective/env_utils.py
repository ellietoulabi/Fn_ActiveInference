"""
Env <-> model utilities for FullyCollective paradigm.

Conceptually: one AIF agent (central planner) and one follower. The AIF decides
the full joint action (a1, a2); the follower only executes the action the AIF
assigns to it (no own policy). At step time: agent_0 executes a1, agent_1 executes a2.
Mapping: model agent1 <-> env agent_0,  model agent2 <-> env agent_1.
"""

from . import model_init


def merge_env_obs_for_collective(env_obs):
    """
    Convert TwoAgentRedBlueButton env observation to joint format for the collective model.

    The env returns observations = {'agent_0': obs_0, 'agent_1': obs_1}, where each
    obs has 'position', 'on_red_button', 'on_blue_button', 'red_button_pressed',
    'blue_button_pressed', 'win_lose_neutral', 'button_just_pressed'.

    Returns a single dict with agent1_* / agent2_* keys expected by env_obs_to_model_obs.
    """
    o0 = env_obs["agent_0"]
    o1 = env_obs["agent_1"]
    pos0 = o0["position"]
    pos1 = o1["position"]
    # position is (x, y) from env - as array or tuple
    xy0 = (int(pos0[0]), int(pos0[1])) if hasattr(pos0, "__len__") else (int(pos0), 0)
    xy1 = (int(pos1[0]), int(pos1[1])) if hasattr(pos1, "__len__") else (int(pos1), 0)
    return {
        "agent1_position": xy0,
        "agent2_position": xy1,
        "agent1_on_red_button": int(o0["on_red_button"]),
        "agent1_on_blue_button": int(o0["on_blue_button"]),
        "agent2_on_red_button": int(o1["on_red_button"]),
        "agent2_on_blue_button": int(o1["on_blue_button"]),
        "red_button_pressed": int(o0["red_button_pressed"]),
        "blue_button_pressed": int(o0["blue_button_pressed"]),
        "win_lose_neutral": int(o0["win_lose_neutral"]),
        "button_just_pressed": o0.get("button_just_pressed"),
    }


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
    Extract D_fn config from the two-agent env for the centralized model.
    Supports both env styles: agent1/agent2 (TwoAgentRedBlueButton) or
    agent_0/agent_1 (TwoAgentRedBlueButtonEnv).
    """
    if hasattr(env, "agent1_start_pos"):
        a1 = env.agent1_start_pos
        a2 = env.agent2_start_pos
    else:
        a1 = env.agent_0_start_pos
        a2 = env.agent_1_start_pos
    return {
        "agent1_start_pos": xy_to_index(int(a1[0]), int(a1[1]), env.width),
        "agent2_start_pos": xy_to_index(int(a2[0]), int(a2[1]), env.width),
        "button_pos_uncertainty": True,
        "button_state_uncertainty": False,
    }


def model_action_to_env_action(model_action):
    """
    Convert joint action index to (a1, a2) for env.step(...).
    One planner chose the joint action; agent_0 executes a1, agent_1 executes a2.
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
