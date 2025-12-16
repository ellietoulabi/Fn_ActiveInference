"""
Utility functions for Multi-Agent Active Inference with TwoAgentRedBlueButton.

Converts between environment observations and model format.
"""

import numpy as np
from . import model_init


def xy_to_index(x, y, width=3):
    """Convert (x, y) coordinates to flat grid index."""
    return y * width + x


def index_to_xy(index, width=3):
    """Convert flat grid index to (x, y) coordinates."""
    y = index // width
    x = index % width
    return x, y


def env_obs_to_model_obs(env_obs, agent_id, width=3):
    """
    Convert two-agent environment observation to model format for a specific agent.
    
    Parameters
    ----------
    env_obs : dict
        Environment observation with keys:
        - 'agent1_position': array [x, y]
        - 'agent2_position': array [x, y]
        - 'agent1_on_red_button': int
        - 'agent1_on_blue_button': int
        - 'agent2_on_red_button': int
        - 'agent2_on_blue_button': int
        - 'red_button_pressed': int
        - 'blue_button_pressed': int
        - 'win_lose_neutral': int
        - 'button_just_pressed': str or None
    agent_id : int
        Which agent (1 or 2)
    width : int
        Grid width
    
    Returns
    -------
    model_obs : dict
        Model observation for the specified agent
    """
    if agent_id == 1:
        my_pos_xy = env_obs['agent1_position']
        other_pos_xy = env_obs['agent2_position']
        my_on_red = env_obs['agent1_on_red_button']
        my_on_blue = env_obs['agent1_on_blue_button']
    else:
        my_pos_xy = env_obs['agent2_position']
        other_pos_xy = env_obs['agent1_position']
        my_on_red = env_obs['agent2_on_red_button']
        my_on_blue = env_obs['agent2_on_blue_button']
    
    my_pos_idx = xy_to_index(int(my_pos_xy[0]), int(my_pos_xy[1]), width)
    other_pos_idx = xy_to_index(int(other_pos_xy[0]), int(other_pos_xy[1]), width)
    
    # Button just pressed conversion
    if env_obs.get('button_just_pressed') is None:
        button_just_pressed = 0
    else:
        button_just_pressed = 1
    
    model_obs = {
        'my_pos': my_pos_idx,
        'other_pos': other_pos_idx,
        'my_on_red_button': int(my_on_red),
        'my_on_blue_button': int(my_on_blue),
        'red_button_state': int(env_obs['red_button_pressed']),
        'blue_button_state': int(env_obs['blue_button_pressed']),
        'game_result': int(env_obs['win_lose_neutral']),
        'button_just_pressed': button_just_pressed,
    }
    
    return model_obs


def get_D_config_from_env(env, agent_id):
    """
    Extract D_fn configuration from environment for a specific agent.
    
    Parameters
    ----------
    env : TwoAgentRedBlueButtonEnv
        Environment instance
    agent_id : int
        Which agent (1 or 2)
    
    Returns
    -------
    config : dict
        Configuration for D_fn
    """
    if agent_id == 1:
        my_start = env.agent1_start_pos
        other_start = env.agent2_start_pos
    else:
        my_start = env.agent2_start_pos
        other_start = env.agent1_start_pos
    
    my_start_idx = xy_to_index(my_start[0], my_start[1], env.width)
    other_start_idx = xy_to_index(other_start[0], other_start[1], env.width)
    
    config = {
        'my_start_pos': my_start_idx,
        'other_start_pos': other_start_idx,
        'button_pos_uncertainty': True,
        'button_state_uncertainty': False,
    }
    
    return config


def model_action_to_env_action(model_action):
    """Convert model action to environment action (identity mapping)."""
    return model_action


def env_action_to_model_action(env_action):
    """Convert environment action to model action (identity mapping)."""
    return env_action
