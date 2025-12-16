"""
Environment <-> model utilities for ObservablePartner variant.

This is the decentralised MA model:
- Observations are converted per-agent (agent_id=1 or 2) into (my_pos, other_pos, ...)
"""

import numpy as np
from . import model_init


def xy_to_index(x, y, width=3):
    return y * width + x


def index_to_xy(index, width=3):
    y = index // width
    x = index % width
    return x, y


def env_obs_to_model_obs(env_obs, agent_id, width=3):
    if agent_id == 1:
        my_pos_xy = env_obs["agent1_position"]
        other_pos_xy = env_obs["agent2_position"]
        my_on_red = env_obs["agent1_on_red_button"]
        my_on_blue = env_obs["agent1_on_blue_button"]
    else:
        my_pos_xy = env_obs["agent2_position"]
        other_pos_xy = env_obs["agent1_position"]
        my_on_red = env_obs["agent2_on_red_button"]
        my_on_blue = env_obs["agent2_on_blue_button"]

    my_pos_idx = xy_to_index(int(my_pos_xy[0]), int(my_pos_xy[1]), width)
    other_pos_idx = xy_to_index(int(other_pos_xy[0]), int(other_pos_xy[1]), width)

    button_just_pressed = 0 if env_obs.get("button_just_pressed") is None else 1

    return {
        "my_pos": my_pos_idx,
        "other_pos": other_pos_idx,
        "my_on_red_button": int(my_on_red),
        "my_on_blue_button": int(my_on_blue),
        "red_button_state": int(env_obs["red_button_pressed"]),
        "blue_button_state": int(env_obs["blue_button_pressed"]),
        "game_result": int(env_obs["win_lose_neutral"]),
        "button_just_pressed": int(button_just_pressed),
    }


def get_D_config_from_env(env, agent_id):
    if agent_id == 1:
        my_start = env.agent1_start_pos
        other_start = env.agent2_start_pos
    else:
        my_start = env.agent2_start_pos
        other_start = env.agent1_start_pos

    my_start_idx = xy_to_index(my_start[0], my_start[1], env.width)
    other_start_idx = xy_to_index(other_start[0], other_start[1], env.width)

    return {
        "my_start_pos": my_start_idx,
        "other_start_pos": other_start_idx,
        "button_pos_uncertainty": True,
        "button_state_uncertainty": False,
    }


def model_action_to_env_action(model_action):
    return int(model_action)


def env_action_to_model_action(env_action):
    return int(env_action)


