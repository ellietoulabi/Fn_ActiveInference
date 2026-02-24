"""
Env <-> model utilities for the Independent baseline.

This adapter converts TwoAgentRedBlueButton observations into SA RedBlueButton model
observations from a specific agent's perspective.
"""

import numpy as np

from generative_models.SA_ActiveInference.RedBlueButton import env_utils as sa_env_utils


def two_agent_obs_to_single_agent_obs(env_obs, agent_id):
    """
    Convert TwoAgentRedBlueButton env obs into the SA environment-observation schema.
    """
    if agent_id == 1:
        position = env_obs["agent1_position"]
        on_red = env_obs["agent1_on_red_button"]
        on_blue = env_obs["agent1_on_blue_button"]
    else:
        position = env_obs["agent2_position"]
        on_red = env_obs["agent2_on_red_button"]
        on_blue = env_obs["agent2_on_blue_button"]

    return {
        "position": position,
        "on_red_button": int(on_red),
        "on_blue_button": int(on_blue),
        "red_button_pressed": int(env_obs["red_button_pressed"]),
        "blue_button_pressed": int(env_obs["blue_button_pressed"]),
        "win_lose_neutral": int(env_obs["win_lose_neutral"]),
        "button_just_pressed": env_obs.get("button_just_pressed"),
    }


def env_obs_to_model_obs(env_obs, agent_id, width=3):
    """
    Convert TwoAgentRedBlueButton obs to SA model obs indices for this agent.
    """
    sa_obs = two_agent_obs_to_single_agent_obs(env_obs, agent_id=agent_id)
    return sa_env_utils.env_obs_to_model_obs(sa_obs, width=width)


def get_D_config_from_env(env, agent_id):
    """
    Config for SA D_fn for this agent.
    """
    if agent_id == 1:
        start_xy = env.agent1_start_pos
    else:
        start_xy = env.agent2_start_pos

    start_idx = sa_env_utils.xy_to_index(int(start_xy[0]), int(start_xy[1]), env.width)
    return {
        "agent_start_pos": int(start_idx),
        "button_pos_uncertainty": True,
        "button_state_uncertainty": False,
    }


def model_action_to_env_action(model_action):
    return int(model_action)


def env_action_to_model_action(env_action):
    return int(env_action)


