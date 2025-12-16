"""
Observation model (A) for ObservablePartner variant.
"""

import numpy as np
from . import model_init


def A_fn(state_indices):
    """
    Compute observation likelihoods given a specific state configuration.
    Returns dict modality -> p(o|s) vector.
    """
    my_pos = int(state_indices["my_pos"])
    other_pos = int(state_indices["other_pos"])
    red_button_pos = int(state_indices["red_button_pos"])
    blue_button_pos = int(state_indices["blue_button_pos"])
    red_button_state = int(state_indices["red_button_state"])
    blue_button_state = int(state_indices["blue_button_state"])

    S = model_init.S
    obs = {}

    obs["my_pos"] = np.zeros(S)
    obs["my_pos"][my_pos] = 1.0

    obs["other_pos"] = np.zeros(S)
    obs["other_pos"][other_pos] = 1.0

    obs["my_on_red_button"] = np.array([1.0, 0.0])
    if my_pos == red_button_pos:
        obs["my_on_red_button"] = np.array([0.0, 1.0])

    obs["my_on_blue_button"] = np.array([1.0, 0.0])
    if my_pos == blue_button_pos:
        obs["my_on_blue_button"] = np.array([0.0, 1.0])

    obs["red_button_state"] = np.zeros(2)
    obs["red_button_state"][red_button_state] = 1.0

    obs["blue_button_state"] = np.zeros(2)
    obs["blue_button_state"][blue_button_state] = 1.0

    obs["game_result"] = np.zeros(3)
    if blue_button_state == 1:
        obs["game_result"][1 if red_button_state == 1 else 2] = 1.0
    else:
        obs["game_result"][0] = 1.0

    # Dynamic observation; approximate (same as SA-like convention)
    obs["button_just_pressed"] = np.zeros(2)
    on_any_button = (my_pos == red_button_pos) or (my_pos == blue_button_pos)
    if on_any_button:
        obs["button_just_pressed"][0] = 0.5
        obs["button_just_pressed"][1] = 0.5
    else:
        obs["button_just_pressed"][0] = 1.0

    return obs


