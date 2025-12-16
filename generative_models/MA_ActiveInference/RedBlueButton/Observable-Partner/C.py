"""
Preference Model (C) for Multi-Agent RedBlueButton.

C_fn computes log-preferences over observations (what the agent wants to observe).
"""

import numpy as np
from . import model_init


# =============================================================================
# Individual preference functions per modality
# =============================================================================

def C_my_pos(obs_idx):
    """Preference for my position (neutral)."""
    return 0.0


def C_other_pos(obs_idx):
    """Preference for other agent's position (neutral)."""
    return 0.0


def C_my_on_red_button(obs_idx):
    """Preference for being on red button."""
    return 0.0  # Neutral - reward comes from game_result


def C_my_on_blue_button(obs_idx):
    """Preference for being on blue button."""
    return 0.0  # Neutral - reward comes from game_result


def C_red_button_state(obs_idx):
    """Preference for red button state (0=not pressed, 1=pressed)."""
    # Small shaping reward: helps the agent discover that pressing red is useful.
    return 0.0 if obs_idx == 1 else 0.0


def C_blue_button_state(obs_idx):
    """Preference for blue button state."""
    # Smaller shaping reward than red; pressing blue before red is still strongly
    # discouraged by the large lose penalty via game_result.
    return 0.0 if obs_idx == 1 else 0.0


def C_game_result(obs_idx):
    """Preference for game result (0=neutral, 1=win, 2=lose)."""
    if obs_idx == 1:    # win
        return 1.0
    elif obs_idx == 2:  # lose
        return -1.0
    else:               # neutral
        return 0.0


def C_button_just_pressed(obs_idx):
    """Preference for button press events."""
    return 0.0  # Disabled - use game_result instead


# =============================================================================
# C function registry
# =============================================================================

C_FUNCTIONS = {
    "my_pos": C_my_pos,
    "other_pos": C_other_pos,
    "my_on_red_button": C_my_on_red_button,
    "my_on_blue_button": C_my_on_blue_button,
    "red_button_state": C_red_button_state,
    "blue_button_state": C_blue_button_state,
    "game_result": C_game_result,
    "button_just_pressed": C_button_just_pressed,
}


# =============================================================================
# C_fn: Main interface
# =============================================================================

def C_fn(observation_indices):
    """
    Compute preferences for each observation modality.
    
    Parameters
    ----------
    observation_indices : dict
        Dictionary mapping modality names to observation indices
    
    Returns
    -------
    preferences : dict
        Dictionary mapping modality names to preference values
    """
    preferences = {}
    
    for modality, obs_idx in observation_indices.items():
        if modality in C_FUNCTIONS:
            C_func = C_FUNCTIONS[modality]
            preferences[modality] = C_func(obs_idx)
    
    return preferences


def build_C_vectors():
    """Build preference vectors for each modality."""
    C_vectors = {}
    for modality, obs_labels in model_init.observations.items():
        num_obs = len(obs_labels)
        if modality in C_FUNCTIONS:
            C_func = C_FUNCTIONS[modality]
            C_vec = np.array([C_func(i) for i in range(num_obs)])
            C_vectors[modality] = C_vec
    return C_vectors
