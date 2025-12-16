"""
Observation Model (A) for Multi-Agent RedBlueButton.

A_fn computes observation likelihoods given hidden states.
"""

import numpy as np
from . import model_init


def A_fn(state_indices):
    """
    Compute observation likelihoods given state configuration.
    
    Parameters
    ----------
    state_indices : dict
        Dictionary with state factor indices:
        - 'my_pos': int (0-8)
        - 'other_pos': int (0-8)
        - 'red_button_pos': int (0-8)
        - 'blue_button_pos': int (0-8)
        - 'red_button_state': int (0=not_pressed, 1=pressed)
        - 'blue_button_state': int (0=not_pressed, 1=pressed)
    
    Returns
    -------
    obs_likelihoods : dict
        Dictionary mapping observation modality to likelihood distributions
    """
    my_pos = state_indices['my_pos']
    other_pos = state_indices['other_pos']
    red_button_pos = state_indices['red_button_pos']
    blue_button_pos = state_indices['blue_button_pos']
    red_button_state = state_indices['red_button_state']
    blue_button_state = state_indices['blue_button_state']
    
    S = model_init.S
    
    obs_likelihoods = {}
    
    # My position: deterministic observation
    obs_likelihoods['my_pos'] = np.zeros(S)
    obs_likelihoods['my_pos'][my_pos] = 1.0
    
    # Other agent's position: deterministic observation
    obs_likelihoods['other_pos'] = np.zeros(S)
    obs_likelihoods['other_pos'][other_pos] = 1.0
    
    # Am I on red button?
    obs_likelihoods['my_on_red_button'] = np.zeros(2)
    if my_pos == red_button_pos:
        obs_likelihoods['my_on_red_button'][1] = 1.0  # TRUE
    else:
        obs_likelihoods['my_on_red_button'][0] = 1.0  # FALSE
    
    # Am I on blue button?
    obs_likelihoods['my_on_blue_button'] = np.zeros(2)
    if my_pos == blue_button_pos:
        obs_likelihoods['my_on_blue_button'][1] = 1.0  # TRUE
    else:
        obs_likelihoods['my_on_blue_button'][0] = 1.0  # FALSE
    
    # Button states: directly observed
    obs_likelihoods['red_button_state'] = np.zeros(2)
    obs_likelihoods['red_button_state'][red_button_state] = 1.0
    
    obs_likelihoods['blue_button_state'] = np.zeros(2)
    obs_likelihoods['blue_button_state'][blue_button_state] = 1.0
    
    # Game result: depends on button states
    obs_likelihoods['game_result'] = np.zeros(3)  # neutral, win, lose
    if blue_button_state == 1:  # Blue pressed
        if red_button_state == 1:  # Red was pressed first
            obs_likelihoods['game_result'][1] = 1.0  # WIN
        else:  # Red not pressed
            obs_likelihoods['game_result'][2] = 1.0  # LOSE
    else:
        obs_likelihoods['game_result'][0] = 1.0  # NEUTRAL
    
    # Button just pressed: depends on position and button states
    obs_likelihoods['button_just_pressed'] = np.zeros(2)
    # This is a dynamic observation that depends on action, 
    # but for the model we just check if we're on a button
    on_any_button = (my_pos == red_button_pos) or (my_pos == blue_button_pos)
    if on_any_button:
        obs_likelihoods['button_just_pressed'][1] = 0.5  # Could press
        obs_likelihoods['button_just_pressed'][0] = 0.5
    else:
        obs_likelihoods['button_just_pressed'][0] = 1.0  # FALSE
    
    return obs_likelihoods
