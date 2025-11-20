"""
Functional C (preferences) for RedBlueButton environment.

C encodes the agent's preferences over observations.
Each observation modality has a preference function that assigns a value (utility/reward)
to each possible observation outcome.

DESIGN PRINCIPLE:
- C functions encode preferences (not beliefs)
- They take specific observation indices and return scalar preference values
- Higher values = more preferred outcomes
- Can be positive (attractive), negative (aversive), or zero (neutral)
"""

import numpy as np
from . import model_init


# =============================================================================
# Core C functions: preference for each observation outcome
# =============================================================================

def C_agent_pos(obs_idx):
    """
    Preference for observing agent at different positions.
    
    Returns 0.0 (neutral) for all positions.
    """
    return 0.0


def C_on_red_button(obs_idx):
    """
    Preference for being on the red button.
    Reward should come only from 'button_just_pressed'.
    0 = FALSE (not on button), 1 = TRUE (on button)
    """
    return 0.0


def C_on_blue_button(obs_idx):
    """
    Preference for being on the blue button.
    Reward should come only from 'button_just_pressed'.
    0 = FALSE (not on button), 1 = TRUE (on button)
    """
    return 0.0


def C_red_button_state(obs_idx):
    """
    Preference for red button state.
    0 = not_pressed, 1 = pressed
    
    Small reward for pressed state to guide agent toward pressing buttons.
    Main reward comes from game_result (win).
    """
    if obs_idx == 1:  # pressed
        return 0.5  # Small reward to encourage pressing
    return 0.0


def C_blue_button_state(obs_idx):
    """
    Preference for blue button state.
    0 = not_pressed, 1 = pressed
    
    Small reward for pressed state to guide agent toward pressing buttons.
    Main reward comes from game_result (win).
    """
    if obs_idx == 1:  # pressed
        return 0.5  # Small reward to encourage pressing
    return 0.0


def C_game_result(obs_idx):
    """
    Preference for game result.
    0 = neutral, 1 = win, 2 = lose
    """
    if obs_idx == 1:    # win
        return 1
    elif obs_idx == 2:  # lose
        return -1
    else:               # neutral (idle)
        return 0.0


def C_button_just_pressed(obs_idx):
    """
    Preference for button press events (TRANSITION reward).
    0 = FALSE (no press), 1 = TRUE (just pressed a button)
    
    NOTE: Disabled because this modality cannot be predicted correctly during
    planning (requires previous state). Use button_state rewards instead.
    """
    return 0.0  # Disabled - use game_result and button_state instead


# =============================================================================
# C function registry - maps modality names to functions
# =============================================================================

C_FUNCTIONS = {
    "agent_pos": C_agent_pos,
    "on_red_button": C_on_red_button,
    "on_blue_button": C_on_blue_button,
    "red_button_state": C_red_button_state,
    "blue_button_state": C_blue_button_state,
    "game_result": C_game_result,
    "button_just_pressed": C_button_just_pressed,
}


# =============================================================================
# C_fn: Main interface (analogous to A_fn and B_fn)
# =============================================================================

def C_fn(observation_indices):
    """
    Compute preferences for each observation modality.
    
    Simple version: No context-dependent rewards/penalties.
    Pressed buttons are treated as neutral/ground state.
    Rewards come from:
    - game_result (win/lose)
    - button_just_pressed (transition reward for pressing)
    """
    preferences = {}
    
    # Get preferences from basic functions
    for modality, obs_idx in observation_indices.items():
        if modality in C_FUNCTIONS:
            C_func = C_FUNCTIONS[modality]
            preferences[modality] = C_func(obs_idx)
    
    return preferences


# def C_fn(observation_indices):
#     """
#     Compute preferences for each observation modality.
    
#     Includes context-dependent penalties (e.g., punish staying on pressed button).
#     """
#     preferences = {}
    
#     # Get basic preferences
#     for modality, obs_idx in observation_indices.items():
#         if modality in C_FUNCTIONS:
#             C_func = C_FUNCTIONS[modality]
#             preferences[modality] = C_func(obs_idx)
    
#     # CONTEXT-AWARE REWARDS/PENALTIES:
#     # Punish being on red button if it's already pressed
#     if ('on_red_button' in observation_indices and 
#         'red_button_state' in observation_indices):
#         on_red = observation_indices['on_red_button']  # 0=FALSE, 1=TRUE
#         red_pressed = observation_indices['red_button_state']  # 0=not_pressed, 1=pressed
#         if on_red == 1 and red_pressed == 1:  # On button AND it's pressed
#             preferences['on_red_button'] = -1.0  # Strong penalty for lingering
    
#     # REWARD being on blue button if red is already pressed (guide toward goal)
#     if ('on_blue_button' in observation_indices and 
#         'red_button_state' in observation_indices and
#         'blue_button_state' in observation_indices):
#         on_blue = observation_indices['on_blue_button']
#         red_pressed = observation_indices['red_button_state']
#         blue_pressed = observation_indices['blue_button_state']
        
#         if on_blue == 1 and red_pressed == 1 and blue_pressed == 0:
#             # On blue, red already pressed, blue not yet pressed = GOOD!
#             preferences['on_blue_button'] = 2.0  # Strong reward for being in right place
#         elif on_blue == 1 and blue_pressed == 1:
#             # On blue but already pressed = neutral/slight penalty
#             preferences['on_blue_button'] = -0.5
    
#     return preferences


def get_total_preference(observation_indices):
    prefs = C_fn(observation_indices)
    return sum(prefs.values())


# =============================================================================
# Utility: Build preference vectors for each modality
# =============================================================================

def build_C_vectors():
    C_vectors = {}
    for modality, obs_labels in model_init.observations.items():
        num_obs = len(obs_labels)
        C_func = C_FUNCTIONS[modality]
        C_vec = np.array([C_func(i) for i in range(num_obs)])
        C_vectors[modality] = C_vec
    return C_vectors


# =============================================================================
# Expected utility computation
# =============================================================================

def compute_expected_utility(observation_likelihoods):
    C_vecs = build_C_vectors()
    modality_utilities = {}
    total_utility = 0.0
    for modality, p_obs in observation_likelihoods.items():
        if modality in C_vecs:
            C_vec = C_vecs[modality]
            utility = np.dot(p_obs, C_vec)
            modality_utilities[modality] = utility
            total_utility += utility
    return total_utility, modality_utilities
