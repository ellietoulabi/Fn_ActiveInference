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
    
    Parameters
    ----------
    obs_idx : int
        Observed agent position (0 to S-1)
    
    Returns
    -------
    preference : float
        Preference value for this observation (0.0 = neutral)
    """
    # Neutral about agent position - no preference
    return 0.0


def C_on_red_button(obs_idx):
    """
    Preference for being on red button.
    
    Parameters
    ----------
    obs_idx : int
        0 = FALSE (not on button), 1 = TRUE (on button)
    
    Returns
    -------
    preference : float
        Preference value
    """
    # Slight preference for being on red button
    return 0.1 if obs_idx == 1 else 0.0


def C_on_blue_button(obs_idx):
    """
    Preference for being on blue button.
    
    Parameters
    ----------
    obs_idx : int
        0 = FALSE (not on button), 1 = TRUE (on button)
    
    Returns
    -------
    preference : float
        Preference value
    """
    # Slight preference for being on blue button
    return 0.1 if obs_idx == 1 else 0.0


def C_red_button_state(obs_idx):
    """
    Preference for red button state.
    
    Parameters
    ----------
    obs_idx : int
        0 = not_pressed, 1 = pressed
    
    Returns
    -------
    preference : float
        Preference value
    """
    if obs_idx == 1:  # pressed
        return 0.5    # Prefer button being pressed
    else:             # not pressed (idle)
        return -0.01  # Small penalty for not having pressed it yet


def C_blue_button_state(obs_idx):
    """
    Preference for blue button state.
    
    Parameters
    ----------
    obs_idx : int
        0 = not_pressed, 1 = pressed
    
    Returns
    -------
    preference : float
        Preference value
    """
    if obs_idx == 1:  # pressed
        return 0.5    # Prefer button being pressed
    else:             # not pressed (idle)
        return -0.01  # Small penalty for not having pressed it yet


def C_game_result(obs_idx):
    """
    Preference for game result.
    
    Parameters
    ----------
    obs_idx : int
        0 = neutral, 1 = win, 2 = lose
    
    Returns
    -------
    preference : float
        Preference value (can be negative for aversive outcomes)
    """
    if obs_idx == 1:    # win
        return 5.0      # Strongly prefer winning
    elif obs_idx == 2:  # lose
        return -5.0     # Strongly avoid losing
    else:               # neutral (idle state)
        return -0.01    # Small penalty for staying idle (encourages action)


def C_button_just_pressed(obs_idx):
    """
    Preference for button press events.
    
    Parameters
    ----------
    obs_idx : int
        0 = FALSE (no press), 1 = TRUE (just pressed)
    
    Returns
    -------
    preference : float
        Preference value
    """
    # Slight preference for NOT pressing (encourage deliberate actions)
    return 0.1 if obs_idx == 1 else 0.0


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
    Get preference values for ALL modalities given specific observations.
    
    This is the main C function analogous to A_fn and B_fn.
    Takes specific observation indices and returns preference values.
    
    Parameters
    ----------
    observation_indices : dict
        Dictionary mapping modality names to specific observation indices.
        Example: {
            "agent_pos": 4,
            "on_red_button": 1,  # TRUE
            "on_blue_button": 0,  # FALSE
            "red_button_state": 1,  # pressed
            "blue_button_state": 0,  # not pressed
            "game_result": 1,  # win
            "button_just_pressed": 0,  # FALSE
        }
    
    Returns
    -------
    preferences : dict
        Dictionary mapping modality names to preference values (scalars).
    """
    preferences = {}
    
    for modality, obs_idx in observation_indices.items():
        if modality in C_FUNCTIONS:
            C_func = C_FUNCTIONS[modality]
            preferences[modality] = C_func(obs_idx)
    
    return preferences


def get_total_preference(observation_indices):
    """
    Get total preference (sum across all modalities).
    
    Parameters
    ----------
    observation_indices : dict
        Dictionary mapping modality names to observation indices
    
    Returns
    -------
    total_pref : float
        Sum of preferences across all modalities
    """
    prefs = C_fn(observation_indices)
    return sum(prefs.values())


# =============================================================================
# Utility: Build preference vectors for each modality
# =============================================================================

def build_C_vectors():
    """
    Build preference vectors for all modalities.
    
    Returns a dict where each modality maps to a vector of preferences,
    one value per possible observation outcome.
    
    Returns
    -------
    C_vectors : dict
        Dictionary mapping modality names to preference vectors.
        Each vector has length = number of possible observations for that modality.
    """
    C_vectors = {}
    
    for modality, obs_labels in model_init.observations.items():
        num_obs = len(obs_labels)
        C_func = C_FUNCTIONS[modality]
        
        # Build vector of preferences for all possible observations
        C_vec = np.array([C_func(i) for i in range(num_obs)])
        C_vectors[modality] = C_vec
    
    return C_vectors


# =============================================================================
# Expected utility computation
# =============================================================================

def compute_expected_utility(observation_likelihoods):
    """
    Compute expected utility given observation likelihood distributions.
    
    This computes: EU = Σ_modalities Σ_o p(o) * C(o)
    
    Parameters
    ----------
    observation_likelihoods : dict
        Dictionary mapping modality names to observation distributions.
        Each distribution is p(o) over possible observations.
        (e.g., output from A_fn or predict_obs_from_beliefs)
    
    Returns
    -------
    expected_utility : float
        Expected utility across all modalities
    modality_utilities : dict
        Expected utility per modality (for debugging/analysis)
    """
    C_vecs = build_C_vectors()
    
    modality_utilities = {}
    total_utility = 0.0
    
    for modality, p_obs in observation_likelihoods.items():
        if modality in C_vecs:
            # Expected utility for this modality: Σ_o p(o) * C(o)
            C_vec = C_vecs[modality]
            utility = np.dot(p_obs, C_vec)
            modality_utilities[modality] = utility
            total_utility += utility
    
    return total_utility, modality_utilities
