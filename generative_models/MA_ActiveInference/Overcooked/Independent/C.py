"""
Preference model (C) for Independent paradigm - Cramped Room.
"""

from . import model_init


def C_fn(observation_indices):
    """
    Preference model: Preferences over observations.
    
    Only uses sparse reward (soup delivery) - no preferences for intermediate states.
    Similar to RedBlueButton: only win/lose preferences, no intermediate shaping.
    
    Args:
        observation_indices: Dictionary mapping observation modality names to their indices
    
    Returns:
        Dictionary mapping observation modality names to preference values
    """
    prefs = {}
    
    for modality, obs_idx in observation_indices.items():
        if modality == "soup_delivered":
            # Only preference: soup delivery (sparse reward = win)
            if obs_idx == 1:  # soup delivered (win)
                prefs[modality] = 1.0
            else:  # no delivery (neutral/lose)
                prefs[modality] = -1.0
        else:
            # No preferences for intermediate states (no shaping)
            prefs[modality] = 0.0
    
    return prefs
