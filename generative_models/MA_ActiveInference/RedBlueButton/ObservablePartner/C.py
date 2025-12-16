"""
Preferences (C) for ObservablePartner variant.
"""

from . import model_init


def C_fn(observation_indices):
    """
    Return per-modality scalar preferences.
    Keep this close to SA defaults: only game_result matters by default.
    """
    prefs = {}
    for modality, obs_idx in observation_indices.items():
        if modality == "game_result":
            if obs_idx == 1:
                prefs[modality] = 1.0
            elif obs_idx == 2:
                prefs[modality] = -1.0
            else:
                prefs[modality] = 0.0
        else:
            prefs[modality] = 0.0
    return prefs


