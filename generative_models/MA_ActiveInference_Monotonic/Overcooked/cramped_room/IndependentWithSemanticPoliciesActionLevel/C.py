# C.py
"""
Preference model (C) for Independent paradigm — Cramped Room (sparse reward).
"""

from . import model_init

SOUP_DELIVERY_REWARD = 20.0
SOUP_DELIVERED_OBS_IDX = 1  # index 1 = "delivered this step" in soup_delivered_obs



def C_fn(observation_indices: dict[str, int]) -> dict[str, float]:
    prefs: dict[str, float] = {}
    for modality, obs_idx in observation_indices.items():
        if modality == "soup_delivered_obs" and int(obs_idx) == SOUP_DELIVERED_OBS_IDX:
            prefs[modality] = SOUP_DELIVERY_REWARD
        else:
            prefs[modality] = 0.0
    return prefs