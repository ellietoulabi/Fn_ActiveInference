# C.py
"""
Preference model (C) for IndividuallyCollective paradigm — Cramped Room (sparse reward).
Counter occupancy modalities (ctr_*_obs) get 0 preference by default.

With a short planning horizon, delivery-only preferences give almost no myopic gradient
toward filling the pot (B links deposits to ck_delivered over many steps). Small monotone
rewards on pot_state_obs (POT_0..POT_3 in model_init) supply that gradient without
competing with the terminal soup delivery preference.
"""

from . import model_init

SOUP_DELIVERY_REWARD = 20.0
SOUP_DELIVERED_OBS_IDX = 1  # index 1 = "delivered this step" in soup_delivered_obs

# Monotone in fill level; keep << SOUP_DELIVERY_REWARD.
_POT_STATE_UTILITY = (0.0, 0.5, 1.0, 1.5)


def C_fn(observation_indices: dict[str, int]) -> dict[str, float]:
    assert len(_POT_STATE_UTILITY) == model_init.N_POT_STATES
    prefs: dict[str, float] = {}
    for modality, obs_idx in observation_indices.items():
        if modality == "soup_delivered_obs" and int(obs_idx) == SOUP_DELIVERED_OBS_IDX:
            prefs[modality] = SOUP_DELIVERY_REWARD
        elif modality == "pot_state_obs":
            k = int(obs_idx)
            prefs[modality] = _POT_STATE_UTILITY[k] if 0 <= k < len(_POT_STATE_UTILITY) else 0.0
        else:
            prefs[modality] = 0.0
    return prefs