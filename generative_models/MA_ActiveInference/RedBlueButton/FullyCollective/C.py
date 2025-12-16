"""
Preference model (C) for FullyCollective paradigm.

Preferences are expressed over JOINT observations.
Keep close to SA conventions: prefer win, avoid lose, small shaping for pressed buttons.
"""


def C_fn(observation_indices):
    prefs = {}
    for modality, obs_idx in observation_indices.items():
        if modality == "game_result":
            # 0=neutral, 1=win, 2=lose
            if obs_idx == 1:
                prefs[modality] = 1.0
            elif obs_idx == 2:
                prefs[modality] = -1.0
            else:
                prefs[modality] = 0.0
        else:
            # No shaping on other modalities
            prefs[modality] = 0.0
    return prefs


