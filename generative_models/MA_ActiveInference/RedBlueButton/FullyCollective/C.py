"""
Preference model (C) for FullyCollective paradigm.

Preferences are expressed over JOINT observations.
Keep close to SA conventions: prefer win, avoid lose, small shaping for pressed buttons.

USE_INTERMEDIATE_REWARD: temporary experiment flag (2026-08-13, ai/02-debug.md
"reward-shaping asymmetry" entry). Set False to strip button-press shaping
entirely (win/lose only), isolating how much of FC/IC's advantage over
Independent comes from this shaping term vs. other structural differences
(joint modeling, collision-aware transitions). Default True preserves the
original, previously-reported behavior exactly.
"""

USE_INTERMEDIATE_REWARD = True


def C_fn(observation_indices):
    prefs = {}
    for modality, obs_idx in observation_indices.items():
        if modality == "game_result":
            if obs_idx == 1:  # win
                prefs[modality] = 4.0
            elif obs_idx == 2:  # lose
                prefs[modality] = -4.0
            else:
                prefs[modality] = 0.0
        elif modality == "red_button_state":
            prefs[modality] = (0.5 if obs_idx == 1 else 0.0) if USE_INTERMEDIATE_REWARD else 0.0
        elif modality == "blue_button_state":
            prefs[modality] = (0.2 if obs_idx == 1 else 0.0) if USE_INTERMEDIATE_REWARD else 0.0
        else:
            prefs[modality] = 0.0
    return prefs


