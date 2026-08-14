"""
Preference model (C) for FullyCollective paradigm.

Preferences are expressed over JOINT observations.
Keep close to SA conventions: prefer win, avoid lose, small shaping for pressed buttons.

NOTE: this package has zero importers anywhere in the repo (confirmed via
repo-wide grep, 2026-08-14) -- dead code, not used by any live run script.
Kept numerically in sync with the live MA_ActiveInference/RedBlueButton/
FullyCollective/C.py anyway, per explicit request that all RedBlueButton
C.py files agree (see ai/02-debug.md, "cross-timestep termination not
tracked in EFE rollout").
"""


def C_fn(observation_indices):
    prefs = {}
    for modality, obs_idx in observation_indices.items():
        if modality == "game_result":
            if obs_idx == 1:  # win
                prefs[modality] = 4.0
            elif obs_idx == 2:  # lose
                prefs[modality] = -40.0
            else:
                prefs[modality] = 0.0
        elif modality == "red_button_state":
            prefs[modality] = 0.5 if obs_idx == 1 else 0.0
        elif modality == "blue_button_state":
            prefs[modality] = 0.2 if obs_idx == 1 else 0.0
        else:
            prefs[modality] = 0.0
    return prefs


