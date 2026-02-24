"""
C.py

Preference model (C) for Independent Monotonic Overcooked — Cramped Room (Stage 1, sparse).

Interface expected by agents/ActiveInference/control.py:
  C_fn(observation_indices) -> dict[modality_name, float]
  where observation_indices is a dict modality -> observed index (int).
  Returns scalar preference (utility) per modality; control uses prefs.get(modality, 0.0).

Observation modalities for this model (see model_init.observations):
  - agent_pos_obs, agent_orientation_obs, agent_held_obs, pot_state_obs, soup_delivered_obs

Sparse objective: strongly prefer a soup delivery event:
  - soup_delivered_obs == 1  → +20 utility
  - all other outcomes and modalities → 0 utility

Note: for robustness we also accept legacy modality name "soup_delivered".
"""

from . import model_init  # noqa: F401

SOUP_DELIVERY_REWARD = 20.0


def C_fn(observation_indices):
    """
    Return preference (utility) per modality for the given observation indices.

    Args
    ----
    observation_indices : dict[str, int]
        Mapping modality name -> observed outcome index (int).
        Control typically calls this with a single modality at a time, e.g.
        {"soup_delivered_obs": 1}.

    Returns
    -------
    prefs : dict[str, float]
        Mapping from each key in observation_indices to a scalar preference value.
    """
    prefs: dict[str, float] = {}
    for modality, obs_idx in observation_indices.items():
        if modality in ("soup_delivered_obs", "soup_delivered"):
            prefs[modality] = SOUP_DELIVERY_REWARD if int(obs_idx) == 1 else 0.0
        else:
            prefs[modality] = 0.0
    return prefs

