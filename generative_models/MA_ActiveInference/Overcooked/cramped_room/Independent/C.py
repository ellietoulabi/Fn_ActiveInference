"""
C.py

Preference model (C) for Independent paradigm — Cramped Room (Stage 1, sparse).

Interface expected by agents/ActiveInference/control.py:
  C_fn(observation_indices) -> dict[modality_name, float]
  where observation_indices is a dict modality -> observed index (int).
  Returns scalar preference (utility) per modality; control uses prefs.get(modality, 0.0).

Sparse objective: prefer soup_delivered = 1 (delivery event → +20). All other modalities 0.
"""

from . import model_init

SOUP_DELIVERY_REWARD = 20.0


def C_fn(observation_indices):
    """
    Return preference (utility) per modality for the given observation indices.

    Args:
        observation_indices: dict mapping modality name -> observed outcome index (int).
            Control calls this with a single modality at a time, e.g. {"soup_delivered": 1}.

    Returns:
        dict mapping each key in observation_indices to a scalar preference value.
    """
    prefs = {}
    for modality, obs_idx in observation_indices.items():
        if modality == "soup_delivered":
            prefs[modality] = SOUP_DELIVERY_REWARD if (obs_idx == 1) else 0.0
        else:
            prefs[modality] = 0.0
    return prefs
