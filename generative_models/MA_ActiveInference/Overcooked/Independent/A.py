"""
Observation model (A) for Independent paradigm (single-agent perspective) - Cramped Room.
"""

import numpy as np
from . import model_init

A_NOISE_LEVEL = 0.01


def _noisy_pos_obs(pos_idx, num_obs, noise_level=A_NOISE_LEVEL):
    """Add small noise to position observations."""
    p = np.full(num_obs, noise_level / max(1, (num_obs - 1)), dtype=float)
    if 0 <= pos_idx < num_obs:
        p[pos_idx] = 1.0 - noise_level
    return p


def A_fn(state_indices):
    """
    Compute observation likelihoods from single-agent perspective.
    
    Args:
        state_indices: Dictionary mapping state factor names to their indices
    
    Returns:
        Dictionary mapping observation modality names to likelihood arrays
    """
    agent_pos = int(state_indices["agent_pos"])
    agent_ori = int(state_indices["agent_orientation"])
    agent_held = int(state_indices["agent_held"])
    other_pos = int(state_indices["other_agent_pos"])
    pot_state = int(state_indices.get("pot_state", 0))
    soup_delivered = int(state_indices.get("soup_delivered", 0))

    S = model_init.GRID_SIZE
    obs = {}

    # Position observations with noise
    obs["agent_pos"] = _noisy_pos_obs(agent_pos, S)
    obs["other_agent_pos"] = _noisy_pos_obs(other_pos, S)

    # Orientation observation
    obs["agent_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= agent_ori < model_init.N_DIRECTIONS:
        obs["agent_orientation"][agent_ori] = 1.0

    # Held object observation
    obs["agent_held"] = np.zeros(model_init.N_HELD_TYPES)
    if 0 <= agent_held < model_init.N_HELD_TYPES:
        obs["agent_held"][agent_held] = 1.0

    # Pot state observation
    obs["pot_state"] = np.zeros(model_init.N_POT_STATES)
    if 0 <= pot_state < model_init.N_POT_STATES:
        obs["pot_state"][pot_state] = 1.0

    # Soup delivery observation
    obs["soup_delivered"] = np.zeros(2)
    obs["soup_delivered"][soup_delivered] = 1.0

    return obs
