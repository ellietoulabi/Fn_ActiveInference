"""
Observation model (A) for FullyCollective paradigm (JOINT state) - Cramped Room.
"""

import numpy as np
from . import model_init

# Keep small observation noise for positions
A_NOISE_LEVEL = 0.01


def _noisy_pos_obs(pos_idx, num_obs, noise_level=A_NOISE_LEVEL):
    """Add small noise to position observations."""
    p = np.full(num_obs, noise_level / max(1, (num_obs - 1)), dtype=float)
    if 0 <= pos_idx < num_obs:
        p[pos_idx] = 1.0 - noise_level
    return p


def A_fn(state_indices):
    """
    Compute observation likelihoods for the full joint observation.
    
    Args:
        state_indices: Dictionary mapping state factor names to their indices
    
    Returns:
        Dictionary mapping observation modality names to likelihood arrays
    """
    a1_pos = int(state_indices["agent1_pos"])
    a2_pos = int(state_indices["agent2_pos"])
    a1_ori = int(state_indices["agent1_orientation"])
    a2_ori = int(state_indices["agent2_orientation"])
    a1_held = int(state_indices["agent1_held"])
    a2_held = int(state_indices["agent2_held"])
    pot_state = int(state_indices.get("pot_state", 0))
    soup_delivered = int(state_indices.get("soup_delivered", 0))

    S = model_init.GRID_SIZE
    obs = {}

    # Position observations with noise
    obs["agent1_pos"] = _noisy_pos_obs(a1_pos, S)
    obs["agent2_pos"] = _noisy_pos_obs(a2_pos, S)

    # Orientation observations (deterministic)
    obs["agent1_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= a1_ori < model_init.N_DIRECTIONS:
        obs["agent1_orientation"][a1_ori] = 1.0
    
    obs["agent2_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= a2_ori < model_init.N_DIRECTIONS:
        obs["agent2_orientation"][a2_ori] = 1.0

    # Held object observations (deterministic)
    obs["agent1_held"] = np.zeros(model_init.N_HELD_TYPES)
    if 0 <= a1_held < model_init.N_HELD_TYPES:
        obs["agent1_held"][a1_held] = 1.0
    
    obs["agent2_held"] = np.zeros(model_init.N_HELD_TYPES)
    if 0 <= a2_held < model_init.N_HELD_TYPES:
        obs["agent2_held"][a2_held] = 1.0

    # Pot state observation (deterministic)
    obs["pot_state"] = np.zeros(model_init.N_POT_STATES)
    if 0 <= pot_state < model_init.N_POT_STATES:
        obs["pot_state"][pot_state] = 1.0

    # Soup delivery observation (deterministic)
    obs["soup_delivered"] = np.zeros(2)
    obs["soup_delivered"][soup_delivered] = 1.0

    return obs
