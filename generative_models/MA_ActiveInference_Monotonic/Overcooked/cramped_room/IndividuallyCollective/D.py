"""
Prior beliefs (D) for IndividuallyCollective paradigm (JOINT state) - Cramped Room.

This mirrors the FullyCollective prior over the JOINT state, but is implemented
locally in the IndividuallyCollective folder.
"""

import numpy as np
from . import model_init


def build_D(
    self_start_pos=None,
    other_start_pos=None,
    self_start_ori=0,  # NORTH
    other_start_ori=0,  # NORTH
):
    """
    Build prior belief distribution for cramped_room layout.
    
    Args:
        self_start_pos: Starting position index for self agent (default: (1, 2) = 11)
        other_start_pos: Starting position index for other agent (default: (3, 1) = 8)
        self_start_ori: Starting orientation index for self agent
        other_start_ori: Starting orientation index for other agent
    
    Returns:
        Dictionary mapping state factor names to prior probability arrays
    """
    S = model_init.GRID_SIZE
    
    # Default start positions for cramped_room
    if self_start_pos is None:
        self_start_pos = model_init.xy_to_index(1, 2)  # (1, 2)
    if other_start_pos is None:
        other_start_pos = model_init.xy_to_index(3, 1)  # (3, 1)
    
    D = {}

    # Agent positions (deterministic at start)
    D["self_pos"] = np.zeros(S)
    if 0 <= self_start_pos < S:
        D["self_pos"][self_start_pos] = 1.0
    else:
        D["self_pos"][0] = 1.0

    D["other_pos"] = np.zeros(S)
    if 0 <= other_start_pos < S:
        D["other_pos"][other_start_pos] = 1.0
    else:
        D["other_pos"][0] = 1.0

    # Agent orientations (deterministic at start - both facing NORTH)
    D["self_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= self_start_ori < model_init.N_DIRECTIONS:
        D["self_orientation"][self_start_ori] = 1.0
    else:
        D["self_orientation"][0] = 1.0  # Default to NORTH

    D["other_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= other_start_ori < model_init.N_DIRECTIONS:
        D["other_orientation"][other_start_ori] = 1.0
    else:
        D["other_orientation"][0] = 1.0  # Default to NORTH

    # Held objects (start with nothing)
    D["self_held"] = np.zeros(model_init.N_HELD_TYPES)
    D["self_held"][model_init.HELD_NONE] = 1.0

    D["other_held"] = np.zeros(model_init.N_HELD_TYPES)
    D["other_held"][model_init.HELD_NONE] = 1.0

    # Pot state (starts idle)
    D["pot_state"] = np.zeros(model_init.N_POT_STATES)
    D["pot_state"][model_init.POT_IDLE] = 1.0

    # Checkbox memory starts with all milestones unchecked (0)
    for ck_name in ["ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"]:
        D[ck_name] = np.zeros(2)
        D[ck_name][0] = 1.0

    # Optional explicit soup_delivered event-state (starts at 0)
    D["soup_delivered"] = np.array([1.0, 0.0])

    return D


def D_fn(config=None):
    """
    Prior beliefs function.
    
    Args:
        config: Dictionary with configuration (start positions, orientations)
    
    Returns:
        Dictionary mapping state factor names to prior probability arrays
    """
    if config is None:
        return build_D()
    
    return build_D(
        self_start_pos=config.get("self_start_pos", None),
        other_start_pos=config.get("other_start_pos", None),
        self_start_ori=config.get("self_start_ori", 0),
        other_start_ori=config.get("other_start_ori", 0),
    )

