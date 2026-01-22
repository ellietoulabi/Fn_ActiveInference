"""
Prior beliefs (D) for FullyCollective paradigm (JOINT state) - Cramped Room.
"""

import numpy as np
from . import model_init


def build_D(
    agent1_start_pos=None,
    agent2_start_pos=None,
    agent1_start_ori=0,  # NORTH
    agent2_start_ori=0,  # NORTH
):
    """
    Build prior belief distribution for cramped_room layout.
    
    Args:
        agent1_start_pos: Starting position index for agent 1 (default: (1, 2) = 11)
        agent2_start_pos: Starting position index for agent 2 (default: (3, 1) = 8)
        agent1_start_ori: Starting orientation index for agent 1
        agent2_start_ori: Starting orientation index for agent 2
    
    Returns:
        Dictionary mapping state factor names to prior probability arrays
    """
    S = model_init.GRID_SIZE
    
    # Default start positions for cramped_room
    if agent1_start_pos is None:
        agent1_start_pos = model_init.xy_to_index(1, 2)  # (1, 2)
    if agent2_start_pos is None:
        agent2_start_pos = model_init.xy_to_index(3, 1)  # (3, 1)
    
    D = {}

    # Agent positions (deterministic at start)
    D["agent1_pos"] = np.zeros(S)
    if 0 <= agent1_start_pos < S:
        D["agent1_pos"][agent1_start_pos] = 1.0
    else:
        D["agent1_pos"][0] = 1.0

    D["agent2_pos"] = np.zeros(S)
    if 0 <= agent2_start_pos < S:
        D["agent2_pos"][agent2_start_pos] = 1.0
    else:
        D["agent2_pos"][0] = 1.0

    # Agent orientations (deterministic at start - both facing NORTH)
    D["agent1_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= agent1_start_ori < model_init.N_DIRECTIONS:
        D["agent1_orientation"][agent1_start_ori] = 1.0
    else:
        D["agent1_orientation"][0] = 1.0  # Default to NORTH

    D["agent2_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= agent2_start_ori < model_init.N_DIRECTIONS:
        D["agent2_orientation"][agent2_start_ori] = 1.0
    else:
        D["agent2_orientation"][0] = 1.0  # Default to NORTH

    # Held objects (start with nothing)
    D["agent1_held"] = np.zeros(model_init.N_HELD_TYPES)
    D["agent1_held"][model_init.HELD_NONE] = 1.0

    D["agent2_held"] = np.zeros(model_init.N_HELD_TYPES)
    D["agent2_held"][model_init.HELD_NONE] = 1.0

    # Pot state (starts idle)
    D["pot_state"] = np.zeros(model_init.N_POT_STATES)
    D["pot_state"][model_init.POT_IDLE] = 1.0

    # Soup delivery (starts at 0)
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
        agent1_start_pos=config.get("agent1_start_pos", None),
        agent2_start_pos=config.get("agent2_start_pos", None),
        agent1_start_ori=config.get("agent1_start_ori", 0),
        agent2_start_ori=config.get("agent2_start_ori", 0),
    )
