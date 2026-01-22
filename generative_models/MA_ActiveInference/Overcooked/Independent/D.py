"""
Prior beliefs (D) for Independent paradigm (single-agent perspective) - Cramped Room.
"""

import numpy as np
from . import model_init


def build_D(
    agent_start_pos=None,
    agent_start_ori=0,  # NORTH
    other_agent_start_pos=None,
):
    """
    Build prior belief distribution for cramped_room layout.
    
    Args:
        agent_start_pos: Starting position index for this agent
        agent_start_ori: Starting orientation index for this agent
        other_agent_start_pos: Starting position index for other agent
    
    Returns:
        Dictionary mapping state factor names to prior probability arrays
    """
    S = model_init.GRID_SIZE
    
    # Default start positions for cramped_room
    if agent_start_pos is None:
        agent_start_pos = model_init.xy_to_index(1, 2)  # Agent 1 default
    if other_agent_start_pos is None:
        other_agent_start_pos = model_init.xy_to_index(3, 1)  # Agent 2 default
    
    D = {}

    # Agent position
    D["agent_pos"] = np.zeros(S)
    if 0 <= agent_start_pos < S:
        D["agent_pos"][agent_start_pos] = 1.0
    else:
        D["agent_pos"][0] = 1.0

    # Agent orientation
    D["agent_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= agent_start_ori < model_init.N_DIRECTIONS:
        D["agent_orientation"][agent_start_ori] = 1.0
    else:
        D["agent_orientation"][0] = 1.0

    # Agent held (start with nothing)
    D["agent_held"] = np.zeros(model_init.N_HELD_TYPES)
    D["agent_held"][model_init.HELD_NONE] = 1.0

    # Other agent position
    D["other_agent_pos"] = np.zeros(S)
    if 0 <= other_agent_start_pos < S:
        D["other_agent_pos"][other_agent_start_pos] = 1.0
    else:
        D["other_agent_pos"][0] = 1.0

    # Pot state (starts idle)
    D["pot_state"] = np.zeros(model_init.N_POT_STATES)
    D["pot_state"][model_init.POT_IDLE] = 1.0

    # Soup delivery
    D["soup_delivered"] = np.array([1.0, 0.0])

    return D


def D_fn(config=None):
    """
    Prior beliefs function.
    
    Args:
        config: Dictionary with configuration
    
    Returns:
        Dictionary mapping state factor names to prior probability arrays
    """
    if config is None:
        return build_D()
    
    return build_D(
        agent_start_pos=config.get("agent_start_pos", None),
        agent_start_ori=config.get("agent_start_ori", 0),
        other_agent_start_pos=config.get("other_agent_start_pos", None),
    )
