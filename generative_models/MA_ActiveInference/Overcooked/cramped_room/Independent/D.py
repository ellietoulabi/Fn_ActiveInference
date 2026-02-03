"""
Prior beliefs (D) for Independent paradigm (single-agent perspective) - Cramped Room.

Layout: 5x4 grid (20 cells total)
  XXPXX
  O1  O
  X  2X
  XDXSX

Key locations (from model_init):
  - Pot: (2, 0) = index 2
  - Serving: (3, 3) = index 18
  - Onion dispensers: (0, 1) = index 5, (4, 1) = index 9
  - Dish dispenser: (1, 3) = index 16
  - Agent 1 start: (1, 1) = index 6 (marked as "1" in layout)
  - Agent 2 start: (3, 2) = index 13 (marked as "2" in layout)
"""

import numpy as np
from . import model_init


def build_D(
    agent_start_pos=None,
    agent_start_ori=0,  # NORTH
    other_agent_start_pos=None,
):

    S = model_init.GRID_SIZE
    
    if agent_start_pos is None:
        agent_start_pos = model_init.xy_to_index(1, 1)  # Agent 1 at (1,1) = index 6
    if other_agent_start_pos is None:
        other_agent_start_pos = model_init.xy_to_index(3, 2)  # Agent 2 at (3,2) = index 13
    
    D = {}

    D["agent_pos"] = np.zeros(S)
    if 0 <= agent_start_pos < S:
        D["agent_pos"][agent_start_pos] = 1.0
    else:
        D["agent_pos"][0] = 1.0

    D["agent_orientation"] = np.zeros(model_init.N_DIRECTIONS)
    if 0 <= agent_start_ori < model_init.N_DIRECTIONS:
        D["agent_orientation"][agent_start_ori] = 1.0
    else:
        D["agent_orientation"][0] = 1.0

    D["agent_held"] = np.zeros(model_init.N_HELD_TYPES)
    D["agent_held"][model_init.HELD_NONE] = 1.0

    D["other_agent_pos"] = np.zeros(S)
    if 0 <= other_agent_start_pos < S:
        D["other_agent_pos"][other_agent_start_pos] = 1.0
    else:
        D["other_agent_pos"][0] = 1.0

    D["pot_state"] = np.zeros(model_init.N_POT_STATES)
    D["pot_state"][model_init.POT_0] = 1.0

    # soup_delivered is observation-only; no prior in D

    return D


def D_fn(config=None):
 
    if config is None:
        return build_D()
    
    return build_D(
        agent_start_pos=config.get("agent_start_pos", None),
        agent_start_ori=config.get("agent_start_ori", 0),
        other_agent_start_pos=config.get("other_agent_start_pos", None),
    )
