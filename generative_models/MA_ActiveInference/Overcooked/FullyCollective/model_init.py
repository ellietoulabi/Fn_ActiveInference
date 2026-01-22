"""
FullyCollective paradigm model init for Overcooked - Cramped Room layout.

This is a JOINT (centralised) model for the cramped_room layout:
- Grid: 5x4 (20 cells)
- 1 pot at (2, 0)
- Serving location at (3, 3)
- Onion dispensers at (0, 1) and (4, 1)
- No tomato dispensers (onion-only recipes)

Hidden state includes both agent positions, orientations, held objects, and pot state.
Actions are JOINT actions (a1, a2) encoded as a single integer in [0, 35].
"""

# -------------------------------------------------
# Grid size (cramped_room specific)
# -------------------------------------------------
GRID_WIDTH = 5
GRID_HEIGHT = 4
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT  # 20 cells

# -------------------------------------------------
# Layout-specific locations
# -------------------------------------------------
POT_LOCATIONS = [(2, 0)]  # Single pot
SERVING_LOCATIONS = [(3, 3)]  # Delivery counter
ONION_DISPENSERS = [(0, 1), (4, 1)]
TOMATO_DISPENSERS = []  # None in cramped_room

# Convert to indices
POT_INDICES = [y * GRID_WIDTH + x for x, y in POT_LOCATIONS]
SERVING_INDICES = [y * GRID_WIDTH + x for x, y in SERVING_LOCATIONS]
ONION_DISPENSER_INDICES = [y * GRID_WIDTH + x for x, y in ONION_DISPENSERS]

# -------------------------------------------------
# Primitive actions (per-agent)
# -------------------------------------------------
NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
N_JOINT_ACTIONS = N_ACTIONS * N_ACTIONS  # 36

# -------------------------------------------------
# Directions
# -------------------------------------------------
DIR_NORTH = (0, -1)
DIR_SOUTH = (0, 1)
DIR_EAST = (1, 0)
DIR_WEST = (-1, 0)
DIRECTIONS = [DIR_NORTH, DIR_SOUTH, DIR_EAST, DIR_WEST]
N_DIRECTIONS = 4

# -------------------------------------------------
# Held object types (cramped_room only has onions, no tomatoes)
# -------------------------------------------------
HELD_NONE = 0
HELD_ONION = 1
HELD_DISH = 2
HELD_SOUP = 3
N_HELD_TYPES = 4

# -------------------------------------------------
# Pot states (for the single pot)
# -------------------------------------------------
POT_IDLE = 0      # Empty pot
POT_COOKING = 1   # Pot has ingredients and is cooking
POT_READY = 2     # Soup is ready to be picked up
N_POT_STATES = 3

# -------------------------------------------------
# States (JOINT)
# -------------------------------------------------
states = {
    "agent1_pos": list(range(GRID_SIZE)),  # Flattened position index (0-19)
    "agent2_pos": list(range(GRID_SIZE)),
    "agent1_orientation": list(range(N_DIRECTIONS)),  # 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST
    "agent2_orientation": list(range(N_DIRECTIONS)),
    "agent1_held": list(range(N_HELD_TYPES)),  # 0=none, 1=onion, 2=dish, 3=soup
    "agent2_held": list(range(N_HELD_TYPES)),
    "pot_state": list(range(N_POT_STATES)),  # 0=idle, 1=cooking, 2=ready
    "soup_delivered": [0, 1],  # 0=no, 1=yes (resets each step)
}

# -------------------------------------------------
# Observations (full joint observation)
# -------------------------------------------------
observations = {
    "agent1_pos": list(range(GRID_SIZE)),
    "agent2_pos": list(range(GRID_SIZE)),
    "agent1_orientation": list(range(N_DIRECTIONS)),
    "agent2_orientation": list(range(N_DIRECTIONS)),
    "agent1_held": list(range(N_HELD_TYPES)),
    "agent2_held": list(range(N_HELD_TYPES)),
    "pot_state": list(range(N_POT_STATES)),
    "soup_delivered": [0, 1],
}

observation_state_dependencies = {
    "agent1_pos": ["agent1_pos"],
    "agent2_pos": ["agent2_pos"],
    "agent1_orientation": ["agent1_orientation"],
    "agent2_orientation": ["agent2_orientation"],
    "agent1_held": ["agent1_held"],
    "agent2_held": ["agent2_held"],
    "pot_state": ["pot_state"],
    "soup_delivered": ["soup_delivered"],
}

state_state_dependencies = {
    # Positions depend on previous positions, orientations, and joint action
    "agent1_pos": ["agent1_pos", "agent1_orientation", "agent2_pos"],
    "agent2_pos": ["agent2_pos", "agent2_orientation", "agent1_pos"],
    # Orientations change when moving
    "agent1_orientation": ["agent1_orientation"],
    "agent2_orientation": ["agent2_orientation"],
    # Held objects depend on positions, pot state, and interactions
    "agent1_held": ["agent1_pos", "agent1_held", "agent2_pos", "pot_state"],
    "agent2_held": ["agent2_pos", "agent2_held", "agent1_pos", "pot_state"],
    # Pot state depends on agent positions, held objects, and interactions
    "pot_state": ["agent1_pos", "agent2_pos", "agent1_held", "agent2_held", "pot_state"],
    # Soup delivery depends on agent positions, held objects, and serving location
    "soup_delivered": ["agent1_pos", "agent2_pos", "agent1_held", "agent2_held"],
}

# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def xy_to_index(x, y, width=GRID_WIDTH):
    """Convert (x, y) coordinates to flattened index."""
    return y * width + x

def index_to_xy(index, width=GRID_WIDTH):
    """Convert flattened index to (x, y) coordinates."""
    y = index // width
    x = index % width
    return x, y

def direction_to_index(direction):
    """Convert direction tuple to index."""
    if direction == DIR_NORTH:
        return 0
    elif direction == DIR_SOUTH:
        return 1
    elif direction == DIR_EAST:
        return 2
    elif direction == DIR_WEST:
        return 3
    return 0  # Default to NORTH

def index_to_direction(idx):
    """Convert index to direction tuple."""
    return DIRECTIONS[idx]

def object_name_to_held_type(obj_name):
    """Convert object name to held type index."""
    if obj_name is None:
        return HELD_NONE
    obj_map = {
        "onion": HELD_ONION,
        "dish": HELD_DISH,
        "soup": HELD_SOUP,
    }
    return obj_map.get(obj_name, HELD_NONE)

def held_type_to_object_name(held_type):
    """Convert held type index to object name."""
    type_map = {
        HELD_NONE: None,
        HELD_ONION: "onion",
        HELD_DISH: "dish",
        HELD_SOUP: "soup",
    }
    return type_map.get(held_type, None)

def is_at_location(pos_idx, location_indices):
    """Check if position index is at any of the given location indices."""
    return pos_idx in location_indices

def is_at_pot(pos_idx):
    """Check if position is at the pot."""
    return is_at_location(pos_idx, POT_INDICES)

def is_at_serving(pos_idx):
    """Check if position is at the serving location."""
    return is_at_location(pos_idx, SERVING_INDICES)

def is_at_onion_dispenser(pos_idx):
    """Check if position is at an onion dispenser."""
    return is_at_location(pos_idx, ONION_DISPENSER_INDICES)
