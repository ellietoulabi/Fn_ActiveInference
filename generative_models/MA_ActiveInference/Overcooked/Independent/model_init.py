"""
Independent paradigm model init for Overcooked - Cramped Room layout.

Each agent uses a single-agent model and selects only its own action.
The other agent is treated as part of the environment.
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
POT_LOCATIONS = [(2, 0)]
SERVING_LOCATIONS = [(3, 3)]
ONION_DISPENSERS = [(0, 1), (4, 1)]

# Convert to indices
POT_INDICES = [y * GRID_WIDTH + x for x, y in POT_LOCATIONS]
SERVING_INDICES = [y * GRID_WIDTH + x for x, y in SERVING_LOCATIONS]
ONION_DISPENSER_INDICES = [y * GRID_WIDTH + x for x, y in ONION_DISPENSERS]

# -------------------------------------------------
# Primitive actions (per-agent)
# -------------------------------------------------
NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6

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
# Pot states
# -------------------------------------------------
POT_IDLE = 0
POT_COOKING = 1
POT_READY = 2
N_POT_STATES = 3

# -------------------------------------------------
# States (single-agent perspective)
# -------------------------------------------------
states = {
    "agent_pos": list(range(GRID_SIZE)),
    "agent_orientation": list(range(N_DIRECTIONS)),
    "agent_held": list(range(N_HELD_TYPES)),
    "other_agent_pos": list(range(GRID_SIZE)),
    "pot_state": list(range(N_POT_STATES)),
    "soup_delivered": [0, 1],
}

# -------------------------------------------------
# Observations (single-agent perspective)
# -------------------------------------------------
observations = {
    "agent_pos": list(range(GRID_SIZE)),
    "agent_orientation": list(range(N_DIRECTIONS)),
    "agent_held": list(range(N_HELD_TYPES)),
    "other_agent_pos": list(range(GRID_SIZE)),
    "pot_state": list(range(N_POT_STATES)),
    "soup_delivered": [0, 1],
}

observation_state_dependencies = {
    "agent_pos": ["agent_pos"],
    "agent_orientation": ["agent_orientation"],
    "agent_held": ["agent_held"],
    "other_agent_pos": ["other_agent_pos"],
    "pot_state": ["pot_state"],
    "soup_delivered": ["soup_delivered"],
}

state_state_dependencies = {
    "agent_pos": ["agent_pos", "agent_orientation", "other_agent_pos"],
    "agent_orientation": ["agent_orientation"],
    "agent_held": ["agent_pos", "agent_held", "other_agent_pos", "pot_state"],
    "other_agent_pos": ["other_agent_pos"],
    "pot_state": ["agent_pos", "agent_held", "other_agent_pos", "pot_state"],
    "soup_delivered": ["agent_pos", "agent_held", "other_agent_pos"],
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
    return 0

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
