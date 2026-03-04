"""
IndividuallyCollective paradigm model init for Overcooked - Cramped Room layout.

This reimplements the JOINT generative model locally for the
IndividuallyCollective paradigm, mirroring the FullyCollective geometry and
state/observation definitions.
"""

# -------------------------------------------------
# Grid size (cramped_room specific)
# -------------------------------------------------
GRID_WIDTH = 5
GRID_HEIGHT = 4
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT  # 20 cells

# -------------------------------------------------
# Recipe / timing (kept consistent with Independent checkbox model)
# -------------------------------------------------
RECIPE_ONIONS = 3
COOK_TIME = 0

# -------------------------------------------------
# Layout-specific locations
# -------------------------------------------------
POT_LOCATIONS = [(2, 0)]  # Single pot
SERVING_LOCATIONS = [(3, 3)]  # Delivery counter
ONION_DISPENSERS = [(0, 1), (4, 1)]
TOMATO_DISPENSERS = []  # None in cramped_room

"""
Model geometry and state/observation definitions for the IndividuallyCollective
paradigm on cramped_room. Note: movement constraints (which cells are
walkable) must match the true Overcooked environment, where only " "
cells are valid player positions and counters / pot / dispensers / serve
are non-walkable.
"""

# Convert to indices
POT_INDICES = [y * GRID_WIDTH + x for x, y in POT_LOCATIONS]
SERVING_INDICES = [y * GRID_WIDTH + x for x, y in SERVING_LOCATIONS]
ONION_DISPENSER_INDICES = [y * GRID_WIDTH + x for x, y in ONION_DISPENSERS]

# -------------------------------------------------
# Walls and non-walkable terrain (fixed cramped-room layout)
# In Overcooked env, get_valid_player_positions() = terrain_pos_dict[" "]
# only. So only " " (empty) is walkable; X (counter), P, O, D, S are NOT.
# Layout (see layout.txt):
#   XXPXX
#   O1  O
#   X  2X
#   XDXSX
# -------------------------------------------------
WALL_INDICES = {
    0, 1, 3, 4,        # row 0: X, X, P, X, X
    10, 14,            # row 2: X, _, _, _, X
    15, 17, 19,        # row 3: X, D, X, S, X
}

# All cells that are not " " (empty) — agents cannot step here (matches env).
BLOCKED_MOVEMENT_INDICES = (
    WALL_INDICES
    | set(POT_INDICES)
    | set(ONION_DISPENSER_INDICES)
    | set(SERVING_INDICES)
)

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
    "self_pos": list(range(GRID_SIZE)),  # Flattened position index (0-19)
    "other_pos": list(range(GRID_SIZE)),
    "self_orientation": list(range(N_DIRECTIONS)),  # 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST
    "other_orientation": list(range(N_DIRECTIONS)),
    "self_held": list(range(N_HELD_TYPES)),  # 0=none, 1=onion, 2=dish, 3=soup
    "other_held": list(range(N_HELD_TYPES)),
    "pot_state": list(range(N_POT_STATES)),  # 0=idle, 1=cooking, 2=ready
    # Checkbox-style task memory (adapted from Independent model):
    # ck_put1/2/3: onions added to pot; ck_plated: soup plated; ck_delivered: soup delivered.
    "ck_put1": [0, 1],
    "ck_put2": [0, 1],
    "ck_put3": [0, 1],
    "ck_plated": [0, 1],
    "ck_delivered": [0, 1],
    # Optional explicit event-state for compatibility with existing code:
    "soup_delivered": [0, 1],  # 0=no, 1=yes (resets each step)
}

# -------------------------------------------------
# Observations (full joint observation)
# -------------------------------------------------
observations = {
    "self_pos": list(range(GRID_SIZE)),
    "other_pos": list(range(GRID_SIZE)),
    "self_orientation": list(range(N_DIRECTIONS)),
    "self_held": list(range(N_HELD_TYPES)),
    "other_held": list(range(N_HELD_TYPES)),
    "pot_state": list(range(N_POT_STATES)),
    "soup_delivered": [0, 1],
}

observation_state_dependencies = {
    "self_pos": ["self_pos"],
    "other_pos": ["other_pos"],
    "self_orientation": ["self_orientation"],
    "other_orientation": ["other_orientation"],
    "self_held": ["self_held"],
    "other_held": ["other_held"],
    "pot_state": ["pot_state"],
    "soup_delivered": ["soup_delivered"],
}

state_state_dependencies = {
    # Positions depend on previous positions, orientations, and joint action
    "self_pos": ["self_pos", "self_orientation", "other_pos"],
    "other_pos": ["other_pos", "other_orientation", "self_pos"],
    # Orientations change when moving
    "self_orientation": ["self_orientation"],
    "other_orientation": ["other_orientation"],
    # Held objects depend on positions, pot state, and interactions
    "self_held": ["self_pos", "self_held", "other_pos", "pot_state"],
    "other_held": ["other_pos", "other_held", "self_pos", "pot_state"],
    # Pot state depends on agent positions, held objects, and interactions
    "pot_state": ["self_pos", "other_pos", "self_held", "other_held", "pot_state"],
    # Checkbox memory over joint context (adapted from Independent model):
    "ck_put1": ["ck_put1", "self_pos", "other_pos", "self_orientation", "other_orientation", "self_held", "other_held", "pot_state"],
    "ck_put2": ["ck_put2", "self_pos", "other_pos", "self_orientation", "other_orientation", "self_held", "other_held", "pot_state"],
    "ck_put3": ["ck_put3", "self_pos", "other_pos", "self_orientation", "other_orientation", "self_held", "other_held", "pot_state"],
    "ck_plated": ["ck_plated", "self_pos", "other_pos", "self_orientation", "other_orientation", "self_held", "other_held", "pot_state"],
    "ck_delivered": ["ck_delivered", "self_pos", "other_pos", "self_orientation", "other_orientation", "self_held", "other_held"],
    # Soup delivery event (state) depends on joint positions and held objects
    "soup_delivered": ["self_pos", "other_pos", "self_held", "other_held"],
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


def position_in_front(pos_idx, orientation_idx, width=GRID_WIDTH, height=GRID_HEIGHT):
    """
    Return the grid index of the cell in front of an agent at pos_idx facing orientation_idx.
    Matches env: INTERACT acts on the cell in the direction the agent is facing.
    Returns None if the cell in front is out of bounds.
    """
    x, y = index_to_xy(pos_idx, width)
    if 0 <= orientation_idx < N_DIRECTIONS:
        dx, dy = DIRECTIONS[orientation_idx]
    else:
        dx, dy = 0, 0
    fx, fy = x + dx, y + dy
    if 0 <= fx < width and 0 <= fy < height:
        return fy * width + fx
    return None

