"""
IndividuallyCollective paradigm model init for Overcooked - Cramped Room layout (Monotonic)
Layout:
  XXPXX
  O1  O
  X  2X
  XDXSX
"""


GRID_WIDTH = 5
GRID_HEIGHT = 4
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT 


RECIPE_ONIONS = 3
COOK_TIME = 0


POT_LOCATIONS = [(2, 0)]
SERVING_LOCATIONS = [(3, 3)]
ONION_DISPENSERS = [(0, 1), (4, 1)]
DISH_DISPENSERS = [(1, 3)]


POT_INDICES = [y * GRID_WIDTH + x for x, y in POT_LOCATIONS]
SERVING_INDICES = [y * GRID_WIDTH + x for x, y in SERVING_LOCATIONS]
ONION_DISPENSER_INDICES = [y * GRID_WIDTH + x for x, y in ONION_DISPENSERS]
DISH_DISPENSER_INDICES = [y * GRID_WIDTH + x for x, y in DISH_DISPENSERS]

COUNTER_INDICES = {
    0, 1, 3, 4,        # row 0: X, X, P, X, X
    10, 14,            # row 2: X, _, _, _, X
    15, 17, 19,        # row 3: X, D, X, S, X
}


WALKABLE_INDICES = [6, 7, 8, 11, 12, 13]
N_WALKABLE = len(WALKABLE_INDICES)

# # Non-walkable cells (for movement / position_in_front)
# WALL_INDICES = {
#     0, 1, 3, 4, 10, 14, 15, 17, 19,
# }
# BLOCKED_MOVEMENT_INDICES = (
#     WALL_INDICES
#     | set(POT_INDICES)
#     | set(ONION_DISPENSER_INDICES)
#     | set(DISH_DISPENSER_INDICES)
#     | set(SERVING_INDICES)
# )



# Actions (per-agent)
NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
# N_JOINT_ACTIONS = N_ACTIONS * N_ACTIONS  # 36

INTERACT_SUCCESS_PROB = 0.9  # 0.9 = 10% chance interact "fails" (no change)

# Directions
DIR_NORTH = (0, -1)
DIR_SOUTH = (0, 1)
DIR_EAST = (1, 0)
DIR_WEST = (-1, 0)
DIRECTIONS = [DIR_NORTH, DIR_SOUTH, DIR_EAST, DIR_WEST]
N_DIRECTIONS = 4

# Held object types
HELD_NONE = 0
HELD_ONION = 1
HELD_DISH = 2
HELD_SOUP = 3
N_HELD_TYPES = 4

POT_0 = 0          # 0 onions; idle
POT_1 = 1          # 1 onion
POT_2 = 2          # 2 onions
POT_3 = 3      # cooked soup ready
N_POT_STATES = 4


FRONT_WALL = 0
FRONT_EMPTY = 1    # walkable cell
FRONT_ONION = 2    # onion dispenser
FRONT_DISH = 3     # dish dispenser
FRONT_POT = 4
FRONT_SERVE = 5
FRONT_COUNTER = 6  # counter (can drop objects)
N_FRONT_TYPES = 7


# States
states = {

    "self_pos": list(range(N_WALKABLE)),
    "self_orientation": list(range(N_DIRECTIONS)),  
    "self_held": list(range(N_HELD_TYPES)),  

    "pot_state": list(range(N_POT_STATES)),

    "ck_put1": list(range(2)),
    "ck_put2": list(range(2)),
    "ck_put3": list(range(2)),
    "ck_plated": list(range(2)),
    "ck_delivered": list(range(2)),

    # --- multi-agent (commented for single-agent run) ---
    # "other_pos": list(range(N_WALKABLE)),
    # "other_held": list(range(N_HELD_TYPES)),

}

# Observations 
observations = {
    "self_pos_obs": list(range(N_WALKABLE)),
    "self_orientation_obs": list(range(N_DIRECTIONS)),
    "self_held_obs": list(range(N_HELD_TYPES)),

    "pot_state_obs": list(range(N_POT_STATES)),
    "soup_delivered_obs": [0, 1],

    # --- multi-agent (commented for single-agent run) ---
    # "other_pos_obs": list(range(N_WALKABLE)),
    # "other_held_obs": list(range(N_HELD_TYPES)),
}

observation_state_dependencies = {
    "self_pos_obs": ["self_pos"],
    "self_orientation_obs": ["self_orientation"],
    "self_held_obs": ["self_held"],

    "pot_state_obs": ["pot_state"],

    "soup_delivered_obs": ["self_pos", "self_orientation", "self_held"],
    
    # --- multi-agent (commented for single-agent run) ---
    # "other_pos_obs": ["other_pos"],
    # "other_held_obs": ["other_held"],
}

state_state_dependencies = {
    "self_pos": ["self_pos"],  # single-agent: no other_pos collision
    # "self_pos": ["self_pos", "other_pos"],  # multi-agent: collision with other
    "self_orientation": ["self_orientation"],
    "self_held": ["self_pos", "self_orientation", "self_held", "pot_state"],

    "pot_state": ["self_pos", "self_orientation", "self_held", "pot_state"],

    # --- multi-agent (commented for single-agent run) ---
    # "other_pos": ["other_pos"],
    # "other_held": ["other_held", "other_pos"],

    # Checkbox memory: each flips from local interaction context; no ck_delivered dep (can jump between them).
    "ck_put1":     ["ck_put1", "self_pos", "self_orientation", "self_held", "pot_state"],
    "ck_put2":     ["ck_put2", "self_pos", "self_orientation", "self_held", "pot_state"],
    "ck_put3":     ["ck_put3", "self_pos", "self_orientation", "self_held", "pot_state"],
    "ck_plated":   ["ck_plated", "self_pos", "self_orientation", "self_held", "pot_state"],
    "ck_delivered":["ck_delivered", "self_pos", "self_orientation", "self_held"],
}

# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def compute_front_tile_type(walkable_idx: int, orientation_idx: int) -> int:
    
    grid_idx = walkable_idx_to_grid_idx(walkable_idx)
    x, y = index_to_xy(grid_idx)
    dx, dy = DIRECTIONS[orientation_idx]
    fx, fy = x + dx, y + dy
    if fx < 0 or fx >= GRID_WIDTH or fy < 0 or fy >= GRID_HEIGHT:
        return FRONT_WALL
    fidx = xy_to_index(fx, fy)
    if fidx in POT_INDICES:
        return FRONT_POT
    if fidx in SERVING_INDICES:
        return FRONT_SERVE
    if fidx in ONION_DISPENSER_INDICES:
        return FRONT_ONION
    if fidx in DISH_DISPENSER_INDICES:
        return FRONT_DISH
    if fidx in COUNTER_INDICES:
        return FRONT_COUNTER
    if fidx in WALKABLE_INDICES:
        return FRONT_EMPTY
    return FRONT_WALL

def xy_to_index(x: int, y: int, width: int = GRID_WIDTH) -> int:
    return y * width + x

def index_to_xy(index: int, width: int = GRID_WIDTH):
    y = index // width
    x = index % width
    return x, y


def direction_to_index(direction):
    """Map (dx, dy) tuple to orientation index 0..3 (NORTH, SOUTH, EAST, WEST)."""
    for i, d in enumerate(DIRECTIONS):
        if d == direction:
            return i
    return 0  # fallback to NORTH


def object_name_to_held_type(obj_name):
    if obj_name is None:
        return HELD_NONE
    obj_map = {"onion": HELD_ONION, "dish": HELD_DISH, "soup": HELD_SOUP}
    return obj_map.get(obj_name, HELD_NONE)


def walkable_idx_to_grid_idx(walkable_idx: int) -> int:
    """Convert walkable position index (0..5) to grid cell index (0..19)."""
    if 0 <= walkable_idx < N_WALKABLE:
        return WALKABLE_INDICES[walkable_idx]
    return WALKABLE_INDICES[0]  # fallback


def grid_idx_to_walkable_idx(grid_idx: int):
    """Convert grid cell index (0..19) to walkable index (0..5). Returns None if cell is not walkable."""
    for w in range(N_WALKABLE):
        if WALKABLE_INDICES[w] == grid_idx:
            return w
    return None


def position_in_front(walkable_idx: int, orientation_idx: int, width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
    """Grid index of the cell in front of (walkable_idx, orientation_idx), or None if out of bounds."""
    grid_idx = walkable_idx_to_grid_idx(walkable_idx)
    x, y = index_to_xy(grid_idx, width)
    if 0 <= orientation_idx < N_DIRECTIONS:
        dx, dy = DIRECTIONS[orientation_idx]
    else:
        dx, dy = 0, 0
    fx, fy = x + dx, y + dy
    if 0 <= fx < width and 0 <= fy < height:
        return xy_to_index(fx, fy, width)
    return None


def is_at_location(grid_idx: int, location_indices) -> bool:
    return grid_idx in location_indices


def is_at_pot(grid_idx: int) -> bool:
    return is_at_location(grid_idx, POT_INDICES)


def is_at_serving(grid_idx: int) -> bool:
    return is_at_location(grid_idx, SERVING_INDICES)


def is_at_onion_dispenser(grid_idx: int) -> bool:
    return is_at_location(grid_idx, ONION_DISPENSER_INDICES)


# -------------------------------------------------
