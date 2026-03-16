# model_init.py
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
    0, 1, 3, 4,
    10, 14,
    15, 17, 19,
}

WALKABLE_INDICES = [6, 7, 8, 11, 12, 13]
N_WALKABLE = len(WALKABLE_INDICES)

# Actions (per-agent)
NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6

SELF = 0
OTHER = 1
N_ACTORS = 2

# Interleaved step-action encoding (single integer)
# 0..5   => (SELF, primitive_action)
# 6..11  => (OTHER, primitive_action)
N_INTERLEAVED_STEP_ACTIONS = N_ACTORS * N_ACTIONS

ACTOR_NAMES = {
    SELF: "SELF",
    OTHER: "OTHER",
}

ACTION_NAMES = {
    NORTH: "NORTH",
    SOUTH: "SOUTH",
    EAST: "EAST",
    WEST: "WEST",
    STAY: "STAY",
    INTERACT: "INTERACT",
}

def encode_interleaved_step(actor: int, action: int) -> int:
    return int(actor) * N_ACTIONS + int(action)

def decode_interleaved_step(step_action: int) -> tuple[int, int]:
    a = int(step_action)
    actor = a // N_ACTIONS
    action = a % N_ACTIONS
    return actor, action

def policy_step_to_actions(actor: int, action: int):
    """
    Convert one policy step (actor, action) into effective primitive actions.
    If one agent acts, the other agent is STAY.
    """
    if actor == SELF:
        return action, STAY
    elif actor == OTHER:
        return STAY, action
    return STAY, STAY

INTERACT_SUCCESS_PROB = 1.0

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

# Pot states
POT_0 = 0
POT_1 = 1
POT_2 = 2
POT_3 = 3
N_POT_STATES = 4

# Front tile types
FRONT_WALL = 0
FRONT_EMPTY = 1
FRONT_ONION = 2
FRONT_DISH = 3
FRONT_POT = 4
FRONT_SERVE = 5
FRONT_COUNTER = 6
N_FRONT_TYPES = 7

# Counter contents (modeled counters only)
# We need more than empty/full so the model can predict picking up specific items.
CTR_EMPTY = 0
CTR_ONION = 1
CTR_DISH = 2
CTR_SOUP = 3
N_CTR_STATES = 4

MODELED_COUNTERS = [1, 3, 10, 14, 17]
COUNTER_FACTORS = [f"ctr_{idx}" for idx in MODELED_COUNTERS]

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

    "other_pos": list(range(N_WALKABLE)),
    "other_orientation": list(range(N_DIRECTIONS)),
    "other_held": list(range(N_HELD_TYPES)),
}

for cf in COUNTER_FACTORS:
    states[cf] = list(range(N_CTR_STATES))

# Observations
observations = {
    "self_pos_obs": list(range(N_WALKABLE)),
    "self_orientation_obs": list(range(N_DIRECTIONS)),
    "self_held_obs": list(range(N_HELD_TYPES)),

    "pot_state_obs": list(range(N_POT_STATES)),
    "soup_delivered_obs": [0, 1],

    "other_pos_obs": list(range(N_WALKABLE)),
    "other_orientation_obs": list(range(N_DIRECTIONS)),
    "other_held_obs": list(range(N_HELD_TYPES)),
}

for cf in COUNTER_FACTORS:
    observations[f"{cf}_obs"] = list(range(N_CTR_STATES))

observation_state_dependencies = {
    "self_pos_obs": ["self_pos"],
    "self_orientation_obs": ["self_orientation"],
    "self_held_obs": ["self_held"],
    "pot_state_obs": ["pot_state"],

    "soup_delivered_obs": ["ck_delivered"],

    "other_pos_obs": ["other_pos"],
    "other_orientation_obs": ["other_orientation"],
    "other_held_obs": ["other_held"],
}

for cf in COUNTER_FACTORS:
    observation_state_dependencies[f"{cf}_obs"] = [cf]

state_state_dependencies = {
    "self_pos": ["self_pos", "other_pos"],
    "self_orientation": ["self_orientation"],

    "self_held": ["self_pos", "self_orientation", "self_held", "pot_state"] + COUNTER_FACTORS,

    "pot_state": [
        "self_pos", "self_orientation", "self_held",
        "other_pos", "other_orientation", "other_held",
        "pot_state"
    ],

    "ck_put1": ["ck_put1", "self_pos", "self_orientation", "self_held", "pot_state", "other_pos", "other_orientation", "other_held"],
    "ck_put2": ["ck_put2", "self_pos", "self_orientation", "self_held", "pot_state", "other_pos", "other_orientation", "other_held"],
    "ck_put3": ["ck_put3", "self_pos", "self_orientation", "self_held", "pot_state", "other_pos", "other_orientation", "other_held"],
    "ck_plated": ["ck_plated", "self_pos", "self_orientation", "self_held", "pot_state", "other_pos", "other_orientation", "other_held"],
    "ck_delivered": ["ck_delivered", "self_pos", "self_orientation", "self_held", "other_pos", "other_orientation", "other_held"],

    "other_pos": ["other_pos", "self_pos"],
    "other_orientation": ["other_orientation"],
    "other_held": ["other_pos", "other_orientation", "other_held", "pot_state"] + COUNTER_FACTORS,
}

for cf in COUNTER_FACTORS:
    state_state_dependencies[cf] = [cf, "self_pos", "self_orientation", "self_held", "other_pos", "other_orientation", "other_held"]

# Utility functions
def xy_to_index(x: int, y: int, width: int = GRID_WIDTH) -> int:
    return y * width + x

def index_to_xy(index: int, width: int = GRID_WIDTH):
    y = index // width
    x = index % width
    return x, y

def direction_to_index(direction):
    for i, d in enumerate(DIRECTIONS):
        if d == direction:
            return i
    return 0

def object_name_to_held_type(obj_name):
    if obj_name is None:
        return HELD_NONE
    obj_map = {"onion": HELD_ONION, "dish": HELD_DISH, "soup": HELD_SOUP}
    return obj_map.get(obj_name, HELD_NONE)

def walkable_idx_to_grid_idx(walkable_idx: int) -> int:
    if 0 <= walkable_idx < N_WALKABLE:
        return WALKABLE_INDICES[walkable_idx]
    return WALKABLE_INDICES[0]

def grid_idx_to_walkable_idx(grid_idx: int):
    for w in range(N_WALKABLE):
        if WALKABLE_INDICES[w] == grid_idx:
            return w
    return None

def position_in_front(walkable_idx: int, orientation_idx: int, width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
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

def modeled_counter_in_front(walkable_idx: int, orientation_idx: int):
    fg = position_in_front(walkable_idx, orientation_idx, GRID_WIDTH, GRID_HEIGHT)
    if fg is None:
        return None
    return fg if fg in MODELED_COUNTERS else None

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

def is_at_location(grid_idx: int, location_indices) -> bool:
    return grid_idx in location_indices

def is_at_pot(grid_idx: int) -> bool:
    return is_at_location(grid_idx, POT_INDICES)

def is_at_serving(grid_idx: int) -> bool:
    return is_at_location(grid_idx, SERVING_INDICES)

def is_at_onion_dispenser(grid_idx: int) -> bool:
    return is_at_location(grid_idx, ONION_DISPENSER_INDICES)