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

# Primitive actions (used by environment + macro terminal mode)
NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5
N_PRIMITIVE_ACTIONS = 6

# Semantic macro-actions (used by planning/policies in this model)
DESTINATIONS = [
    "onion1",
    "onion2",
    "dish",
    "serve",
    "pot",
    "cntr1",
    "cntr2",
    "cntr3",
    "cntr4",
    "cntr5",
    "noop",   # semantic no-op used for interleaved compatibility
]
MODES = ["stay", "interact"]
SEMANTIC_ACTIONS = [(dst, mode) for dst in DESTINATIONS for mode in MODES]
N_ACTIONS = len(SEMANTIC_ACTIONS)  # 22

SELF = 0
OTHER = 1
N_ACTORS = 2

# Interleaved step-action encoding (single integer) over semantic actions:
# 0..N_ACTIONS-1              => (SELF, semantic_action)
# N_ACTIONS..2*N_ACTIONS-1    => (OTHER, semantic_action)
N_INTERLEAVED_STEP_ACTIONS = N_ACTORS * N_ACTIONS

ACTOR_NAMES = {
    SELF: "SELF",
    OTHER: "OTHER",
}

ACTION_NAMES = {i: f"{dst}:{mode}" for i, (dst, mode) in enumerate(SEMANTIC_ACTIONS)}


def encode_interleaved_step(actor: int, action: int) -> int:
    return int(actor) * N_ACTIONS + int(action)


def decode_interleaved_step(step_action: int) -> tuple[int, int]:
    a = int(step_action)
    actor = a // N_ACTIONS
    action = a % N_ACTIONS
    return actor, action


def semantic_action_from_index(action_idx: int) -> tuple[str, str]:
    i = int(action_idx)
    if i < 0 or i >= N_ACTIONS:
        return SEMANTIC_ACTIONS[0]
    return SEMANTIC_ACTIONS[i]


def semantic_index(dst: str, mode: str) -> int:
    return SEMANTIC_ACTIONS.index((dst, mode))


def construct_semantic_policies(policy_len: int = 2) -> list[list[int]]:
    """
    Enumerate all semantic policies over action indices [0..N_ACTIONS-1].
    """
    from itertools import product

    if policy_len <= 0:
        return []
    return [list(p) for p in product(range(N_ACTIONS), repeat=int(policy_len))]


# Policy step: simultaneous semantic pair for global agent_0, global agent_1.
# Passed to B_fn as (JOINT_PAIR_LABEL, a0, a1); B_fn maps to ego frame via ego_agent_index.
JOINT_PAIR_LABEL = "__joint_pair__"

# Mark policy steps that are env primitives (0..N_PRIMITIVE_ACTIONS-1) for B_fn rollout.
# Must match the convention used by runners that compile dynamic policies to primitives:
#   (PRIMITIVE_POLICY_STEP, primitive_action)
# This disambiguates primitive 0..5 from semantic indices 0..N_ACTIONS-1 inside B_fn.
PRIMITIVE_POLICY_STEP = "__primitive_policy_step__"


def policy_step_to_actions(actor: int, action: int):
    """
    Convert one interleaved semantic policy step (actor, semantic_action)
    into a pair of semantic action indices.

    The inactive agent receives a semantic noop.
    """
    noop_sem = semantic_index("noop", "stay")

    if actor == SELF:
        return int(action), int(noop_sem)
    elif actor == OTHER:
        return int(noop_sem), int(action)
    return int(noop_sem), int(noop_sem)


# For each semantic destination, define a canonical walkable index and orientation
# such that the landmark is in front of the agent.
# "noop" is a placeholder only and is handled specially in B.py.
SEMANTIC_DEST_TARGET_POSE = {
    "onion1": (0, WEST),   # walkable 0 (grid 6), face dispenser/counter at grid 5
    "onion2": (2, EAST),   # walkable 2 (grid 8), face dispenser at grid 9
    "dish": (3, SOUTH),    # walkable 3 (grid 11), face dish dispenser at grid 16
    "serve": (5, SOUTH),   # walkable 5 (grid 13), face serving at grid 18
    "pot": (1, NORTH),     # walkable 1 (grid 7), face pot at grid 2
    "cntr1": (0, NORTH),   # counter grid 1
    "cntr2": (2, NORTH),   # counter grid 3
    "cntr3": (3, WEST),    # counter grid 10
    "cntr4": (5, EAST),    # counter grid 14
    "cntr5": (4, SOUTH),   # counter grid 17
    "noop": (0, NORTH),    # placeholder only
}

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