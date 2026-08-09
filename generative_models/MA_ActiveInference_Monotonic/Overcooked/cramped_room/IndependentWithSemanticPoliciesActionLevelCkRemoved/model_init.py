"""
Independent paradigm model init for Overcooked - Cramped Room layout (Monotonic).

Each agent has a single-agent generative model.  The other agent is observed
and inferred but not co-planned: its state factors transition as identity
(autonomous/uncontrolled) during policy rollout.

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
# Independent agent plans only over its own actions; no joint-pair or noop needed.
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
]
MODES = ["stay", "interact"]
SEMANTIC_ACTIONS = [(dst, mode) for dst in DESTINATIONS for mode in MODES]
N_ACTIONS = len(SEMANTIC_ACTIONS)  # 20

SELF = 0
OTHER = 1

ACTION_NAMES = {i: f"{dst}:{mode}" for i, (dst, mode) in enumerate(SEMANTIC_ACTIONS)}


def semantic_action_from_index(action_idx: int) -> tuple[str, str]:
    i = int(action_idx)
    if i < 0 or i >= N_ACTIONS:
        return SEMANTIC_ACTIONS[0]
    return SEMANTIC_ACTIONS[i]


def semantic_index(dst: str, mode: str) -> int:
    return SEMANTIC_ACTIONS.index((dst, mode))


def construct_semantic_policies(policy_len: int = 2) -> list[list[int]]:
    """Enumerate all self-semantic policies over action indices [0..N_ACTIONS-1]."""
    from itertools import product

    if policy_len <= 0:
        return []
    return [list(p) for p in product(range(N_ACTIONS), repeat=int(policy_len))]


# Mark policy steps that are env primitives (0..N_PRIMITIVE_ACTIONS-1) for B_fn rollout.
# Format: (PRIMITIVE_POLICY_STEP, a_self)
# This disambiguates primitive 0..5 from semantic indices 0..N_ACTIONS-1 inside B_fn.
PRIMITIVE_POLICY_STEP = "__primitive_policy_step__"


# For each semantic destination, define a canonical walkable index and orientation
# such that the landmark is in front of the agent after macro-teleport.
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
}

INTERACT_SUCCESS_PROB = 1.0

# Deliberately less than 1.0, unlike INTERACT_SUCCESS_PROB above. Models genuine
# uncertainty about whether a chosen INTERACT that would advance the recipe (onion
# pickup, dish pickup, onion deposit at pot, soup pickup at pot) actually completes
# as intended. With INTERACT_SUCCESS_PROB=1.0 and already-confident self_pos/
# self_held/pot_state beliefs, every transition is fully predictable regardless of
# which candidate policy is chosen, so information gain -- which only rewards
# resolving genuine uncertainty, not making progress toward a known-good outcome --
# cannot differentiate the objectively correct next action from an irrelevant one
# (see ai/02-debug.md, MA Overcooked section I.1). This is a deliberate, small,
# intentional divergence between the agent's belief model and the (fully
# deterministic) real environment, introduced specifically to restore that
# epistemic signal without touching the sparse preference model (C_fn) or the
# policy prior (E). Does NOT apply to serving/delivery (already differentiated by
# utility, since delivery is one macro-step away for that specific transition) or
# to counter drop/pickup (not implicated in the diagnosed last-mile stall).
PROGRESS_SUCCESS_PROB = 0.85

# Softens B_self_pos's collision-blocking prediction against the believed
# other_pos. IND treats the other agent as environment, not a co-planned actor
# (by design), so this check can only ever compare the intended move against
# the other agent's *current* believed position -- it has no way, even in
# principle, to foresee a same-instant collision from a simultaneous move
# toward the same tile (that would require modeling the other's intent, which
# is deliberately out of scope for this paradigm). Left at a hard 0/1 blocking
# probability, that blind spot produces a perfectly deterministic, identically
# wrong prediction on every retry after a real collision, which combined with
# a sharp action-selection precision can lock two independent agents into a
# repeating collision deadlock (see ai/02-debug.md, MA Overcooked section
# I.1's 2026-08-08 follow-up). This constant blends in a small, honest amount
# of uncertainty about whether a move actually lands -- not a prediction about
# the other agent's behavior, just an acknowledgment that this model's
# current-position-only check cannot always be trusted -- enough to keep
# retries from being identical every time, without giving IND any actual
# foresight into or reasoning about the other agent.
#
# DELIBERATELY set to 0.0 (inert), not deleted. An isolation test (2026-08-08,
# see ai/02-debug.md I.1) found that lowering the action-selection precision
# alpha (8.0 -> 1.0) already fixes the collision deadlock on its own -- a
# stochastic sampler doesn't hard-lock on a near-tie the way a sharp one does,
# so no belief-level fix was actually needed for that failure mode. With
# alpha=1.0, enabling this constant (>0) measured *worse* outcomes than
# leaving it at 0: it leaks uncertainty into self_pos's info-gain contribution
# in proportion to how many movement steps a candidate policy takes, biasing
# the agent toward farther/less relevant destinations (the "distance bias"
# finding) regardless of collision risk. Left in place, at 0, as a documented,
# available mechanism -- not reintroduced, since alpha already covers the
# problem this was built for. Do not re-enable without re-reading the
# isolation test results first.
MOVE_UNCERTAINTY = 0.0

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

    # v3 (CkRemoved) only: ck_put1/2/3 and ck_plated deleted (confirmed dead
    # computation, ai/02-debug.md section I.1). ck_delivered kept -- the sole
    # link between "the agent just delivered" and C_fn's only preference.
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
    # Ego agent: normal action-driven transitions.
    # self_pos still depends on other_pos for collision avoidance.
    "self_pos": ["self_pos", "other_pos"],
    "self_orientation": ["self_orientation"],
    "self_held": ["self_pos", "self_orientation", "self_held", "pot_state"] + COUNTER_FACTORS,

    # Shared environment: driven by ego agent only (other assumed STAY).
    "pot_state": ["self_pos", "self_orientation", "self_held", "pot_state"],

    "ck_delivered": ["ck_delivered", "self_pos", "self_orientation", "self_held"],

    # Other agent: identity transitions (observed/inferred, not controlled).
    "other_pos": ["other_pos"],
    "other_orientation": ["other_orientation"],
    "other_held": ["other_held"],
}

for cf in COUNTER_FACTORS:
    state_state_dependencies[cf] = [cf, "self_pos", "self_orientation", "self_held"]


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