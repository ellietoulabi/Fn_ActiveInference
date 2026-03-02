

GRID_WIDTH = 5
GRID_HEIGHT = 4
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT  # 20 cells


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


NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
# INTERACT: use the adjacent cell in front (facing direction); agent does not step onto pot/serve/etc.

# INTERACT outcome noise: with this probability the intended transition happens;
# with (1 - INTERACT_SUCCESS_PROB) "nothing happens" (held state unchanged).
# Slightly stochastic INTERACT increases expected info gain so the agent may prefer it over movement.
INTERACT_SUCCESS_PROB = 0.9  # 0.9 = 10% chance interact "fails" (no change)


DIR_NORTH = (0, -1)
DIR_SOUTH = (0, 1)
DIR_EAST = (1, 0)
DIR_WEST = (-1, 0)
DIRECTIONS = [DIR_NORTH, DIR_SOUTH, DIR_EAST, DIR_WEST]
N_DIRECTIONS = 4


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


def compute_front_grid_idx(walkable_idx: int, orientation_idx: int) -> int | None:
    """
    Return the grid cell index directly in front of (walkable_idx, orientation_idx),
    or None if out of bounds.
    """
    grid_idx = walkable_idx_to_grid_idx(walkable_idx)
    x, y = index_to_xy(grid_idx)
    dx, dy = DIRECTIONS[orientation_idx]
    fx, fy = x + dx, y + dy
    if fx < 0 or fx >= GRID_WIDTH or fy < 0 or fy >= GRID_HEIGHT:
        return None
    return xy_to_index(fx, fy)


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


# Five counters are adjacent to the 6 walkable cells in this reduced model.
# These are the only counters the agent can interact with (via FRONT_COUNTER).
COUNTER_SLOT_GRID_IDXS = [1, 3, 10, 14, 17]
COUNTER_SLOT_BY_GRID = {g: i for i, g in enumerate(COUNTER_SLOT_GRID_IDXS)}
N_COUNTERS = len(COUNTER_SLOT_GRID_IDXS)


def front_counter_slot(walkable_idx: int, orientation_idx: int) -> int | None:
    """If the front tile is one of the modeled counters, return its slot index 0..4."""
    fidx = compute_front_grid_idx(walkable_idx, orientation_idx)
    if fidx is None:
        return None
    return COUNTER_SLOT_BY_GRID.get(int(fidx))


states = {
    "agent_pos": list(range(6)),
    "agent_orientation": list(range(N_DIRECTIONS)),
    "agent_held": list(range(N_HELD_TYPES)),
    "pot_state": list(range(N_POT_STATES)),
    "ck_put1": list(range(2)),
    "ck_put2": list(range(2)),
    "ck_put3": list(range(2)),
    "ck_plated": list(range(2)),
    "ck_delivered": list(range(2)),
    "counter_0": list(range(N_HELD_TYPES)),
    "counter_1": list(range(N_HELD_TYPES)),
    "counter_2": list(range(N_HELD_TYPES)),
    "counter_3": list(range(N_HELD_TYPES)),
    "counter_4": list(range(N_HELD_TYPES)),
}


observations = {
    "agent_pos_obs": list(range(6)),
    "agent_orientation_obs": list(range(N_DIRECTIONS)),
    "agent_held_obs": list(range(N_HELD_TYPES)),
    "pot_state_obs": list(range(N_POT_STATES)),
    "soup_delivered_obs": [0, 1],
    "counter_0_obs": list(range(N_HELD_TYPES)),
    "counter_1_obs": list(range(N_HELD_TYPES)),
    "counter_2_obs": list(range(N_HELD_TYPES)),
    "counter_3_obs": list(range(N_HELD_TYPES)),
    "counter_4_obs": list(range(N_HELD_TYPES)),
}


observation_state_dependencies = {
    "agent_pos_obs": ["agent_pos"],
    "agent_orientation_obs": ["agent_orientation"],
    "agent_held_obs": ["agent_held"],
    "pot_state_obs": ["pot_state"],
    # soup_delivered_obs is an observation-only event flag (from env sparse reward),
    # not a stable latent state. We keep formal state deps for A_fn's interface,
    # but the event itself is passed separately (and is 0 during planning rollouts).
    "soup_delivered_obs": ["agent_pos", "agent_orientation", "agent_held"],
    "counter_0_obs": ["counter_0"],
    "counter_1_obs": ["counter_1"],
    "counter_2_obs": ["counter_2"],
    "counter_3_obs": ["counter_3"],
    "counter_4_obs": ["counter_4"],
}

state_state_dependencies = {
    "agent_pos": ["agent_pos"],
    "agent_orientation": ["agent_orientation"],
    # agent_held can change due to counter pickup/drop as well as pot/dispensers/serve.
    "agent_held": [
        "agent_pos",
        "agent_orientation",
        "agent_held",
        "pot_state",
        "counter_0",
        "counter_1",
        "counter_2",
        "counter_3",
        "counter_4",
    ],
    "pot_state": ["agent_pos", "agent_orientation", "agent_held", "pot_state"],
    # Checkbox memory:
    # Each checkbox flips based on the same local interaction context that causes the milestone.
    # I decided thta they dont depend on each other because changes happening in one checkbox might be because of another agent's action. so we can jump between them.
    "ck_put1":     ["ck_put1", "agent_pos", "agent_orientation", "agent_held", "pot_state"],
    "ck_put2":     ["ck_put2", "agent_pos", "agent_orientation", "agent_held", "pot_state"],
    "ck_put3":     ["ck_put3", "agent_pos", "agent_orientation", "agent_held", "pot_state"],
    "ck_plated":   ["ck_plated", "agent_pos", "agent_orientation", "agent_held", "pot_state"],
    "ck_delivered":["ck_delivered", "agent_pos", "agent_orientation", "agent_held"],
    # Counters can change on INTERACT depending on agent_pos/orientation/held.
    "counter_0": ["counter_0", "agent_pos", "agent_orientation", "agent_held"],
    "counter_1": ["counter_1", "agent_pos", "agent_orientation", "agent_held"],
    "counter_2": ["counter_2", "agent_pos", "agent_orientation", "agent_held"],
    "counter_3": ["counter_3", "agent_pos", "agent_orientation", "agent_held"],
    "counter_4": ["counter_4", "agent_pos", "agent_orientation", "agent_held"],
}

# -------------------------------------------------
# Utility functions
# -------------------------------------------------
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


