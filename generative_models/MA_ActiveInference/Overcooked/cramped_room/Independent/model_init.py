# model_init.py
"""
Independent paradigm model init for Overcooked - Cramped Room layout (Stage 1).

Each agent uses a single-agent model and selects only its own action.
The other agent is treated as part of the environment.

Stage 1:
- Known map (no map latent factor).
- Strong affordance observation: front_tile_type (deterministic from pos+orientation+other_pos+layout).
- soup_delivered is OBSERVATION-ONLY (event), NOT a hidden state factor.
- Recipe requires 3 onions (per your config), cook_time = 1.
"""

# -------------------------------------------------
# Grid size (cramped_room specific)
# -------------------------------------------------
GRID_WIDTH = 5
GRID_HEIGHT = 4
GRID_SIZE = GRID_WIDTH * GRID_HEIGHT  # 20 cells

# -------------------------------------------------
# Recipe / cooking (from your config)
# -------------------------------------------------
RECIPE_ONIONS = 3
COOK_TIME = 1  # cook_time: 1 step in your config

# -------------------------------------------------
# Layout-specific locations
# -------------------------------------------------
POT_LOCATIONS = [(2, 0)]
SERVING_LOCATIONS = [(3, 3)]
ONION_DISPENSERS = [(0, 1), (4, 1)]
DISH_DISPENSERS = [(1, 3)]

# Convert to indices
POT_INDICES = [y * GRID_WIDTH + x for x, y in POT_LOCATIONS]
SERVING_INDICES = [y * GRID_WIDTH + x for x, y in SERVING_LOCATIONS]
ONION_DISPENSER_INDICES = [y * GRID_WIDTH + x for x, y in ONION_DISPENSERS]
DISH_DISPENSER_INDICES = [y * GRID_WIDTH + x for x, y in DISH_DISPENSERS]

# -------------------------------------------------
# Walls and non-walkable terrain (fixed cramped-room layout)
# In Overcooked env, get_valid_player_positions() = terrain_pos_dict[" "]
# only. So only " " (empty) is walkable; X (counter), P, O, D, S are NOT.
# Layout:
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
    | set(DISH_DISPENSER_INDICES)
    | set(SERVING_INDICES)
)

def is_wall_index(idx: int) -> bool:
    return idx in WALL_INDICES

def is_blocked_for_movement(idx: int) -> bool:
    return idx in BLOCKED_MOVEMENT_INDICES

# -------------------------------------------------
# Primitive actions (per-agent)
# -------------------------------------------------
NORTH, SOUTH, EAST, WEST, STAY, INTERACT = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6

# INTERACT outcome noise: with this probability the intended transition happens;
# with (1 - INTERACT_SUCCESS_PROB) "nothing happens" (held state unchanged).
# Slightly stochastic INTERACT increases expected info gain so the agent may prefer it over movement.
INTERACT_SUCCESS_PROB = 0.7  # 0.9 = 10% chance interact "fails" (no change)

# -------------------------------------------------
# Directions (match action/orientation conventions)
# orientation index -> (dx, dy)
# 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST
# -------------------------------------------------
DIR_NORTH = (0, -1)
DIR_SOUTH = (0, 1)
DIR_EAST = (1, 0)
DIR_WEST = (-1, 0)
DIRECTIONS = [DIR_NORTH, DIR_SOUTH, DIR_EAST, DIR_WEST]
N_DIRECTIONS = 4

# -------------------------------------------------
# Held object types
# -------------------------------------------------
HELD_NONE = 0
HELD_ONION = 1
HELD_DISH = 2
HELD_SOUP = 3
N_HELD_TYPES = 4

# -------------------------------------------------
# Pot states (expanded for 3 onions)
# -------------------------------------------------
POT_0 = 0          # 0 onions
POT_1 = 1          # 1 onion
POT_2 = 2          # 2 onions
POT_COOKING = 3    # 3 onions added, cooking
POT_READY = 4      # cooked soup ready
N_POT_STATES = 5

# -------------------------------------------------
# front tile type observation
# -------------------------------------------------
FRONT_WALL = 0
FRONT_EMPTY = 1
FRONT_ONION = 2
FRONT_DISH = 3
FRONT_POT = 4
FRONT_SERVE = 5
FRONT_BLOCKED_BY_OTHER = 6
FRONT_COUNTER = 7   # empty counter "X" — can drop/pick up objects
N_FRONT_TYPES = 8

# -------------------------------------------------
# States (single-agent perspective)
# NOTE: soup_delivered is OBSERVATION ONLY and NOT a state factor.
# -------------------------------------------------
states = {
    "agent_pos": list(range(GRID_SIZE)),
    "agent_orientation": list(range(N_DIRECTIONS)),
    "agent_held": list(range(N_HELD_TYPES)),
    "other_agent_pos": list(range(GRID_SIZE)),
    "pot_state": list(range(N_POT_STATES)),
}

# -------------------------------------------------
# Observations (single-agent perspective)
# soup_delivered is an event observation: 1 only on delivery timestep.
# -------------------------------------------------
observations = {
    "agent_pos": list(range(GRID_SIZE)),
    "agent_orientation": list(range(N_DIRECTIONS)),
    "agent_held": list(range(N_HELD_TYPES)),
    "other_agent_pos": list(range(GRID_SIZE)),
    "pot_state": list(range(N_POT_STATES)),
    "front_tile_type": list(range(N_FRONT_TYPES)),
    "soup_delivered": [0, 1],
}

# -------------------------------------------------
# Observation dependencies (for factorized A)
# soup_delivered is event obs; depends on states that determine delivery (pos/held/other).
# -------------------------------------------------
observation_state_dependencies = {
    "agent_pos": ["agent_pos"],
    "agent_orientation": ["agent_orientation"],
    "agent_held": ["agent_held"],
    "other_agent_pos": ["other_agent_pos"],
    "pot_state": ["pot_state"],
    "front_tile_type": ["agent_pos", "agent_orientation", "other_agent_pos"],
    "soup_delivered": ["agent_pos", "agent_held", "other_agent_pos"],
}

# -------------------------------------------------
# State transition dependencies (for factorized B or functional B)
# soup_delivered intentionally omitted (not a hidden state)
# -------------------------------------------------
state_state_dependencies = {
    "agent_pos": ["agent_pos", "agent_orientation", "other_agent_pos"],
    "agent_orientation": ["agent_orientation"],
    "agent_held": ["agent_pos", "agent_orientation", "agent_held", "other_agent_pos", "pot_state"],
    "other_agent_pos": ["other_agent_pos"],
    "pot_state": ["agent_pos", "agent_orientation", "agent_held", "other_agent_pos", "pot_state"],
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

def object_name_to_held_type(obj_name):
    if obj_name is None:
        return HELD_NONE
    obj_map = {"onion": HELD_ONION, "dish": HELD_DISH, "soup": HELD_SOUP}
    return obj_map.get(obj_name, HELD_NONE)

def is_at_location(pos_idx: int, location_indices) -> bool:
    return pos_idx in location_indices

def is_at_serving(pos_idx: int) -> bool:
    return is_at_location(pos_idx, SERVING_INDICES)

def is_at_onion_dispenser(pos_idx: int) -> bool:
    return is_at_location(pos_idx, ONION_DISPENSER_INDICES)

def is_at_dish_dispenser(pos_idx: int) -> bool:
    return is_at_location(pos_idx, DISH_DISPENSER_INDICES)

def is_at_pot(pos_idx: int) -> bool:
    return is_at_location(pos_idx, POT_INDICES)

def direction_to_index(direction) -> int:
    """Convert (dx, dy) direction tuple to index. 0=N, 1=S, 2=E, 3=W."""
    if direction == DIR_NORTH:
        return 0
    if direction == DIR_SOUTH:
        return 1
    if direction == DIR_EAST:
        return 2
    if direction == DIR_WEST:
        return 3
    return 0

def index_to_direction(idx: int):
    """Convert index to (dx, dy) direction tuple."""
    return DIRECTIONS[idx]

# -------------------------------------------------
# compute front tile type from known map
# -------------------------------------------------
def compute_front_cell_index(agent_pos_idx: int, agent_ori_idx: int):
    if agent_ori_idx < 0 or agent_ori_idx >= N_DIRECTIONS:
        return None
    x, y = index_to_xy(agent_pos_idx)
    dx, dy = DIRECTIONS[agent_ori_idx]
    fx, fy = x + dx, y + dy
    if fx < 0 or fx >= GRID_WIDTH or fy < 0 or fy >= GRID_HEIGHT:
        return None
    return xy_to_index(fx, fy)

def compute_front_tile_type(agent_pos_idx: int, agent_ori_idx: int, other_pos_idx: int) -> int:
    front_idx = compute_front_cell_index(agent_pos_idx, agent_ori_idx)
    if front_idx is None:
        return FRONT_WALL
    if front_idx == other_pos_idx:
        return FRONT_BLOCKED_BY_OTHER
    if front_idx in WALL_INDICES:
        return FRONT_COUNTER  # "X" = counter (drop/pick up); only out-of-bounds is FRONT_WALL
    if front_idx in POT_INDICES:
        return FRONT_POT
    if front_idx in SERVING_INDICES:
        return FRONT_SERVE
    if front_idx in DISH_DISPENSER_INDICES:
        return FRONT_DISH
    if front_idx in ONION_DISPENSER_INDICES:
        return FRONT_ONION
    return FRONT_EMPTY
