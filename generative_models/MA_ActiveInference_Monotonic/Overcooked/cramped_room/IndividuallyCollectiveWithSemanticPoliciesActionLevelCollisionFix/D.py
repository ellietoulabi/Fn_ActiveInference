# D.py
"""
Prior beliefs (D) for IndividuallyCollective paradigm — Cramped Room.

Includes:
- self position / orientation / held
- other agent position / orientation / held
- pot state
- checkbox factors
- binary counter occupancy factors (all start EMPTY)
"""

import numpy as np
from . import model_init

DEFAULT_START_GRID_XY = (1, 2)
DEFAULT_OTHER_START_GRID_XY = (3, 2)

CHECKBOX_FACTORS = ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered")


def _grid_xy_to_walkable(xy) -> int:
    gx, gy = xy
    grid_idx = model_init.xy_to_index(gx, gy)
    walkable_idx = model_init.grid_idx_to_walkable_idx(grid_idx)
    return int(walkable_idx) if walkable_idx is not None else 0


def build_D(
    self_start_pos: int | None = None,
    self_start_ori: int = 0,
    other_start_pos: int | None = None,
    other_start_ori: int = 0,
) -> dict[str, np.ndarray]:

    if self_start_pos is None:
        self_start_pos = _grid_xy_to_walkable(DEFAULT_START_GRID_XY)
    else:
        self_start_pos = int(self_start_pos)

    if other_start_pos is None:
        other_start_pos = _grid_xy_to_walkable(DEFAULT_OTHER_START_GRID_XY)
    else:
        other_start_pos = int(other_start_pos)

    self_start_ori = int(self_start_ori)
    other_start_ori = int(other_start_ori)

    D: dict[str, np.ndarray] = {}

    # self_pos
    D["self_pos"] = np.zeros(model_init.N_WALKABLE, dtype=float)
    idx = self_start_pos if 0 <= self_start_pos < model_init.N_WALKABLE else 0
    D["self_pos"][idx] = 1.0

    # self_orientation
    D["self_orientation"] = np.zeros(model_init.N_DIRECTIONS, dtype=float)
    ori_idx = self_start_ori if 0 <= self_start_ori < model_init.N_DIRECTIONS else 0
    D["self_orientation"][ori_idx] = 1.0

    # self_held
    D["self_held"] = np.zeros(model_init.N_HELD_TYPES, dtype=float)
    D["self_held"][model_init.HELD_NONE] = 1.0

    # other_pos
    D["other_pos"] = np.zeros(model_init.N_WALKABLE, dtype=float)
    other_idx = other_start_pos if 0 <= other_start_pos < model_init.N_WALKABLE else 0
    D["other_pos"][other_idx] = 1.0

    # other_orientation
    D["other_orientation"] = np.zeros(model_init.N_DIRECTIONS, dtype=float)
    other_ori_idx = other_start_ori if 0 <= other_start_ori < model_init.N_DIRECTIONS else 0
    D["other_orientation"][other_ori_idx] = 1.0

    # other_held
    D["other_held"] = np.zeros(model_init.N_HELD_TYPES, dtype=float)
    D["other_held"][model_init.HELD_NONE] = 1.0

    # pot_state
    D["pot_state"] = np.zeros(model_init.N_POT_STATES, dtype=float)
    D["pot_state"][model_init.POT_0] = 1.0

    # Counters: start empty
    for grid_idx in model_init.MODELED_COUNTERS:
        cf = f"ctr_{grid_idx}"
        D[cf] = np.zeros(model_init.N_CTR_STATES, dtype=float)
        D[cf][model_init.CTR_EMPTY] = 1.0

    # Checkboxes: all start unchecked
    for ck in CHECKBOX_FACTORS:
        D[ck] = np.array([1.0, 0.0], dtype=float)

    # Any factor in model_init.states not yet in D gets uniform prior
    for factor, values in model_init.states.items():
        if factor in D:
            continue
        n = len(values)
        D[factor] = np.ones(n, dtype=float) / float(n) if n > 0 else np.array([], dtype=float)

    return D


def D_fn(config: dict | None = None) -> dict[str, np.ndarray]:
    if config is None:
        return build_D()

    return build_D(
        self_start_pos=config.get("self_start_pos"),
        self_start_ori=config.get("self_start_ori", 0),
        other_start_pos=config.get("other_start_pos"),
        other_start_ori=config.get("other_start_ori", 0),
    )