# D.py
"""
Prior beliefs (D) for IndividuallyCollective paradigm (single-agent) — Cramped Room.
Includes binary counter occupancy factors (all start EMPTY).
"""

import numpy as np
from . import model_init

DEFAULT_START_GRID_XY = (1, 2)
CHECKBOX_FACTORS = ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered")


def build_D(
    self_start_pos: int | None = None,
    self_start_ori: int = 0,
) -> dict[str, np.ndarray]:

    if self_start_pos is None:
        gx, gy = DEFAULT_START_GRID_XY
        start_grid = model_init.xy_to_index(gx, gy)
        start_walkable = model_init.grid_idx_to_walkable_idx(start_grid)
        self_start_pos = int(start_walkable) if start_walkable is not None else 0
    else:
        self_start_pos = int(self_start_pos)

    self_start_ori = int(self_start_ori)

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

    # pot_state
    D["pot_state"] = np.zeros(model_init.N_POT_STATES, dtype=float)
    D["pot_state"][model_init.POT_0] = 1.0

    # Counters: start empty (binary)
    for grid_idx in model_init.MODELED_COUNTERS:
        cf = f"ctr_{grid_idx}"
        D[cf] = np.array([1.0, 0.0], dtype=float)  # [EMPTY, FULL]

    # Checkboxes: all start at 0 (unchecked)
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
    )