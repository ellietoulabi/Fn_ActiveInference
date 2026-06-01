# D.py
"""
Prior beliefs (D) for FullyCollective paradigm — Cramped Room.

Identical to the IC priors: the IC brain holds priors over both its own
(self_*) and the puppet's (other_*) starting pose and held object, plus the
pot, checkboxes, and modeled counters.

Includes:
- self position / orientation / held
- other agent position / orientation / held
- pot state
- checkbox factors
- binary counter occupancy factors (all start EMPTY)
"""

import numpy as np
from . import model_init

CHECKBOX_FACTORS = ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered")


def build_D(
    self_start_pos: int | None = None,
    self_start_ori: int = 0,
    other_start_pos: int | None = None,
    other_start_ori: int = 0,
) -> dict[str, np.ndarray]:
    """
    Build the FC brain's prior beliefs.

    When start positions are provided (via config), `self_pos` and `other_pos`
    are point-masses at those positions.  When called with no config (``D_fn(None)``),
    positions are left **uniform** so the brain infers them from the first
    observation rather than starting with a hard-coded default.

    The uniform prior is what makes `infer_states` leave a small residual
    entropy on `self_pos` / `other_pos` (combined with the slightly noisy A
    likelihood), which in turn pushes those factors above the IG "dynamic
    factor" entropy threshold so joint pairs get distinct EFE.
    """

    self_start_ori = int(self_start_ori)
    other_start_ori = int(other_start_ori)

    D: dict[str, np.ndarray] = {}

    # self_pos — point-mass when known, uniform otherwise
    if self_start_pos is not None and 0 <= int(self_start_pos) < model_init.N_WALKABLE:
        D["self_pos"] = np.zeros(model_init.N_WALKABLE, dtype=float)
        D["self_pos"][int(self_start_pos)] = 1.0
    else:
        D["self_pos"] = np.ones(model_init.N_WALKABLE, dtype=float) / model_init.N_WALKABLE

    # self_orientation
    D["self_orientation"] = np.zeros(model_init.N_DIRECTIONS, dtype=float)
    ori_idx = self_start_ori if 0 <= self_start_ori < model_init.N_DIRECTIONS else 0
    D["self_orientation"][ori_idx] = 1.0

    # self_held
    D["self_held"] = np.zeros(model_init.N_HELD_TYPES, dtype=float)
    D["self_held"][model_init.HELD_NONE] = 1.0

    # other_pos — point-mass when known, uniform otherwise
    if other_start_pos is not None and 0 <= int(other_start_pos) < model_init.N_WALKABLE:
        D["other_pos"] = np.zeros(model_init.N_WALKABLE, dtype=float)
        D["other_pos"][int(other_start_pos)] = 1.0
    else:
        D["other_pos"] = np.ones(model_init.N_WALKABLE, dtype=float) / model_init.N_WALKABLE

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