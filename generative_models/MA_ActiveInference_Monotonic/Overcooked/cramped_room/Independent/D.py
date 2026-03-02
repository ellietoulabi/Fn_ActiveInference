"""
Prior beliefs (D) for Independent Monotonic paradigm (single-agent perspective) - Cramped Room.

This version matches the 6-walkable-position, checkbox-based model in model_init.py:

State factors (model_init.states):
  - agent_pos         : walkable index 0..5
  - agent_orientation : 0..3  (NORTH,SOUTH,EAST,WEST)
  - agent_held        : {NONE, ONION, DISH, SOUP}
  - pot_state         : POT_0..POT_3   (4 states, cook_time=0)
  - ck_put1, ck_put2, ck_put3, ck_plated, ck_delivered : binary memory (0/1)
  - counter_0..counter_4 : {NONE, ONION, DISH, SOUP} (counter slot contents)

Defaults:
  - agent_pos         : start at agent 1's layout position (1,1) mapped into walkable index
  - agent_orientation : NORTH (0)
  - agent_held        : HELD_NONE
  - pot_state         : POT_0
  - all checkboxes    : 0 (no progress yet)
  - all counters      : HELD_NONE (empty)
"""

import numpy as np
from . import model_init


def build_D(agent_start_pos=None, agent_start_ori: int = 0):
    """
    Build prior D over all hidden state factors for a single agent.

    Parameters
    ----------
    agent_start_pos : int or None
        Agent start position in walkable index space (0..N_WALKABLE-1).
        If None, use layout start at (1,1) mapped via grid_idx_to_walkable_idx.
    agent_start_ori : int
        Initial orientation index (default NORTH=0).

    Returns
    -------
    D : dict[str, np.ndarray]
        Prior belief over all state factors defined in model_init.states.
    """
    # Resolve default start position in walkable index space
    if agent_start_pos is None:
        start_grid = model_init.xy_to_index(1, 1)  # Agent 1 start in cramped_room
        start_walkable = model_init.grid_idx_to_walkable_idx(start_grid)
        agent_start_pos = 0 if start_walkable is None else int(start_walkable)
    else:
        agent_start_pos = int(agent_start_pos)

    D: dict[str, np.ndarray] = {}

    # agent_pos (walkable 0..N_WALKABLE-1)
    D["agent_pos"] = np.zeros(model_init.N_WALKABLE, dtype=float)
    if 0 <= agent_start_pos < model_init.N_WALKABLE:
        D["agent_pos"][agent_start_pos] = 1.0
    else:
        D["agent_pos"][0] = 1.0

    # agent_orientation
    D["agent_orientation"] = np.zeros(model_init.N_DIRECTIONS, dtype=float)
    agent_start_ori = int(agent_start_ori)
    if 0 <= agent_start_ori < model_init.N_DIRECTIONS:
        D["agent_orientation"][agent_start_ori] = 1.0
    else:
        D["agent_orientation"][0] = 1.0

    # agent_held
    D["agent_held"] = np.zeros(model_init.N_HELD_TYPES, dtype=float)
    D["agent_held"][model_init.HELD_NONE] = 1.0

    # pot_state
    D["pot_state"] = np.zeros(model_init.N_POT_STATES, dtype=float)
    D["pot_state"][model_init.POT_0] = 1.0

    # Checkbox memory factors: all start at 0 (unchecked / no progress)
    for ck in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
        D[ck] = np.zeros(2, dtype=float)
        D[ck][0] = 1.0

    # Counter slots: assume empty at start
    for i in range(getattr(model_init, "N_COUNTERS", 5)):
        key = f"counter_{i}"
        D[key] = np.zeros(model_init.N_HELD_TYPES, dtype=float)
        D[key][model_init.HELD_NONE] = 1.0

    # Safety: ensure we return a prior for every declared state factor.
    # If a new factor is added to model_init.states and not handled above,
    # default to a uniform prior rather than crashing downstream.
    for factor, values in model_init.states.items():
        if factor in D:
            continue
        n = len(values)
        if n <= 0:
            D[factor] = np.array([], dtype=float)
        else:
            D[factor] = np.ones(n, dtype=float) / float(n)

    return D


def D_fn(config=None):
    """
    Entry point used by agents/control: D_fn(config) -> prior beliefs dict.

    config : dict or None
        Optional configuration with keys:
          - agent_start_pos : walkable index (0..5)
          - agent_start_ori : orientation index (0..3)
        Extra keys are ignored.
    """
    if config is None:
        return build_D()

    return build_D(
        agent_start_pos=config.get("agent_start_pos", None),
        agent_start_ori=config.get("agent_start_ori", 0),
    )

