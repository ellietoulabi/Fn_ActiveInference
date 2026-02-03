"""
D.py

Prior over initial hidden states (D) for Independent paradigm — Cramped Room (Stage 1).

- Known map, fixed start positions.
- Priors can be sharp (delta) or slightly uncertain via noise.
"""

import numpy as np
from . import model_init


def normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    return p / max(s, 1e-8)


def delta_with_noise(idx: int, n: int, noise: float = 0.0) -> np.ndarray:
    """
    Delta distribution at idx with optional symmetric noise.
    noise=0.0 -> pure delta.
    noise in (0,1) -> (1-noise) at idx, noise spread uniformly across all states.
    """
    p = np.zeros(n, dtype=float)
    if not (0 <= idx < n):
        # fallback: uniform
        return np.ones(n, dtype=float) / max(1, n)

    if noise <= 0.0:
        p[idx] = 1.0
        return p

    noise = min(max(float(noise), 0.0), 1.0)
    p[:] = noise / max(1, n)
    p[idx] += 1.0 - noise
    return normalize(p)


def uniform(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) / max(1, n)


def D_fn(
    agent_id: int = 1,
    pos_noise: float = 0.0,
    other_pos_noise: float = 0.0,
    ori_uniform: bool = True,
    fixed_orientation: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Build priors over hidden states.

    Args:
        agent_id: 1 or 2 (matches your grid text "O1  O" and "X  2X")
                 agent 1 starts at index 6 (x=1,y=1)
                 agent 2 starts at index 13 (x=3,y=2)
        pos_noise: noise on our own start position prior (0.0 = certain)
        other_pos_noise: noise on other agent's start position prior
        ori_uniform: if True, prior over orientation is uniform
        fixed_orientation: if not None and ori_uniform is False, make a delta prior at this ori index

    Returns:
        dict factor -> probability vector
    """
    # Start positions from your docstring / earlier notes:
    # agent 1: (1,1) -> idx 6
    # agent 2: (3,2) -> idx 13
    START_POS_AGENT_1 = 6
    START_POS_AGENT_2 = 13

    if agent_id == 1:
        my_start = START_POS_AGENT_1
        other_start = START_POS_AGENT_2
    else:
        my_start = START_POS_AGENT_2
        other_start = START_POS_AGENT_1

    D = {}

    # agent_pos
    D["agent_pos"] = delta_with_noise(my_start, model_init.GRID_SIZE, noise=pos_noise)

    # agent_orientation
    if ori_uniform or fixed_orientation is None:
        D["agent_orientation"] = uniform(model_init.N_DIRECTIONS)
    else:
        D["agent_orientation"] = delta_with_noise(int(fixed_orientation), model_init.N_DIRECTIONS, noise=0.0)

    # agent_held (start empty)
    D["agent_held"] = delta_with_noise(model_init.HELD_NONE, model_init.N_HELD_TYPES, noise=0.0)

    # pot_state (start empty pot: POT_0)
    D["pot_state"] = delta_with_noise(model_init.POT_0, model_init.N_POT_STATES, noise=0.0)

    # other_agent_pos (environment factor; can be certain or slightly uncertain)
    D["other_agent_pos"] = delta_with_noise(other_start, model_init.GRID_SIZE, noise=other_pos_noise)

    return D
