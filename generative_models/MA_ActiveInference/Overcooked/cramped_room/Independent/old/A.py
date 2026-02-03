# A.py
"""
Observation model (A) for Independent paradigm — Cramped Room.

- Known map (no map latent factor).
- observation: front_tile_type computed from model_init.compute_front_tile_type(...).
- soup_delivered is OBSERVATION-ONLY (event), NOT a hidden state factor.
"""

import numpy as np
from . import model_init

A_NOISE_LEVEL = 0.01  # optional noise for position observations


def _noisy_categorical(idx: int, n: int, noise_level: float = A_NOISE_LEVEL) -> np.ndarray:
    p = np.full(n, noise_level / max(1, (n - 1)), dtype=float)
    if 0 <= idx < n:
        p[idx] = 1.0 - noise_level
    return p


def _one_hot(idx: int, n: int) -> np.ndarray:
    p = np.zeros(n, dtype=float)
    if 0 <= idx < n:
        p[idx] = 1.0
    return p


def A_fn(state_indices: dict) -> dict[str, np.ndarray]:
    """
    Hidden-state keys expected in state_indices:
      - agent_pos
      - agent_orientation
      - agent_held
      - other_agent_pos
      - pot_state (optional; defaults to POT_0)

    Observation-only event key optionally present:
      - soup_delivered (0/1)  (1 only on the timestep delivery occurs)
    """
    agent_pos = int(state_indices["agent_pos"])
    agent_ori = int(state_indices["agent_orientation"])
    agent_held = int(state_indices["agent_held"])
    other_pos = int(state_indices["other_agent_pos"])
    pot_state = int(state_indices.get("pot_state", model_init.POT_0))

    soup_delivered = int(state_indices.get("soup_delivered", 0))
    soup_delivered = 0 if soup_delivered <= 0 else 1

    obs: dict[str, np.ndarray] = {}

    obs["agent_pos"] = _noisy_categorical(agent_pos, model_init.GRID_SIZE)
    obs["other_agent_pos"] = _noisy_categorical(other_pos, model_init.GRID_SIZE)

    obs["agent_orientation"] = _one_hot(agent_ori, model_init.N_DIRECTIONS)
    obs["agent_held"] = _one_hot(agent_held, model_init.N_HELD_TYPES)
    obs["pot_state"] = _one_hot(pot_state, model_init.N_POT_STATES)

    front_type = model_init.compute_front_tile_type(agent_pos, agent_ori, other_pos)
    obs["front_tile_type"] = _one_hot(front_type, model_init.N_FRONT_TYPES)

    # observation-only event
    obs["soup_delivered"] = _one_hot(soup_delivered, 2)

    return obs
