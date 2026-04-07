# A.py
"""
Observation likelihoods p(o | state) model (A) for IndividuallyCollective paradigm — Cramped Room.

Includes:
- self position / orientation / held observations
- other agent position / orientation / held observations
- pot state observation
- soup delivered observation (event pulse: delivered this step)
- counter occupancy observations for modeled counters
"""

import numpy as np
from . import model_init

# A_NOISE_LEVEL = 0.001
A_NOISE_LEVEL = 0.001


def _noisy_categorical(idx: int, n: int, noise_level: float = A_NOISE_LEVEL) -> np.ndarray:
    """Return a noisy categorical centered on `idx` over `n` outcomes."""
    if n <= 0:
        return np.array([], dtype=float)
    p = np.full(n, noise_level / max(1, n - 1), dtype=float)
    if 0 <= idx < n:
        p[idx] = 1.0 - noise_level
    return p


def A_self_pos_obs(self_pos: int) -> np.ndarray:
    n = len(model_init.observations["self_pos_obs"])
    return _noisy_categorical(int(self_pos), n, A_NOISE_LEVEL)


def A_self_orientation_obs(self_orientation: int) -> np.ndarray:
    n = len(model_init.observations["self_orientation_obs"])
    return _noisy_categorical(int(self_orientation), n, A_NOISE_LEVEL)


def A_self_held_obs(self_held: int) -> np.ndarray:
    n = len(model_init.observations["self_held_obs"])
    return _noisy_categorical(int(self_held), n, A_NOISE_LEVEL)


def A_other_pos_obs(other_pos: int) -> np.ndarray:
    n = len(model_init.observations["other_pos_obs"])
    return _noisy_categorical(int(other_pos), n, A_NOISE_LEVEL)


def A_other_orientation_obs(other_orientation: int) -> np.ndarray:
    n = len(model_init.observations["other_orientation_obs"])
    return _noisy_categorical(int(other_orientation), n, A_NOISE_LEVEL)


def A_other_held_obs(other_held: int) -> np.ndarray:
    n = len(model_init.observations["other_held_obs"])
    return _noisy_categorical(int(other_held), n, A_NOISE_LEVEL)


def A_pot_state_obs(pot_state: int) -> np.ndarray:
    n = len(model_init.observations["pot_state_obs"])
    return _noisy_categorical(int(pot_state), n, A_NOISE_LEVEL)


def A_soup_delivered_obs(soup_delivered_event: int) -> np.ndarray:
    """
    Event semantics:
      soup_delivered_obs = 1 iff ck_delivered = 1 for the current step only.
    """
    n = len(model_init.observations["soup_delivered_obs"])
    idx = 1 if int(soup_delivered_event) > 0 else 0
    return _noisy_categorical(idx, n, A_NOISE_LEVEL)


def A_counter_occ_obs(counter_occ: int, factor_name: str) -> np.ndarray:
    """
    Counter contents observation for factor ctr_<grid>:
      0 empty, 1 onion, 2 dish, 3 soup
    """
    n = len(model_init.observations[f"{factor_name}_obs"])
    return _noisy_categorical(int(counter_occ), n, 0)


def A_fn(state_indices: dict) -> dict[str, np.ndarray]:
    pot_state = int(state_indices.get("pot_state", model_init.POT_0))
    delivered_event = int(state_indices.get("ck_delivered", 0))

    obs: dict[str, np.ndarray] = {}

    obs["self_pos_obs"] = A_self_pos_obs(state_indices["self_pos"])
    obs["self_orientation_obs"] = A_self_orientation_obs(state_indices["self_orientation"])
    obs["self_held_obs"] = A_self_held_obs(state_indices["self_held"])

    obs["other_pos_obs"] = A_other_pos_obs(state_indices["other_pos"])
    obs["other_orientation_obs"] = A_other_orientation_obs(state_indices["other_orientation"])
    obs["other_held_obs"] = A_other_held_obs(state_indices["other_held"])

    obs["pot_state_obs"] = A_pot_state_obs(pot_state)
    obs["soup_delivered_obs"] = A_soup_delivered_obs(delivered_event)

    for grid_idx in model_init.MODELED_COUNTERS:
        factor = f"ctr_{grid_idx}"
        obs[f"{factor}_obs"] = A_counter_occ_obs(state_indices[factor], factor)

    return obs