"""
Observation likelihoods p(o | state) model (A) for IndividuallyCollective paradigm — Cramped Room (monotonic, single-agent).
"""

import numpy as np
from . import model_init

A_NOISE_LEVEL = 0.1


def _noisy_categorical(idx: int, n: int, noise_level: float = A_NOISE_LEVEL) -> np.ndarray:
    """Return a noisy categorical centered on `idx` over `n` outcomes."""
    if n <= 0:
        return np.array([], dtype=float)
    p = np.full(n, noise_level / max(1, n - 1), dtype=float)
    if 0 <= idx < n:
        p[idx] = 1.0 - noise_level
    return p


def _one_hot(idx: int, n: int) -> np.ndarray:
    """Return a deterministic categorical with all mass on `idx`."""
    p = np.zeros(n, dtype=float)
    if 0 <= idx < n:
        p[idx] = 1.0
    return p


def A_self_pos_obs(self_pos: int) -> np.ndarray:
    """p(self_pos_obs | self_pos). Deterministic over N_WALKABLE."""
    n = len(model_init.observations["self_pos_obs"])
    return _noisy_categorical(int(self_pos), n, A_NOISE_LEVEL)


def A_self_orientation_obs(self_orientation: int) -> np.ndarray:
    """p(self_orientation_obs | self_orientation). Deterministic over N_DIRECTIONS."""
    n = len(model_init.observations["self_orientation_obs"])
    return _noisy_categorical(int(self_orientation), n, A_NOISE_LEVEL)


def A_self_held_obs(self_held: int) -> np.ndarray:
    """p(self_held_obs | self_held). Noisy categorical over held object types."""
    n = len(model_init.observations["self_held_obs"])
    return _noisy_categorical(int(self_held), n, A_NOISE_LEVEL)


def A_pot_state_obs(pot_state: int) -> np.ndarray:
    """p(pot_state_obs | pot_state). Noisy categorical over pot states."""
    n = len(model_init.observations["pot_state_obs"])
    return _noisy_categorical(int(pot_state), n, A_NOISE_LEVEL)


def A_soup_delivered_obs(soup_delivered: int) -> np.ndarray:
    """
    p(soup_delivered_obs | event/ck_delivered).

    soup_delivered is treated as a binary flag:
      0 = no delivery event, 1 = delivery event.
    """
    n = len(model_init.observations["soup_delivered_obs"])
    idx = 1 if int(soup_delivered) > 0 else 0
    return _noisy_categorical(idx, n, A_NOISE_LEVEL)


def A_fn(state_indices: dict) -> dict[str, np.ndarray]:

    pot_state = int(state_indices.get("pot_state", model_init.POT_0))
    soup_delivered = int(state_indices.get("ck_delivered", 0))

    obs: dict[str, np.ndarray] = {}
    obs["self_pos_obs"] = A_self_pos_obs(state_indices["self_pos"])
    obs["self_orientation_obs"] = A_self_orientation_obs(state_indices["self_orientation"])
    obs["self_held_obs"] = A_self_held_obs(state_indices["self_held"])
    obs["pot_state_obs"] = A_pot_state_obs(pot_state)
    obs["soup_delivered_obs"] = A_soup_delivered_obs(soup_delivered)

    return obs


# if __name__ == "__main__":
#     # Simple sanity tests for A functions
#     print("=== Testing A_self_pos_obs ===")
#     p_pos = A_self_pos_obs(0)
#     print("self_pos=0 ->", p_pos, "sum=", float(p_pos.sum()))

#     print("\n=== Testing A_self_orientation_obs ===")
#     p_ori = A_self_orientation_obs(1)
#     print("self_orientation=1 ->", p_ori, "sum=", float(p_ori.sum()))

#     print("\n=== Testing A_self_held_obs ===")
#     p_held = A_self_held_obs(2)
#     print("self_held=2 ->", p_held, "sum=", float(p_held.sum()))

#     print("\n=== Testing A_pot_state_obs ===")
#     p_pot = A_pot_state_obs(3)
#     print("pot_state=3 ->", p_pot, "sum=", float(p_pot.sum()))

#     print("\n=== Testing A_soup_delivered_obs ===")
#     p_sd0 = A_soup_delivered_obs(0)
#     p_sd1 = A_soup_delivered_obs(1)
#     print("soup_delivered=0 ->", p_sd0, "sum=", float(p_sd0.sum()))
#     print("soup_delivered=1 ->", p_sd1, "sum=", float(p_sd1.sum()))

#     print("\n=== Testing A_fn ===")
#     example_state = {
#         "self_pos": 0,
#         "self_orientation": 0,
#         "self_held": 0,
#         "pot_state": model_init.POT_0,
#         "ck_delivered": 0,
#     }
#     obs_all = A_fn(example_state)
#     for k, v in obs_all.items():
#         print(f"{k}: {v}, sum={float(v.sum())}")
