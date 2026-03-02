# A.py
"""
Observation likelihood model (A = P (o | s)) for Independent paradigm — Cramped Room (monotonic model).
"""

import numpy as np
from . import model_init

# Noise for position observations (same role as RedBlueButton A_NOISE_LEVEL)
A_NOISE_LEVEL = 0.03
# Counters are observable but not perfectly (avoid absolute certainty)
COUNTER_OBS_NOISE_LEVEL = 0.1


def _noisy_categorical(idx: int, n: int, noise_level: float = A_NOISE_LEVEL) -> np.ndarray:
    """
    p(observe each of n outcomes | true state is idx).
    Off-diagonal mass = noise_level / (n-1), diagonal = 1 - noise_level.
    Returns float array of length n that sums to 1.
    """
    if n <= 0:
        return np.array([], dtype=float)
    p = np.full(n, noise_level / max(1, n - 1), dtype=float)
    if 0 <= idx < n:
        p[idx] = 1.0 - noise_level
    # If idx is ever out-of-range, p would not sum to 1; normalize.
    s = float(p.sum())
    return p / max(s, 1e-12)




def A_agent_pos_obs(state_indices: dict) -> np.ndarray:
    """p(agent_pos_obs | agent_pos)."""
    n = len(model_init.observations["agent_pos_obs"])
    return _noisy_categorical(int(state_indices["agent_pos"]), n, A_NOISE_LEVEL)


def A_agent_orientation_obs(state_indices: dict) -> np.ndarray:
    """p(agent_orientation_obs | agent_orientation)."""
    n = len(model_init.observations["agent_orientation_obs"])
    return _noisy_categorical(int(state_indices["agent_orientation"]), n, A_NOISE_LEVEL)


def A_agent_held_obs(state_indices: dict) -> np.ndarray:
    """p(agent_held_obs | agent_held)."""
    n = len(model_init.observations["agent_held_obs"])
    return _noisy_categorical(int(state_indices["agent_held"]), n, A_NOISE_LEVEL)


def A_pot_state_obs(state_indices: dict) -> np.ndarray:
    """p(pot_state_obs | pot_state)."""
    pot = int(state_indices.get("pot_state", model_init.POT_0))
    n = len(model_init.observations["pot_state_obs"])
    return _noisy_categorical(pot, n, A_NOISE_LEVEL)


def A_soup_delivered_obs(state_indices: dict) -> np.ndarray:
    """
    p(soup_delivered_obs | agent_pos, agent_orientation, agent_held).
    """
    agent_pos = int(state_indices["agent_pos"])
    agent_orientation = int(state_indices["agent_orientation"])
    agent_held = int(state_indices["agent_held"])

    delivered = 1 if (
        agent_pos == 5
        and agent_orientation == model_init.SOUTH
        and agent_held == model_init.HELD_SOUP
    ) else 0
    n = len(model_init.observations["soup_delivered_obs"])
    return _noisy_categorical(delivered, n, 0.7)

def A_counter_0_obs(state_indices: dict) -> np.ndarray:
    """p(counter_0_obs | counter_0)."""
    n = len(model_init.observations["counter_0_obs"])
    s = int(state_indices.get("counter_0", model_init.HELD_NONE))
    return _noisy_categorical(s, n, COUNTER_OBS_NOISE_LEVEL)


def A_counter_1_obs(state_indices: dict) -> np.ndarray:
    """p(counter_1_obs | counter_1)."""
    n = len(model_init.observations["counter_1_obs"])
    s = int(state_indices.get("counter_1", model_init.HELD_NONE))
    return _noisy_categorical(s, n, COUNTER_OBS_NOISE_LEVEL)


def A_counter_2_obs(state_indices: dict) -> np.ndarray:
    """p(counter_2_obs | counter_2)."""
    n = len(model_init.observations["counter_2_obs"])
    s = int(state_indices.get("counter_2", model_init.HELD_NONE))
    return _noisy_categorical(s, n, COUNTER_OBS_NOISE_LEVEL)


def A_counter_3_obs(state_indices: dict) -> np.ndarray:
    """p(counter_3_obs | counter_3)."""
    n = len(model_init.observations["counter_3_obs"])
    s = int(state_indices.get("counter_3", model_init.HELD_NONE))
    return _noisy_categorical(s, n, COUNTER_OBS_NOISE_LEVEL)


def A_counter_4_obs(state_indices: dict) -> np.ndarray:
    """p(counter_4_obs | counter_4)."""
    n = len(model_init.observations["counter_4_obs"])
    s = int(state_indices.get("counter_4", model_init.HELD_NONE))
    return _noisy_categorical(s, n, COUNTER_OBS_NOISE_LEVEL)

def A_fn(state_indices: dict) -> dict[str, np.ndarray]:
    """
    Observation likelihoods p(o | state) for all modalities in model_init.observations.
    Each modality is computed by its corresponding function above.
    """
    obs = {
        "agent_pos_obs": A_agent_pos_obs(state_indices),
        "agent_orientation_obs": A_agent_orientation_obs(state_indices),
        "agent_held_obs": A_agent_held_obs(state_indices),
        "pot_state_obs": A_pot_state_obs(state_indices),
        "soup_delivered_obs": A_soup_delivered_obs(state_indices),
        "counter_0_obs": A_counter_0_obs(state_indices),
        "counter_1_obs": A_counter_1_obs(state_indices),
        "counter_2_obs": A_counter_2_obs(state_indices),
        "counter_3_obs": A_counter_3_obs(state_indices),
        "counter_4_obs": A_counter_4_obs(state_indices),
    }

    if set(obs.keys()) != set(model_init.observations.keys()):
        missing = set(model_init.observations.keys()) - set(obs.keys())
        extra = set(obs.keys()) - set(model_init.observations.keys())
        raise KeyError(f"A_fn modalities mismatch. missing={sorted(missing)} extra={sorted(extra)}")

    return obs

