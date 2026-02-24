# A.py
"""
Observation model (A) for Independent paradigm — Cramped Room (Stage 1, checkbox model).

Each observation modality is p(observation | state). A_fn takes specific state indices
(not belief distributions) and returns a dict of observation likelihood vectors keyed
by model_init.observations.

State-factor dependencies (from model_init.observation_state_dependencies):
- agent_pos_obs         depends on: agent_pos
- agent_orientation_obs depends on: agent_orientation
- agent_held_obs        depends on: agent_held
- pot_state_obs         depends on: pot_state
- soup_delivered_obs    depends on: agent_pos, agent_orientation, agent_held
  (and observation-only event soup_delivered 0/1; the event is not a state factor)

IMPORTANT CONVENTIONS (aligned with RedBlueButton A):
- Observation keys must exactly match model_init.observations.
- Binary soup_delivered_obs uses [no_event, event]: index 0 = no delivery, 1 = delivery.
- All returned arrays are float probability vectors (sum to 1).
- agent_pos is walkable index 0..N_WALKABLE-1 (6 positions); not grid index.

DESIGN:
- soup_delivered is OBSERVATION-ONLY (event), NOT a hidden state factor; it is passed
  in state_indices when the event occurs (e.g. from env reward_info).
- No other_agent_pos; no front_tile_type (per model_init.observations).
- Checkbox state (ck_put1, ...) may appear in state_indices but A does not use it.
"""

import numpy as np
from . import model_init

# Noise for position observations (same role as RedBlueButton A_NOISE_LEVEL)
A_NOISE_LEVEL = 0.1


def _noisy_categorical(idx: int, n: int, noise_level: float = A_NOISE_LEVEL) -> np.ndarray:
    """
    p(observe each of n outcomes | true index is idx).
    Off-diagonal mass = noise_level / (n-1), diagonal = 1 - noise_level.
    Returns float array of length n that sums to 1.
    """
    if n <= 0:
        return np.array([], dtype=float)
    p = np.full(n, noise_level / max(1, n - 1), dtype=float)
    if 0 <= idx < n:
        p[idx] = 1.0 - noise_level
    return p


def _one_hot(idx: int, n: int) -> np.ndarray:
    """
    Deterministic observation: p(observe k | true index idx) = 1 if k==idx else 0.
    Returns float array of length n.
    """
    p = np.zeros(n, dtype=float)
    if 0 <= idx < n:
        p[idx] = 1.0
    return p


def A_fn(state_indices: dict) -> dict[str, np.ndarray]:
    """
    Observation likelihoods p(o | state) for all modalities in model_init.observations.
    Each observation is computed from exactly the state factors in
    model_init.observation_state_dependencies[modality].

    Parameters
    ----------
    state_indices : dict
        Must contain all keys required by observation_state_dependencies:
          - agent_pos         : int, walkable index 0..5 (N_WALKABLE)
          - agent_orientation : int, 0..3 (N_DIRECTIONS)
          - agent_held        : int, 0..3 (N_HELD_TYPES)
        Optional:
          - pot_state         : int, 0..3 (default POT_0)
          - soup_delivered    : int, 0 or 1 (observation-only event; default 0)
        Ignored if present: ck_put1, ck_put2, ck_put3, ck_plated, ck_delivered.

    Returns
    -------
    obs : dict[str, np.ndarray]
        Keys exactly model_init.observations. Each value is a 1D float array
        (probability distribution) with length len(model_init.observations[key]).
    """
    # Optional state-factor defaults (required deps must be in state_indices)
    _optional_defaults = {"pot_state": model_init.POT_0}
    # Observation-only event (not a state factor): 0 = no delivery, 1 = delivery this step
    soup_delivered = 1 if int(state_indices.get("soup_delivered", 0)) > 0 else 0

    obs: dict[str, np.ndarray] = {}

    for modality, deps in model_init.observation_state_dependencies.items():
        n = len(model_init.observations[modality])
        dep_vals = {}
        for d in deps:
            if d in _optional_defaults:
                dep_vals[d] = int(state_indices.get(d, _optional_defaults[d]))
            else:
                dep_vals[d] = int(state_indices[d])

        if modality == "agent_pos_obs":
            # Deps: [agent_pos]
            obs[modality] = _noisy_categorical(dep_vals["agent_pos"], n, A_NOISE_LEVEL)
        elif modality == "agent_orientation_obs":
            # Deps: [agent_orientation]
            obs[modality] = _noisy_categorical(dep_vals["agent_orientation"], n, A_NOISE_LEVEL)
        elif modality == "agent_held_obs":
            # Deps: [agent_held]
            obs[modality] = _noisy_categorical(dep_vals["agent_held"], n, A_NOISE_LEVEL)
        elif modality == "pot_state_obs":
            # Deps: [pot_state]
            pot = dep_vals["pot_state"]
            obs[modality] = _noisy_categorical(pot, n, A_NOISE_LEVEL)
        elif modality == "soup_delivered_obs":
            # Deps: [agent_pos, agent_orientation, agent_held]; also event soup_delivered
            # Likelihood is determined by the event flag; pos/ori/held are the formal state deps
            # (e.g. for factorised belief propagation). Here p(obs | pos, ori, held, event)
            # is a noisy-categorical centered on the event value.
            obs[modality] = _noisy_categorical(soup_delivered, n, A_NOISE_LEVEL)
        else:
            raise KeyError("A_fn: unknown modality {}".format(modality))

    return obs
