# A.py
"""
Observation likelihoods p(o | state) model (A) for Independent paradigm — Cramped Room.

Includes:
- self position / orientation / held observations
- other agent position / orientation / held observations
- pot state observation
- soup delivered observation (event pulse: delivered this step)
- counter occupancy observations for modeled counters
"""

import numpy as np
from . import model_init

A_NOISE_LEVEL = 0.01

# Per-checkpoint penalty applied when ck_delivered=1 but a put-checkbox is still at 1.
# Each unresolved checkbox multiplies the delivery likelihood by this factor, giving an
# independent signal for each factor.
#
# Constraint: FACTOR^4 * (1-NOISE) > NOISE  →  FACTOR > (NOISE/(1-NOISE))^0.25 ≈ 0.32
# This ensures P(soup_del=1 | ck_del=1, all_puts_stuck) > P(soup_del=1 | ck_del=0),
# so the ck_delivered update fires in the right direction even before any put resets.
# With FACTOR=0.4: worst case is 0.99*0.4^4 = 0.025 >> NOISE=0.01.  ✓
DELIVERY_RESET_FACTOR = 0.4
POT_CHECKBOX_MISMATCH_PENALTY = 0.20
POT_MIN_CORRECT_OBS_PROB = 0.35


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


def A_self_held_obs(self_held: int, ck_plated: int = 0) -> np.ndarray:
    n = len(model_init.observations["self_held_obs"])
    idx = int(self_held)
    # Seeing soup in hand is strong evidence that plating has happened.
    mismatch = int(idx == model_init.HELD_SOUP and int(ck_plated) == 0)
    p_correct = (1.0 - A_NOISE_LEVEL) * (0.10 if mismatch else 1.0)
    p_correct = float(np.clip(p_correct, 1e-8, 1.0 - 1e-8))
    p = np.full(n, (1.0 - p_correct) / max(1, n - 1), dtype=float)
    if 0 <= idx < n:
        p[idx] = p_correct
    return p


def A_other_pos_obs(other_pos: int) -> np.ndarray:
    n = len(model_init.observations["other_pos_obs"])
    return _noisy_categorical(int(other_pos), n, A_NOISE_LEVEL)


def A_other_orientation_obs(other_orientation: int) -> np.ndarray:
    n = len(model_init.observations["other_orientation_obs"])
    return _noisy_categorical(int(other_orientation), n, A_NOISE_LEVEL)


def A_other_held_obs(other_held: int, ck_plated: int = 0) -> np.ndarray:
    n = len(model_init.observations["other_held_obs"])
    idx = int(other_held)
    mismatch = int(idx == model_init.HELD_SOUP and int(ck_plated) == 0)
    p_correct = (1.0 - A_NOISE_LEVEL) * (0.10 if mismatch else 1.0)
    p_correct = float(np.clip(p_correct, 1e-8, 1.0 - 1e-8))
    p = np.full(n, (1.0 - p_correct) / max(1, n - 1), dtype=float)
    if 0 <= idx < n:
        p[idx] = p_correct
    return p


def A_pot_state_obs(
    pot_state: int,
    ck_put1: int = 0,
    ck_put2: int = 0,
    ck_put3: int = 0,
) -> np.ndarray:
    """
    Likelihood of pot_state observation.

    Primary signal: observation reflects latent pot_state.
    Consistency signal: ck_put1/2/3 must match deterministic thresholds implied
    by pot_state (>=1, >=2, >=3 onions). Mismatches reduce likelihood, enabling
    inference of checkpoint factors from pot observation alone.
    """
    n = len(model_init.observations["pot_state_obs"])
    idx = int(pot_state)
    p_state = int(pot_state)
    expected_ck1 = 1 if p_state >= model_init.POT_1 else 0
    expected_ck2 = 1 if p_state >= model_init.POT_2 else 0
    expected_ck3 = 1 if p_state >= model_init.POT_3 else 0

    mismatches = 0
    mismatches += int(int(ck_put1) != expected_ck1)
    mismatches += int(int(ck_put2) != expected_ck2)
    mismatches += int(int(ck_put3) != expected_ck3)

    # Keep pot observation dominant, while still using checkpoint consistency.
    #
    # Important: if this penalty is too strong, incorrect latent pot states can
    # become more likely than the correct one for a given observed pot index
    # (because their off-diagonal mass grows). We therefore use a bounded linear
    # penalty and a floor on p_correct.
    p_correct = (1.0 - A_NOISE_LEVEL) * (1.0 - POT_CHECKBOX_MISMATCH_PENALTY * mismatches)
    p_correct = float(np.clip(p_correct, POT_MIN_CORRECT_OBS_PROB, 1.0 - 1e-8))
    p = np.full(n, (1.0 - p_correct) / max(1, n - 1), dtype=float)
    if 0 <= idx < n:
        p[idx] = p_correct
    return p


def A_soup_delivered_obs(
    ck_delivered: int,
    ck_put1: int = 0,
    ck_put2: int = 0,
    ck_put3: int = 0,
    ck_plated: int = 0,
) -> np.ndarray:
    """
    Likelihood of soup_delivered_obs given delivery-related state factors.

    Primary signal: soup_delivered_obs=1 iff ck_delivered=1.

    Reset consistency: after a delivery the put-checkboxes and ck_plated must have
    reset to 0.  Each checkbox still at 1 multiplies the likelihood of obs=1 by
    DELIVERY_RESET_FACTOR independently, creating a strong per-factor gradient
    without a joint floor that would collapse the signal when all are still at 1.

    This enables the inference engine to push each checkpoint factor toward 0
    individually when delivery is observed, even if the others have not yet updated.
    """
    n = len(model_init.observations["soup_delivered_obs"])

    if int(ck_delivered) == 1:
        # Each unresolved checkbox independently penalises the delivery likelihood.
        penalty = 1.0
        for ck in (int(ck_put1), int(ck_put2), int(ck_put3), int(ck_plated)):
            if ck == 1:
                penalty *= DELIVERY_RESET_FACTOR
        p_del = float(np.clip((1.0 - A_NOISE_LEVEL) * penalty, 1e-12, 1.0 - 1e-12))
    else:
        p_del = A_NOISE_LEVEL

    p = np.zeros(n, dtype=float)
    p[0] = 1.0 - p_del
    if n > 1:
        p[1] = p_del
    return p


def A_counter_occ_obs(counter_occ: int, factor_name: str) -> np.ndarray:
    """
    Counter contents observation for factor ctr_<grid>:
      0 empty, 1 onion, 2 dish, 3 soup
    """
    n = len(model_init.observations[f"{factor_name}_obs"])
    return _noisy_categorical(int(counter_occ), n, 0)


def A_fn(state_indices: dict) -> dict[str, np.ndarray]:
    pot_state = int(state_indices.get("pot_state", model_init.POT_0))

    obs: dict[str, np.ndarray] = {}

    obs["self_pos_obs"] = A_self_pos_obs(state_indices["self_pos"])
    obs["self_orientation_obs"] = A_self_orientation_obs(state_indices["self_orientation"])
    obs["self_held_obs"] = A_self_held_obs(
        state_indices["self_held"],
        ck_plated=int(state_indices.get("ck_plated", 0)),
    )

    obs["other_pos_obs"] = A_other_pos_obs(state_indices["other_pos"])
    obs["other_orientation_obs"] = A_other_orientation_obs(state_indices["other_orientation"])
    obs["other_held_obs"] = A_other_held_obs(
        state_indices["other_held"],
        ck_plated=int(state_indices.get("ck_plated", 0)),
    )

    obs["pot_state_obs"] = A_pot_state_obs(
        pot_state,
        ck_put1=int(state_indices.get("ck_put1", 0)),
        ck_put2=int(state_indices.get("ck_put2", 0)),
        ck_put3=int(state_indices.get("ck_put3", 0)),
    )

    obs["soup_delivered_obs"] = A_soup_delivered_obs(
        ck_delivered=int(state_indices.get("ck_delivered", 0)),
        ck_put1=int(state_indices.get("ck_put1", 0)),
        ck_put2=int(state_indices.get("ck_put2", 0)),
        ck_put3=int(state_indices.get("ck_put3", 0)),
        ck_plated=int(state_indices.get("ck_plated", 0)),
    )
    for grid_idx in model_init.MODELED_COUNTERS:
        factor = f"ctr_{grid_idx}"
        obs[f"{factor}_obs"] = A_counter_occ_obs(state_indices[factor], factor)

    return obs