"""
C.py

Preference model (C) for Independent paradigm — Cramped Room (Stage 1, sparse).

Sparse objective:
- Prefer observing soup_delivered = 1 (delivery event).
- No other preferences (all other observation modalities are uniform).

Interpretation:
- delivery event corresponds to environment reward +20.
- This is *not* shaping: no intermediate progress preferences.
"""

import numpy as np
from . import model_init


def normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    return p / max(s, 1e-8)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / max(float(np.sum(ex)), 1e-8)


def C_fn(
    delivery_reward: float = 20.0,
    delivery_prior_odds: float | None = None,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Build preferences over observations.

    Args:
        delivery_reward:
            Your env gives +20 for delivery; we map that to preference strength.
            Larger => stronger preference for soup_delivered=1.
        delivery_prior_odds:
            Optional alternative to control preference strength directly.
            If set, defines odds = P(delivered=1) / P(delivered=0).
            Example: odds=99 => P(1)=0.99, P(0)=0.01
        temperature:
            Softmax temperature for the log-preferences construction.
            Higher => softer preferences. Default 1.0.

    Returns:
        dict modality -> probability vector (categorical preferences)
    """
    C = {}

    # -------------------------------------------------
    # Flat (uninformative) preferences for all modalities
    # -------------------------------------------------
    for name, space in model_init.observations.items():
        n = len(space)
        C[name] = np.ones(n, dtype=float) / max(1, n)

    # -------------------------------------------------
    # Sparse preference ONLY on soup_delivered (event obs)
    # soup_delivered ∈ {0,1}
    # -------------------------------------------------
    if "soup_delivered" in C:
        if delivery_prior_odds is not None:
            # Directly specify odds for delivered vs not delivered
            odds = max(float(delivery_prior_odds), 1e-8)
            p1 = odds / (1.0 + odds)
            p0 = 1.0 - p1
            C["soup_delivered"] = np.array([p0, p1], dtype=float)
        else:
            # Log-preferences: prefer delivered=1 by +delivery_reward units
            # then convert to probabilities by softmax.
            # (This is a standard AIF construction: C as a softmax of utilities.)
            logits = np.array([0.0, float(delivery_reward)], dtype=float) / max(float(temperature), 1e-8)
            C["soup_delivered"] = softmax(logits)

    return C
