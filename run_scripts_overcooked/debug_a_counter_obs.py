"""
Debug runner for the 5 counter observation modalities in Independent/A.py:
  A_counter_0_obs .. A_counter_4_obs

Run from repo root:
    python3 run_scripts_overcooked/debug_a_counter_obs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent import (  # noqa: E402
    model_init,
)
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent.A import (  # noqa: E402
    A_counter_0_obs,
    A_counter_1_obs,
    A_counter_2_obs,
    A_counter_3_obs,
    A_counter_4_obs,
)


def _base_state() -> dict:
    """Minimal full-state dict (extra keys are fine)."""
    s = {
        "agent_pos": 0,
        "agent_orientation": model_init.NORTH,
        "agent_held": model_init.HELD_NONE,
        "pot_state": model_init.POT_0,
        "ck_put1": 0,
        "ck_put2": 0,
        "ck_put3": 0,
        "ck_plated": 0,
        "ck_delivered": 0,
        "counter_0": model_init.HELD_NONE,
        "counter_1": model_init.HELD_NONE,
        "counter_2": model_init.HELD_NONE,
        "counter_3": model_init.HELD_NONE,
        "counter_4": model_init.HELD_NONE,
    }
    return s


def main() -> None:
    # Try a few representative counter contents
    values = [
        ("NONE", model_init.HELD_NONE),
        ("ONION", model_init.HELD_ONION),
        ("DISH", model_init.HELD_DISH),
        ("SOUP", model_init.HELD_SOUP),
    ]

    fns = [
        ("counter_0_obs", A_counter_0_obs, "counter_0"),
        ("counter_1_obs", A_counter_1_obs, "counter_1"),
        ("counter_2_obs", A_counter_2_obs, "counter_2"),
        ("counter_3_obs", A_counter_3_obs, "counter_3"),
        ("counter_4_obs", A_counter_4_obs, "counter_4"),
    ]

    print("Testing A_counter_*_obs functions")
    print("Held types: 0=NONE, 1=ONION, 2=DISH, 3=SOUP")

    for label, v in values:
        print(f"\n--- Setting ALL counters = {label} ({v}) ---")
        s = _base_state()
        for k in ("counter_0", "counter_1", "counter_2", "counter_3", "counter_4"):
            s[k] = int(v)

        for obs_name, fn, state_key in fns:
            dist = np.array(fn(s), dtype=float)
            print(
                f"{obs_name:<12} | {state_key}={s[state_key]} -> "
                f"dist={np.round(dist, 6)} sum={dist.sum():.6f} argmax={int(np.argmax(dist))}"
            )


if __name__ == "__main__":
    main()

