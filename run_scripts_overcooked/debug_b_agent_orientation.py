"""
Quick debug runner for `B_agent_orientation` in the Independent Cramped Room B model.

Run from repo root:
    python3 run_scripts_overcooked/debug_b_agent_orientation.py --noise 0.1
"""

from __future__ import annotations

import argparse
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
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent.B import (  # noqa: E402
    B_agent_orientation,
)


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=float)
    v[i] = 1.0
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug B_agent_orientation transitions.")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Orientation noise level (default: 0.0).",
    )
    args = parser.parse_args()
    noise_level = float(args.noise)

    actions = {
        "NORTH": model_init.NORTH,
        "SOUTH": model_init.SOUTH,
        "EAST": model_init.EAST,
        "WEST": model_init.WEST,
        "STAY": model_init.STAY,
        "INTERACT": model_init.INTERACT,
    }

    print("Orientations: 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST")
    print(f"noise_level={noise_level}")

    for start_ori in range(model_init.N_DIRECTIONS):
        parents = {"agent_orientation": one_hot(start_ori, model_init.N_DIRECTIONS)}
        print(f"\nFrom start ori={start_ori}:")
        for name, a in actions.items():
            nxt = B_agent_orientation(parents, a, noise_level=noise_level)
            print(f"  action {name:<8} -> dist={np.round(nxt, 6)} (argmax={int(np.argmax(nxt))})")


if __name__ == "__main__":
    main()

