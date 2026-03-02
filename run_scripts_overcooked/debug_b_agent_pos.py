"""
Quick debug runner for the Independent Cramped Room transition `B_agent_pos`.

Run from repo root:
    python3 run_scripts_overcooked/debug_b_agent_pos.py
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

from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent import (
    model_init,
)
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent.B import (
    B_agent_pos,
)


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=float)
    v[i] = 1.0
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug B_agent_pos transitions.")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise level mixed into next-state distribution (default: 0.0).",
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

    print("Walkable indices (walkable_idx -> grid_idx -> (x,y))")
    for w, grid in enumerate(model_init.WALKABLE_INDICES):
        x, y = model_init.index_to_xy(grid)
        print(f"  w={w} -> grid={grid} -> ({x},{y})")

    for start in range(model_init.N_WALKABLE):
        parents = {"agent_pos": one_hot(start, model_init.N_WALKABLE)}
        print(f"\nFrom start walkable w={start} (grid={model_init.WALKABLE_INDICES[start]}):")
        for name, a in actions.items():
            nxt = B_agent_pos(parents, a, noise_level=noise_level)
            arg = int(np.argmax(nxt))
            print(
                f"  action {name:<8} -> next w={arg} (grid={model_init.WALKABLE_INDICES[arg]})"
                f"  dist={np.round(nxt, 6)}"
            )


if __name__ == "__main__":
    main()

