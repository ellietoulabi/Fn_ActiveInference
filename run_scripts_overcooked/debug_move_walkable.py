"""
Quick debug runner for `_move_walkable` in the Independent Cramped Room B model.

Run from repo root:
    python3 run_scripts_overcooked/debug_move_walkable.py
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
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent.B import (  # noqa: E402
    _move_walkable,
)


def main() -> None:
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

    print("\nTransitions: next_w = _move_walkable(w, action)")
    for w in range(model_init.N_WALKABLE):
        print(f"\nFrom w={w} (grid={model_init.WALKABLE_INDICES[w]}):")
        for name, a in actions.items():
            nxt = _move_walkable(w, a)
            note = ""
            if nxt == w and name in ("NORTH", "SOUTH", "EAST", "WEST"):
                note = " (blocked -> stay)"
            print(f"  {name:<8} -> {nxt}{note}")

    # Small matrix view (rows=start w, cols=actions)
    print("\nMatrix view (rows=start w, cols=[N,S,E,W,STAY,INT])")
    col_order = ["NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"]
    mat = np.zeros((model_init.N_WALKABLE, len(col_order)), dtype=int)
    for i, w in enumerate(range(model_init.N_WALKABLE)):
        for j, name in enumerate(col_order):
            mat[i, j] = _move_walkable(w, actions[name])
    print(mat)


if __name__ == "__main__":
    main()

