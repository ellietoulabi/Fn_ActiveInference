"""
Quick debug runner for `_get_front` in the Independent Cramped Room B model.

Run from repo root:
    python3 run_scripts_overcooked/debug_get_front.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent import (  # noqa: E402
    model_init,
)
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent.B import (  # noqa: E402
    _get_front,
)


def main() -> None:
    ori_names = {
        model_init.NORTH: "NORTH",
        model_init.SOUTH: "SOUTH",
        model_init.EAST: "EAST",
        model_init.WEST: "WEST",
    }
    front_names = {
        model_init.FRONT_WALL: "WALL",
        model_init.FRONT_EMPTY: "EMPTY",
        model_init.FRONT_ONION: "ONION",
        model_init.FRONT_DISH: "DISH",
        model_init.FRONT_POT: "POT",
        model_init.FRONT_SERVE: "SERVE",
        model_init.FRONT_COUNTER: "COUNTER",
    }

    print("Walkable indices (w -> grid -> (x,y))")
    for w, grid in enumerate(model_init.WALKABLE_INDICES):
        x, y = model_init.index_to_xy(grid)
        print(f"  w={w} -> grid={grid} -> ({x},{y})")

    print("\nFront tile types: _get_front(w, ori)")
    for w, grid in enumerate(model_init.WALKABLE_INDICES):
        x, y = model_init.index_to_xy(grid)
        print(f"\nFrom w={w} at ({x},{y}):")
        for ori in (model_init.NORTH, model_init.SOUTH, model_init.EAST, model_init.WEST):
            f = int(_get_front(w, ori))
            print(f"  ori {ori_names[ori]:<5} -> {front_names.get(f, str(f))} ({f})")


if __name__ == "__main__":
    main()

