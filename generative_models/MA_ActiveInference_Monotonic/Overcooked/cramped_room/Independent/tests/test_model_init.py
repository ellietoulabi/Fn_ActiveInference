"""
Tests for MA_ActiveInference_Monotonic Overcooked cramped_room Independent model_init.
Validates layout, walkable cells, and conversion functions against the cramped_room layout.
Run from project root: python -m generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.Independent.tests.test_model_init
Or from this directory: python test_model_init.py
"""

import importlib.util
from pathlib import Path

# Load model_init from parent directory (Independent/model_init.py)
_this_dir = Path(__file__).resolve().parent
_model_init_path = _this_dir.parent / "model_init.py"
_spec = importlib.util.spec_from_file_location("model_init", _model_init_path)
model_init = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_init)


# Cramped room layout grid (same as layout.txt). Walkable = ' ', '1', '2'; P = pot; S = serve.
_CRAMPED_ROOM_GRID = [
    "XXPXX",
    "O1  O",
    "X  2X",
    "XDXSX",
]


def _parse_cramped_room_layout():
    """Parse cramped_room layout grid. Returns (width, height, valid_xy, pot_xy, serve_xy)."""
    rows = _CRAMPED_ROOM_GRID
    height = len(rows)
    width = len(rows[0]) if rows else 0
    valid_xy = []
    pot_xy = []
    serve_xy = []
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell in " 12":  # walkable (space or player start)
                valid_xy.append((x, y))
            elif cell == "P":
                pot_xy.append((x, y))
            elif cell == "S":
                serve_xy.append((x, y))
    return width, height, valid_xy, pot_xy, serve_xy


def test_grid_shape():
    """model_init grid dimensions must match layout."""
    print("  test_grid_shape:")
    width, height, _, _, _ = _parse_cramped_room_layout()
    print(f"    step 1: layout parsed -> width={width}, height={height}")
    print(f"    step 2: model_init has GRID_WIDTH={model_init.GRID_WIDTH}, GRID_HEIGHT={model_init.GRID_HEIGHT}")
    assert width == model_init.GRID_WIDTH, f"width: layout={width} model_init={model_init.GRID_WIDTH}"
    print(f"    step 3: width match OK")
    assert height == model_init.GRID_HEIGHT, f"height: layout={height} model_init={model_init.GRID_HEIGHT}"
    print(f"    step 4: height match OK -> test_grid_shape: OK")


def test_walkable_indices_match_env():
    """WALKABLE_INDICES must equal layout walkable cells (as sorted flat indices)."""
    print("  test_walkable_indices_match_env:")
    _, _, valid_xy, _, _ = _parse_cramped_room_layout()
    print(f"    step 1: layout walkable (x,y) = {valid_xy}")
    layout_flat = sorted([model_init.xy_to_index(x, y) for x, y in valid_xy])
    print(f"    step 2: layout flat indices (sorted) = {layout_flat}")
    expected = sorted(model_init.WALKABLE_INDICES)
    print(f"    step 3: model_init.WALKABLE_INDICES (sorted) = {expected}")
    assert layout_flat == expected, (
        f"Walkable mismatch: layout (sorted flat)={layout_flat} model_init.WALKABLE_INDICES (sorted)={expected}"
    )
    print(f"    step 4: flat lists match OK")
    assert len(layout_flat) == model_init.N_WALKABLE, (
        f"N_WALKABLE={model_init.N_WALKABLE} but layout has {len(layout_flat)} walkable cells"
    )
    print(f"    step 5: N_WALKABLE={model_init.N_WALKABLE} -> test_walkable_indices_match_env: OK")


def test_pot_and_serving_match_env():
    """POT_INDICES and SERVING_INDICES must match layout."""
    print("  test_pot_and_serving_match_env:")
    _, _, _, pot_xy, serve_xy = _parse_cramped_room_layout()
    print(f"    step 1: layout pot (x,y) = {pot_xy}, serve (x,y) = {serve_xy}")
    layout_pot_flat = [model_init.xy_to_index(x, y) for x, y in pot_xy]
    layout_serve_flat = [model_init.xy_to_index(x, y) for x, y in serve_xy]
    print(f"    step 2: layout pot flat = {layout_pot_flat}, serve flat = {layout_serve_flat}")
    print(f"    step 3: model_init.POT_INDICES = {model_init.POT_INDICES}, SERVING_INDICES = {model_init.SERVING_INDICES}")
    assert set(layout_pot_flat) == set(model_init.POT_INDICES), (
        f"Pot: layout={layout_pot_flat} model_init={model_init.POT_INDICES}"
    )
    print(f"    step 4: pot match OK")
    assert set(layout_serve_flat) == set(model_init.SERVING_INDICES), (
        f"Serve: layout={layout_serve_flat} model_init={model_init.SERVING_INDICES}"
    )
    print(f"    step 5: serving match OK -> test_pot_and_serving_match_env: OK")


def test_walkable_conversion_roundtrip():
    """walkable_idx_to_grid_idx and grid_idx_to_walkable_idx must be inverses on walkable cells."""
    print("  test_walkable_conversion_roundtrip:")
    for w in range(model_init.N_WALKABLE):
        g = model_init.walkable_idx_to_grid_idx(w)
        assert g is not None
        w2 = model_init.grid_idx_to_walkable_idx(g)
        assert w2 == w, f"walkable {w} -> grid {g} -> walkable {w2}"
        print(f"    step {w + 1}: walkable {w} -> grid {g} -> walkable {w2} OK")
    print(f"    -> test_walkable_conversion_roundtrip: OK")


def test_grid_to_walkable_non_walkable_returns_none():
    """grid_idx_to_walkable_idx must return None for non-walkable grid indices."""
    print("  test_grid_to_walkable_non_walkable_returns_none:")
    valid_flat = set(model_init.WALKABLE_INDICES)  # use model_init as source of truth
    non_walkable = [i for i in range(model_init.GRID_SIZE) if i not in valid_flat]
    print(f"    step 1: non-walkable grid indices = {non_walkable}")
    for grid_idx in non_walkable:
        w = model_init.grid_idx_to_walkable_idx(grid_idx)
        assert w is None, f"grid_idx={grid_idx} should not be walkable, got walkable_idx={w}"
    print(f"    step 2: grid_idx_to_walkable_idx(i) == None for all {len(non_walkable)} non-walkable i OK")
    print(f"    -> test_grid_to_walkable_non_walkable_returns_none: OK")


def test_walkable_to_grid_in_bounds():
    """walkable_idx_to_grid_idx must return indices in [0, GRID_SIZE-1]."""
    print("  test_walkable_to_grid_in_bounds:")
    for w in range(model_init.N_WALKABLE):
        g = model_init.walkable_idx_to_grid_idx(w)
        assert 0 <= g < model_init.GRID_SIZE, f"walkable {w} -> grid {g} out of range"
        print(f"    step {w + 1}: walkable {w} -> grid {g} (in [0, {model_init.GRID_SIZE - 1}]) OK")
    g_neg = model_init.walkable_idx_to_grid_idx(-1)
    assert 0 <= g_neg < model_init.GRID_SIZE
    print(f"    step {model_init.N_WALKABLE + 1}: fallback walkable_idx_to_grid_idx(-1) = {g_neg} OK")
    g_big = model_init.walkable_idx_to_grid_idx(99)
    assert 0 <= g_big < model_init.GRID_SIZE
    print(f"    step {model_init.N_WALKABLE + 2}: fallback walkable_idx_to_grid_idx(99) = {g_big} OK")
    print(f"    -> test_walkable_to_grid_in_bounds: OK")


def test_xy_index_roundtrip():
    """xy_to_index and index_to_xy must be inverses."""
    print("  test_xy_index_roundtrip:")
    step = 0
    for y in range(model_init.GRID_HEIGHT):
        for x in range(model_init.GRID_WIDTH):
            step += 1
            idx = model_init.xy_to_index(x, y)
            x2, y2 = model_init.index_to_xy(idx)
            assert (x2, y2) == (x, y), f"(x,y)=({x},{y}) -> idx={idx} -> (x,y)=({x2},{y2})"
            print(f"    step {step}: (x,y)=({x},{y}) -> idx={idx} -> (x,y)=({x2},{y2}) OK")
    print(f"    -> test_xy_index_roundtrip: OK")


def test_state_and_observation_sizes():
    """State and observation factor sizes must match model_init constants."""
    print("  test_state_and_observation_sizes:")
    print(f"    step 1: len(states['agent_pos']) = {len(model_init.states['agent_pos'])} == N_WALKABLE = {model_init.N_WALKABLE} OK")
    assert len(model_init.states["agent_pos"]) == model_init.N_WALKABLE
    print(f"    step 2: len(states['agent_orientation']) = {len(model_init.states['agent_orientation'])} == N_DIRECTIONS = {model_init.N_DIRECTIONS} OK")
    assert len(model_init.states["agent_orientation"]) == model_init.N_DIRECTIONS
    print(f"    step 3: len(states['agent_held']) = {len(model_init.states['agent_held'])} == N_HELD_TYPES = {model_init.N_HELD_TYPES} OK")
    assert len(model_init.states["agent_held"]) == model_init.N_HELD_TYPES
    print(f"    step 4: len(states['pot_state']) = {len(model_init.states['pot_state'])} == N_POT_STATES = {model_init.N_POT_STATES} OK")
    assert len(model_init.states["pot_state"]) == model_init.N_POT_STATES
    for i, k in enumerate(("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered")):
        assert len(model_init.states[k]) == 2
        print(f"    step {5 + i}: len(states['{k}']) = 2 OK")
    print(f"    step 10: len(observations['agent_pos_obs']) = {len(model_init.observations['agent_pos_obs'])} == N_WALKABLE OK")
    assert len(model_init.observations["agent_pos_obs"]) == model_init.N_WALKABLE
    print(f"    step 11: len(observations['soup_delivered_obs']) = {len(model_init.observations['soup_delivered_obs'])} == 2 OK")
    assert len(model_init.observations["soup_delivered_obs"]) == 2
    print(f"    -> test_state_and_observation_sizes: OK")


def run_all():
    print("Testing model_init (Independent Monotonic cramped_room) against layout\n")
    test_grid_shape()
    test_walkable_indices_match_env()
    test_pot_and_serving_match_env()
    test_walkable_conversion_roundtrip()
    test_grid_to_walkable_non_walkable_returns_none()
    test_walkable_to_grid_in_bounds()
    test_xy_index_roundtrip()
    test_state_and_observation_sizes()
    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all()
