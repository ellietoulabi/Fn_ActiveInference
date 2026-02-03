"""
Test suite for model_init - Independent paradigm, Cramped Room.

Tests constants, location indices, utility functions, and data structures.
"""

import numpy as np
import sys
import importlib.util
from pathlib import Path

# Load model_init directly
independent_dir = Path(__file__).parent
model_init_path = independent_dir / "model_init.py"
spec = importlib.util.spec_from_file_location("model_init", model_init_path)
model_init = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_init)


def test_constants():
    """Test that constants are correctly defined."""
    print("\n" + "="*70)
    print("TEST: Constants")
    print("="*70)
    
    # Grid size
    assert model_init.GRID_WIDTH == 5, f"GRID_WIDTH should be 5, got {model_init.GRID_WIDTH}"
    assert model_init.GRID_HEIGHT == 4, f"GRID_HEIGHT should be 4, got {model_init.GRID_HEIGHT}"
    assert model_init.GRID_SIZE == 20, f"GRID_SIZE should be 20, got {model_init.GRID_SIZE}"
    print(f"\nTest: Grid size constants")
    print(f"  GRID_WIDTH: {model_init.GRID_WIDTH}")
    print(f"  GRID_HEIGHT: {model_init.GRID_HEIGHT}")
    print(f"  GRID_SIZE: {model_init.GRID_SIZE}")
    print("  ✓ PASS")
    
    # Actions
    assert model_init.NORTH == 0, "NORTH should be 0"
    assert model_init.SOUTH == 1, "SOUTH should be 1"
    assert model_init.EAST == 2, "EAST should be 2"
    assert model_init.WEST == 3, "WEST should be 3"
    assert model_init.STAY == 4, "STAY should be 4"
    assert model_init.INTERACT == 5, "INTERACT should be 5"
    assert model_init.N_ACTIONS == 6, "N_ACTIONS should be 6"
    print(f"\nTest: Action constants")
    print(f"  Actions: NORTH={model_init.NORTH}, SOUTH={model_init.SOUTH}, EAST={model_init.EAST}, WEST={model_init.WEST}, STAY={model_init.STAY}, INTERACT={model_init.INTERACT}")
    print("  ✓ PASS")
    
    # Directions
    assert model_init.DIR_NORTH == (0, -1), "DIR_NORTH should be (0, -1)"
    assert model_init.DIR_SOUTH == (0, 1), "DIR_SOUTH should be (0, 1)"
    assert model_init.DIR_EAST == (1, 0), "DIR_EAST should be (1, 0)"
    assert model_init.DIR_WEST == (-1, 0), "DIR_WEST should be (-1, 0)"
    assert model_init.N_DIRECTIONS == 4, "N_DIRECTIONS should be 4"
    print(f"\nTest: Direction constants")
    print(f"  Directions: {model_init.DIRECTIONS}")
    print("  ✓ PASS")
    
    # Held object types
    assert model_init.HELD_NONE == 0, "HELD_NONE should be 0"
    assert model_init.HELD_ONION == 1, "HELD_ONION should be 1"
    assert model_init.HELD_DISH == 2, "HELD_DISH should be 2"
    assert model_init.HELD_SOUP == 3, "HELD_SOUP should be 3"
    assert model_init.N_HELD_TYPES == 4, "N_HELD_TYPES should be 4"
    print(f"\nTest: Held object type constants")
    print(f"  HELD_NONE={model_init.HELD_NONE}, HELD_ONION={model_init.HELD_ONION}, HELD_DISH={model_init.HELD_DISH}, HELD_SOUP={model_init.HELD_SOUP}")
    print("  ✓ PASS")
    
    # Pot states
    assert model_init.POT_IDLE == 0, "POT_IDLE should be 0"
    assert model_init.POT_COOKING == 1, "POT_COOKING should be 1"
    assert model_init.POT_READY == 2, "POT_READY should be 2"
    assert model_init.N_POT_STATES == 3, "N_POT_STATES should be 3"
    print(f"\nTest: Pot state constants")
    print(f"  POT_IDLE={model_init.POT_IDLE}, POT_COOKING={model_init.POT_COOKING}, POT_READY={model_init.POT_READY}")
    print("  ✓ PASS")
    
    return True


def test_location_indices():
    """Test that location indices are correctly computed."""
    print("\n" + "="*70)
    print("TEST: Location Indices")
    print("="*70)
    
    # Pot location: (2, 0) should be index 2
    pot_pos = model_init.POT_LOCATIONS[0]
    pot_idx = model_init.POT_INDICES[0]
    expected_pot_idx = model_init.xy_to_index(pot_pos[0], pot_pos[1])
    
    print(f"\nTest: Pot location")
    print(f"  Position: {pot_pos} → index {pot_idx}")
    assert pot_idx == expected_pot_idx, f"Pot index should be {expected_pot_idx}, got {pot_idx}"
    assert pot_idx == 2, f"Pot at (2, 0) should be index 2, got {pot_idx}"
    print("  ✓ PASS")
    
    # Serving location: (3, 3) should be index 18
    serving_pos = model_init.SERVING_LOCATIONS[0]
    serving_idx = model_init.SERVING_INDICES[0]
    expected_serving_idx = model_init.xy_to_index(serving_pos[0], serving_pos[1])
    
    print(f"\nTest: Serving location")
    print(f"  Position: {serving_pos} → index {serving_idx}")
    assert serving_idx == expected_serving_idx, f"Serving index should be {expected_serving_idx}, got {serving_idx}"
    assert serving_idx == 18, f"Serving at (3, 3) should be index 18, got {serving_idx}"
    print("  ✓ PASS")
    
    # Onion dispensers: (0, 1) = index 5, (4, 1) = index 9
    onion_dispensers = model_init.ONION_DISPENSERS
    onion_indices = model_init.ONION_DISPENSER_INDICES
    
    print(f"\nTest: Onion dispensers")
    for i, (pos, idx) in enumerate(zip(onion_dispensers, onion_indices)):
        expected_idx = model_init.xy_to_index(pos[0], pos[1])
        print(f"  Dispenser {i}: {pos} → index {idx}")
        assert idx == expected_idx, f"Dispenser {i} index should be {expected_idx}, got {idx}"
    assert onion_indices[0] == 5, f"First dispenser should be index 5, got {onion_indices[0]}"
    assert onion_indices[1] == 9, f"Second dispenser should be index 9, got {onion_indices[1]}"
    print("  ✓ PASS")
    
    return True


def test_xy_conversion():
    """Test xy_to_index and index_to_xy functions."""
    print("\n" + "="*70)
    print("TEST: XY Coordinate Conversion")
    print("="*70)
    
    # Test xy_to_index
    test_cases = [
        ((0, 0), 0),
        ((1, 1), 6),
        ((2, 0), 2),
        ((3, 3), 18),
        ((4, 1), 9),
        ((4, 3), 19),  # Last position
    ]
    
    print(f"\nTest: xy_to_index")
    for (x, y), expected_idx in test_cases:
        idx = model_init.xy_to_index(x, y)
        print(f"  ({x}, {y}) → {idx} (expected {expected_idx})")
        assert idx == expected_idx, f"({x}, {y}) should map to {expected_idx}, got {idx}"
    print("  ✓ PASS")
    
    # Test index_to_xy
    print(f"\nTest: index_to_xy")
    for (x, y), idx in test_cases:
        result_x, result_y = model_init.index_to_xy(idx)
        print(f"  {idx} → ({result_x}, {result_y}) (expected ({x}, {y}))")
        assert (result_x, result_y) == (x, y), f"{idx} should map to ({x}, {y}), got ({result_x}, {result_y})"
    print("  ✓ PASS")
    
    # Test round-trip conversion
    print(f"\nTest: Round-trip conversion")
    for idx in range(model_init.GRID_SIZE):
        x, y = model_init.index_to_xy(idx)
        idx_roundtrip = model_init.xy_to_index(x, y)
        assert idx == idx_roundtrip, f"Round-trip failed: {idx} → ({x}, {y}) → {idx_roundtrip}"
    print(f"  All {model_init.GRID_SIZE} positions round-trip correctly")
    print("  ✓ PASS")
    
    return True


def test_direction_conversion():
    """Test direction_to_index and index_to_direction functions."""
    print("\n" + "="*70)
    print("TEST: Direction Conversion")
    print("="*70)
    
    # Test direction_to_index
    direction_tests = [
        (model_init.DIR_NORTH, 0),
        (model_init.DIR_SOUTH, 1),
        (model_init.DIR_EAST, 2),
        (model_init.DIR_WEST, 3),
    ]
    
    print(f"\nTest: direction_to_index")
    for direction, expected_idx in direction_tests:
        idx = model_init.direction_to_index(direction)
        print(f"  {direction} → {idx} (expected {expected_idx})")
        assert idx == expected_idx, f"{direction} should map to {expected_idx}, got {idx}"
    print("  ✓ PASS")
    
    # Test index_to_direction
    print(f"\nTest: index_to_direction")
    for direction, idx in direction_tests:
        result_dir = model_init.index_to_direction(idx)
        print(f"  {idx} → {result_dir} (expected {direction})")
        assert result_dir == direction, f"{idx} should map to {direction}, got {result_dir}"
    print("  ✓ PASS")
    
    # Test round-trip conversion
    print(f"\nTest: Round-trip conversion")
    for idx in range(model_init.N_DIRECTIONS):
        direction = model_init.index_to_direction(idx)
        idx_roundtrip = model_init.direction_to_index(direction)
        assert idx == idx_roundtrip, f"Round-trip failed: {idx} → {direction} → {idx_roundtrip}"
    print("  All directions round-trip correctly")
    print("  ✓ PASS")
    
    # Test invalid direction (should default to 0)
    invalid_dir = (99, 99)
    idx = model_init.direction_to_index(invalid_dir)
    assert idx == 0, f"Invalid direction should default to 0, got {idx}"
    print(f"\nTest: Invalid direction handling")
    print("  ✓ PASS")
    
    return True


def test_object_name_conversion():
    """Test object_name_to_held_type function."""
    print("\n" + "="*70)
    print("TEST: Object Name Conversion")
    print("="*70)
    
    test_cases = [
        (None, model_init.HELD_NONE),
        ("onion", model_init.HELD_ONION),
        ("dish", model_init.HELD_DISH),
        ("soup", model_init.HELD_SOUP),
        ("unknown", model_init.HELD_NONE),  # Unknown should default to HELD_NONE
        ("tomato", model_init.HELD_NONE),  # Not in cramped_room
    ]
    
    print(f"\nTest: object_name_to_held_type")
    for obj_name, expected_type in test_cases:
        held_type = model_init.object_name_to_held_type(obj_name)
        print(f"  '{obj_name}' → {held_type} (expected {expected_type})")
        assert held_type == expected_type, f"'{obj_name}' should map to {expected_type}, got {held_type}"
    print("  ✓ PASS")
    
    return True


def test_location_checking():
    """Test location checking functions."""
    print("\n" + "="*70)
    print("TEST: Location Checking")
    print("="*70)
    
    # Test is_at_location
    print(f"\nTest: is_at_location")
    pot_idx = model_init.POT_INDICES[0]
    assert model_init.is_at_location(pot_idx, model_init.POT_INDICES), "Should detect pot location"
    assert not model_init.is_at_location(6, model_init.POT_INDICES), "Should not detect non-pot location"
    print("  ✓ PASS")
    
    # Test is_at_pot
    print(f"\nTest: is_at_pot")
    pot_idx = model_init.POT_INDICES[0]  # Index 2
    assert model_init.is_at_pot(pot_idx), f"Position {pot_idx} should be at pot"
    assert not model_init.is_at_pot(6), "Position 6 should not be at pot"
    assert not model_init.is_at_pot(18), "Position 18 should not be at pot"
    print(f"  Pot at index {pot_idx}: {model_init.is_at_pot(pot_idx)}")
    print("  ✓ PASS")
    
    # Test is_at_serving
    print(f"\nTest: is_at_serving")
    serving_idx = model_init.SERVING_INDICES[0]  # Index 18
    assert model_init.is_at_serving(serving_idx), f"Position {serving_idx} should be at serving"
    assert not model_init.is_at_serving(6), "Position 6 should not be at serving"
    assert not model_init.is_at_serving(2), "Position 2 should not be at serving"
    print(f"  Serving at index {serving_idx}: {model_init.is_at_serving(serving_idx)}")
    print("  ✓ PASS")
    
    # Test is_at_onion_dispenser
    print(f"\nTest: is_at_onion_dispenser")
    dispenser_indices = model_init.ONION_DISPENSER_INDICES  # [5, 9]
    for dispenser_idx in dispenser_indices:
        assert model_init.is_at_onion_dispenser(dispenser_idx), f"Position {dispenser_idx} should be at dispenser"
    assert not model_init.is_at_onion_dispenser(6), "Position 6 should not be at dispenser"
    assert not model_init.is_at_onion_dispenser(2), "Position 2 should not be at dispenser"
    print(f"  Dispensers at indices {dispenser_indices}")
    print("  ✓ PASS")
    
    return True


def test_state_observation_definitions():
    """Test state and observation definitions."""
    print("\n" + "="*70)
    print("TEST: State and Observation Definitions")
    print("="*70)
    
    # Test states dictionary
    print(f"\nTest: states dictionary")
    assert "agent_pos" in model_init.states, "Should have agent_pos"
    assert "agent_orientation" in model_init.states, "Should have agent_orientation"
    assert "agent_held" in model_init.states, "Should have agent_held"
    assert "other_agent_pos" in model_init.states, "Should have other_agent_pos"
    assert "pot_state" in model_init.states, "Should have pot_state"
    assert "soup_delivered" not in model_init.states, "soup_delivered is observation-only, not state"

    assert len(model_init.states["agent_pos"]) == model_init.GRID_SIZE, "agent_pos should have GRID_SIZE states"
    assert len(model_init.states["agent_orientation"]) == model_init.N_DIRECTIONS, "agent_orientation should have N_DIRECTIONS states"
    assert len(model_init.states["agent_held"]) == model_init.N_HELD_TYPES, "agent_held should have N_HELD_TYPES states"
    assert len(model_init.states["other_agent_pos"]) == model_init.GRID_SIZE, "other_agent_pos should have GRID_SIZE states"
    assert len(model_init.states["pot_state"]) == model_init.N_POT_STATES, "pot_state should have N_POT_STATES states"
    
    print(f"  States: {list(model_init.states.keys())}")
    print("  ✓ PASS")
    
    # Test observations dictionary
    print(f"\nTest: observations dictionary")
    assert "agent_pos" in model_init.observations, "Should have agent_pos observation"
    assert "agent_orientation" in model_init.observations, "Should have agent_orientation observation"
    assert "agent_held" in model_init.observations, "Should have agent_held observation"
    assert "other_agent_pos" in model_init.observations, "Should have other_agent_pos observation"
    assert "pot_state" in model_init.observations, "Should have pot_state observation"
    assert "soup_delivered" in model_init.observations, "Should have soup_delivered observation"
    
    # Every state factor has a corresponding observation; observations can have extras (e.g. front_tile_type, soup_delivered)
    for key in model_init.states:
        assert key in model_init.observations, f"Observation {key} should exist"
        assert len(model_init.observations[key]) == len(model_init.states[key]), \
            f"Observation {key} should have same size as state {key}"
    
    print(f"  Observations: {list(model_init.observations.keys())}")
    print("  ✓ PASS")
    
    return True


def test_dependencies():
    """Test dependency mappings."""
    print("\n" + "="*70)
    print("TEST: Dependency Mappings")
    print("="*70)
    
    # Test observation_state_dependencies
    print(f"\nTest: observation_state_dependencies")
    for obs_key in model_init.observations:
        assert obs_key in model_init.observation_state_dependencies, \
            f"Observation {obs_key} should have state dependencies"
        deps = model_init.observation_state_dependencies[obs_key]
        assert isinstance(deps, list), "Dependencies should be a list"
        for dep in deps:
            assert dep in model_init.states, f"Dependency {dep} should be a valid state"
    
    # Verify specific dependencies
    assert model_init.observation_state_dependencies["agent_pos"] == ["agent_pos"], \
        "agent_pos observation should depend only on agent_pos state"
    assert model_init.observation_state_dependencies["soup_delivered"] == ["agent_pos", "agent_held", "other_agent_pos"], \
        "soup_delivered is event obs only; should depend on pos/held/other"
    
    print(f"  Observation dependencies: {model_init.observation_state_dependencies}")
    print("  ✓ PASS")
    
    # Test state_state_dependencies
    print(f"\nTest: state_state_dependencies")
    for state_key in model_init.states:
        assert state_key in model_init.state_state_dependencies, \
            f"State {state_key} should have state dependencies"
        deps = model_init.state_state_dependencies[state_key]
        assert isinstance(deps, list), "Dependencies should be a list"
        for dep in deps:
            assert dep in model_init.states, f"Dependency {dep} should be a valid state"
    
    # Verify specific dependencies
    assert "agent_pos" in model_init.state_state_dependencies["agent_pos"], \
        "agent_pos should depend on itself"
    assert "agent_orientation" in model_init.state_state_dependencies["agent_pos"], \
        "agent_pos should depend on agent_orientation"
    assert "other_agent_pos" in model_init.state_state_dependencies["agent_pos"], \
        "agent_pos should depend on other_agent_pos"
    
    print(f"  State dependencies: {model_init.state_state_dependencies}")
    print("  ✓ PASS")
    
    return True


def test_layout_consistency():
    """Test that layout matches the cramped_room layout."""
    print("\n" + "="*70)
    print("TEST: Layout Consistency")
    print("="*70)
    
    # Layout from layout.txt:
    # XXPXX
    # O1  O
    # X  2X
    # XDXSX
    # 
    # Where:
    # X = wall
    # P = pot (2, 0)
    # O = onion dispenser (0, 1) and (4, 1)
    # 1 = agent 1 start (1, 1)
    # 2 = agent 2 start (3, 2)
    # D = dish dispenser (1, 3)
    # S = serving (3, 3)
    
    print(f"\nTest: Layout consistency")
    print(f"  Pot location: {model_init.POT_LOCATIONS[0]} (should be (2, 0))")
    assert model_init.POT_LOCATIONS[0] == (2, 0), "Pot should be at (2, 0)"
    
    print(f"  Serving location: {model_init.SERVING_LOCATIONS[0]} (should be (3, 3))")
    assert model_init.SERVING_LOCATIONS[0] == (3, 3), "Serving should be at (3, 3)"
    
    print(f"  Onion dispensers: {model_init.ONION_DISPENSERS} (should be [(0, 1), (4, 1)])")
    assert model_init.ONION_DISPENSERS == [(0, 1), (4, 1)], "Onion dispensers should be at [(0, 1), (4, 1)]"
    
    # Verify indices match
    assert model_init.POT_INDICES[0] == 2, "Pot index should be 2"
    assert model_init.SERVING_INDICES[0] == 18, "Serving index should be 18"
    assert model_init.ONION_DISPENSER_INDICES == [5, 9], "Onion dispenser indices should be [5, 9]"
    
    print("  ✓ PASS")
    
    return True


def test_independent_paradigm_consistency():
    """Test that model_init is consistent with Independent paradigm."""
    print("\n" + "="*70)
    print("TEST: Independent Paradigm Consistency")
    print("="*70)
    
    print(f"\nTest: Single-agent perspective")
    # In Independent paradigm, each agent has:
    # - agent_pos: their own position
    # - other_agent_pos: other agent's position (treated as part of environment)
    # - agent_orientation: their own orientation
    # - agent_held: their own held object
    # - pot_state: shared pot state
    # - soup_delivered: observation-only (event), not a state factor
    
    assert "agent_pos" in model_init.states, "Should have agent_pos"
    assert "other_agent_pos" in model_init.states, "Should have other_agent_pos (key for Independent paradigm)"
    assert "agent_orientation" in model_init.states, "Should have agent_orientation"
    assert "agent_held" in model_init.states, "Should have agent_held"
    assert "pot_state" in model_init.states, "Should have pot_state"
    assert "soup_delivered" not in model_init.states, "soup_delivered is observation-only"
    assert "soup_delivered" in model_init.observations, "soup_delivered should be an observation"
    
    # Verify other_agent_pos is treated as part of environment (in dependencies)
    assert "other_agent_pos" in model_init.state_state_dependencies["agent_pos"], \
        "agent_pos should depend on other_agent_pos (collision avoidance)"
    
    print("  ✓ PASS - Model structure is consistent with Independent paradigm")
    
    return True


def test_A_model_init_compatibility():
    """Check model_init satisfies A.py contract: init has required attrs and observation spec matches A output."""
    print("\n" + "="*70)
    print("TEST: A.py and model_init compatibility")
    print("="*70)

    # 1) model_init must provide everything A expects (see A.py docstring)
    assert hasattr(model_init, "GRID_SIZE"), "model_init must define GRID_SIZE"
    assert hasattr(model_init, "N_DIRECTIONS"), "model_init must define N_DIRECTIONS"
    assert hasattr(model_init, "N_HELD_TYPES"), "model_init must define N_HELD_TYPES"
    assert hasattr(model_init, "N_POT_STATES"), "model_init must define N_POT_STATES"
    assert hasattr(model_init, "N_FRONT_TYPES"), "model_init must define N_FRONT_TYPES"
    assert callable(getattr(model_init, "compute_front_tile_type", None)), "model_init must define compute_front_tile_type"
    print("  model_init provides GRID_SIZE, N_DIRECTIONS, N_HELD_TYPES, N_POT_STATES, N_FRONT_TYPES, compute_front_tile_type")

    # 2) Observation keys and shapes must match what A_fn returns (A outputs one array per modality)
    expected_obs_spec = {
        "agent_pos": model_init.GRID_SIZE,
        "other_agent_pos": model_init.GRID_SIZE,
        "agent_orientation": model_init.N_DIRECTIONS,
        "agent_held": model_init.N_HELD_TYPES,
        "pot_state": model_init.N_POT_STATES,
        "front_tile_type": model_init.N_FRONT_TYPES,
        "soup_delivered": 2,
    }
    for key, expected_len in expected_obs_spec.items():
        assert key in model_init.observations, f"model_init.observations must have '{key}' for A_fn output"
        assert len(model_init.observations[key]) == expected_len, (
            f"model_init.observations['{key}'] length {len(model_init.observations[key])} != {expected_len}"
        )
    assert set(model_init.observations.keys()) == set(expected_obs_spec.keys()), (
        "model_init.observations keys must match A_fn output keys"
    )
    print(f"  model_init.observations keys and lengths match A_fn output spec")

    # 3) observation_state_dependencies must include every observation and only reference state factors
    for obs_key in model_init.observations:
        assert obs_key in model_init.observation_state_dependencies, (
            f"observation_state_dependencies must have '{obs_key}'"
        )
        for dep in model_init.observation_state_dependencies[obs_key]:
            assert dep in model_init.states, (
                f"observation_state_dependencies['{obs_key}'] references non-state '{dep}'"
            )
    print("  observation_state_dependencies consistent with states")

    # 4) compute_front_tile_type(agent_pos, agent_ori, other_pos) returns int in [0, N_FRONT_TYPES)
    ft = model_init.compute_front_tile_type(0, 0, 1)
    assert isinstance(ft, int), "compute_front_tile_type must return int"
    assert 0 <= ft < model_init.N_FRONT_TYPES, f"compute_front_tile_type returned {ft}, must be in [0, N_FRONT_TYPES)"
    print("  compute_front_tile_type signature and return value OK")

    print("  ✓ PASS - A and model_init are compatible")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL_INIT TEST SUITE - Overcooked Independent Cramped Room")
    print("="*70)
    
    results = {}
    
    try:
        results["constants"] = test_constants()
        results["location_indices"] = test_location_indices()
        results["xy_conversion"] = test_xy_conversion()
        results["direction_conversion"] = test_direction_conversion()
        results["object_name_conversion"] = test_object_name_conversion()
        results["location_checking"] = test_location_checking()
        results["state_observation_definitions"] = test_state_observation_definitions()
        results["dependencies"] = test_dependencies()
        results["layout_consistency"] = test_layout_consistency()
        results["independent_paradigm"] = test_independent_paradigm_consistency()
        results["A_model_init_compatibility"] = test_A_model_init_compatibility()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = False
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:25s}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*70}")
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("✓ model_init.py is correctly configured for Independent paradigm")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("="*70 + "\n")
