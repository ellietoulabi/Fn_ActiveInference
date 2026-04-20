"""
Comprehensive test file for the dynamic Overcooked semantic policy generator
implemented in utils.py.

This file is intentionally verbose. It prints:
- the exact state at the beginning of each case
- semantic compilation results for every (destination, mode)
- valid primitive policies
- padded policies
- action sampling results

It assumes:
- policy generation happens inside utils.py
- bfs_shortest_action_path only targets WALKABLE tiles
- semantic destinations are NON-WALKABLE object/counter tiles
- compilation goes through adjacent walkable support tiles
- compiled policies must preserve full semantic cardinality:
    len(DESTINATIONS) * len(MODES)

Run:
    python test_policy.py

Adjust the import below to match your project layout.
"""

import numpy as np

# Adjust this import to your project structure
import utils


# =============================================================================
# Pretty printing helpers
# =============================================================================

def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_subheader(title):
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def print_layout():
    print("Layout:")
    for row in utils.LAYOUT_ROWS:
        print("  " + row)


def print_state(state):
    print("State:")
    print(f"  name            : {state['name']}")
    print(f"  self_pos        : {state['self_pos']}")
    print(f"  self_orient     : {state['self_orient']}")
    print(f"  self_held       : {state['self_held']}")
    print(f"  other_pos       : {state['other_pos']}")
    print(f"  other_orient    : {state['other_orient']}")
    print(f"  other_held      : {state['other_held']}")
    print(f"  pot_state       : {state['pot_state']}")
    print(f"  pot_onions      : {state['pot_onions']}")
    print(f"  soup_ready      : {state['soup_ready']}")
    print("  counter_contents:")
    for k in sorted(state["counter_contents"].keys()):
        print(f"    {k}: {state['counter_contents'][k]}")


def print_policies(label, policies):
    print(f"{label} ({len(policies)} total):")
    for i, pol in enumerate(policies):
        print(f"  {i:2d}: {utils.action_seq_to_str(pol)}")


def print_metadata(entries):
    print(f"Metadata entries ({len(entries)} total):")
    for i, m in enumerate(entries):
        req_face = None if m["required_facing"] is None else utils.ACTION_NAMES[m["required_facing"]]
        final_face = None if m["final_facing"] is None else utils.ACTION_NAMES[m["final_facing"]]
        print(
            f"  {i:2d}: "
            f"({m['destination']}, {m['mode']}) "
            f"target={m['target_tile']} "
            f"approach={m['approach_tile']} "
            f"req_face={req_face} "
            f"final_face={final_face} "
            f"path={utils.action_seq_to_str(m['path'])} "
            f"actions={utils.action_seq_to_str(m['actions'])} "
            f"valid={m['valid']} "
            f"reason={m['reason']}"
        )


# =============================================================================
# Test states
# =============================================================================

TEST_STATES = [
    {
        "name": "Case 1: empty-handed at left side, facing WEST, pot empty, all counters empty",
        "self_pos": (1, 1),
        "self_orient": "WEST",
        "self_held": "nothing",
        "other_pos": (2, 3),
        "other_orient": "NORTH",
        "other_held": "nothing",
        "pot_state": "empty",
        "pot_onions": 0,
        "soup_ready": False,
        "counter_contents": {
            "cntr1": "empty",
            "cntr2": "empty",
            "cntr3": "empty",
            "cntr4": "empty",
            "cntr5": "empty",
        },
    },
    {
        "name": "Case 2: holding onion, facing EAST, pot has 2 onions, one dish on counter",
        "self_pos": (1, 1),
        "self_orient": "EAST",
        "self_held": "onion",
        "other_pos": (2, 3),
        "other_orient": "NORTH",
        "other_held": "nothing",
        "pot_state": "two_onions",
        "pot_onions": 2,
        "soup_ready": False,
        "counter_contents": {
            "cntr1": "empty",
            "cntr2": "empty",
            "cntr3": "dish",
            "cntr4": "empty",
            "cntr5": "empty",
        },
    },
    {
        "name": "Case 3: soup ready, holding dish, facing NORTH, onion on counter",
        "self_pos": (2, 1),
        "self_orient": "NORTH",
        "self_held": "dish",
        "other_pos": (2, 3),
        "other_orient": "WEST",
        "other_held": "nothing",
        "pot_state": "ready",
        "pot_onions": 3,
        "soup_ready": True,
        "counter_contents": {
            "cntr1": "onion",
            "cntr2": "empty",
            "cntr3": "empty",
            "cntr4": "empty",
            "cntr5": "empty",
        },
    },
    {
        "name": "Case 4: holding soup, facing SOUTH, should enable serve interact",
        "self_pos": (2, 3),
        "self_orient": "SOUTH",
        "self_held": "soup",
        "other_pos": (1, 1),
        "other_orient": "EAST",
        "other_held": "nothing",
        "pot_state": "empty",
        "pot_onions": 0,
        "soup_ready": False,
        "counter_contents": {
            "cntr1": "empty",
            "cntr2": "empty",
            "cntr3": "empty",
            "cntr4": "empty",
            "cntr5": "dish",
        },
    },
    {
        "name": "Case 5: empty-handed, facing EAST, soup on counter and onion on another counter",
        "self_pos": (2, 3),
        "self_orient": "EAST",
        "self_held": "nothing",
        "other_pos": (1, 1),
        "other_orient": "WEST",
        "other_held": "nothing",
        "pot_state": "empty",
        "pot_onions": 0,
        "soup_ready": False,
        "counter_contents": {
            "cntr1": "empty",
            "cntr2": "soup",
            "cntr3": "empty",
            "cntr4": "empty",
            "cntr5": "onion",
        },
    },
    {
        "name": "Case 6: already adjacent to cntr5 but facing NORTH",
        "self_pos": (2, 2),
        "self_orient": "NORTH",
        "self_held": "nothing",
        "other_pos": (1, 1),
        "other_orient": "WEST",
        "other_held": "nothing",
        "pot_state": "empty",
        "pot_onions": 0,
        "soup_ready": False,
        "counter_contents": {
            "cntr1": "empty",
            "cntr2": "empty",
            "cntr3": "empty",
            "cntr4": "empty",
            "cntr5": "onion",
        },
    },
]


# =============================================================================
# Layout / geometry tests
# =============================================================================

def test_layout_sanity():
    print_header("TEST 1: layout sanity")

    print_layout()

    assert utils.GRID_H == 4
    assert utils.GRID_W == 5

    expected_walkables = {
        (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3),
    }
    assert utils.WALKABLE_TILES == expected_walkables

    assert utils.is_walkable((1, 1))
    assert utils.is_walkable((2, 2))
    assert not utils.is_walkable((0, 2))  # pot
    assert not utils.is_walkable((3, 2))  # cntr5

    print("Walkable tiles:", sorted(utils.WALKABLE_TILES))
    print("Destination tiles:", utils.DESTINATION_TO_TILE)

    print("\nPASS: layout sanity checks passed.")


def test_adjacent_walkable_tiles():
    print_header("TEST 2: adjacent walkable tiles")

    expected = {
        "onion1": [(1, 1)],
        "onion2": [(1, 3)],
        "pot": [(1, 2)],
        "dish": [(2, 1)],
        "serve": [(2, 3)],
        "cntr1": [(1, 1)],
        "cntr2": [(1, 3)],
        "cntr3": [(2, 1)],
        "cntr4": [(2, 3)],
        "cntr5": [(2, 2)],
    }

    for dest, tile in utils.DESTINATION_TO_TILE.items():
        adj = utils.adjacent_walkable_tiles(tile)
        print(f"{dest:>6} at {tile} -> adjacent walkable: {adj}")
        assert adj == expected[dest], f"Unexpected support tiles for {dest}"

    print("\nPASS: adjacent walkable tile checks passed.")


def test_required_facing():
    print_header("TEST 3: required facing from approach tile")

    checks = [
        ((1, 1), (1, 0), utils.LEFT),    # onion1
        ((1, 3), (1, 4), utils.RIGHT),   # onion2
        ((1, 2), (0, 2), utils.UP),      # pot
        ((2, 1), (3, 1), utils.DOWN),    # dish
        ((2, 3), (3, 3), utils.DOWN),    # serve
        ((2, 2), (3, 2), utils.DOWN),    # cntr5
    ]

    for approach, target, expected in checks:
        got = utils.required_facing_from_approach(approach, target)
        print(
            f"approach={approach}, target={target}, "
            f"required={utils.ACTION_NAMES[got]}"
        )
        assert got == expected

    print("\nPASS: required facing checks passed.")


def test_bfs_paths():
    print_header("TEST 4: BFS shortest path checks")

    start = (1, 1)

    goals = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for goal in goals:
        path = utils.bfs_shortest_action_path(start, goal)
        print(f"start={start}, goal={goal}, path={utils.action_seq_to_str(path or [])}")
        assert path is not None

    non_walkable_goals = [(0, 2), (3, 1), (3, 2), (3, 3)]
    for goal in non_walkable_goals:
        path = utils.bfs_shortest_action_path(start, goal)
        print(f"start={start}, non-walkable goal={goal}, path={path}")
        assert path is None

    blocked_path = utils.bfs_shortest_action_path((1, 1), (2, 3), blocked_tiles=[(1, 2)])
    print(
        f"\nWith block at (1,2): path to (2,3) = "
        f"{utils.action_seq_to_str(blocked_path or [])}"
    )
    assert blocked_path == [utils.DOWN, utils.RIGHT, utils.RIGHT]

    print("\nPASS: BFS path checks passed.")


def test_oriented_bfs():
    print_header("TEST 5: oriented BFS checks")

    # Already at support tile but wrong facing: should find a loop if one exists.
    start_pos = (2, 2)
    start_facing = utils.UP
    goal_pos = (2, 2)
    goal_facing = utils.DOWN

    path = utils.bfs_shortest_action_path_oriented(
        start_pos=start_pos,
        start_facing=start_facing,
        goal_pos=goal_pos,
        goal_facing=goal_facing,
        blocked_tiles=None,
    )
    print(
        f"oriented path from {(start_pos, start_facing)} "
        f"to {(goal_pos, goal_facing)} = {utils.action_seq_to_str(path or [])}"
    )
    assert path is not None
    assert len(path) > 0

    print("\nPASS: oriented BFS checks passed.")


# =============================================================================
# Policy utility tests
# =============================================================================

def test_validate_policies():
    print_header("TEST 6: validate_policies")

    valid = [
        [utils.RIGHT, utils.RIGHT, utils.INTERACT],
        [utils.STAY],
        [utils.LEFT, utils.UP],
    ]
    print_policies("Valid policies", valid)
    assert utils.validate_policies(valid) is True

    bad_cases = [
        None,
        [],
        [[], [utils.RIGHT]],
        [["not_an_int"]],
        [[utils.RIGHT, 3.5]],
    ]

    for i, case in enumerate(bad_cases, start=1):
        print(f"\nBad case {i}: {case}")
        try:
            utils.validate_policies(case)
            raise AssertionError("Expected ValueError but none was raised.")
        except ValueError as e:
            print(f"Raised as expected: {e}")

    print("\nPASS: validate_policies passed.")


def test_deduplicate_and_pad():
    print_header("TEST 7: deduplicate_policies and pad_policies")

    policies = [
        [utils.RIGHT, utils.INTERACT],
        [utils.RIGHT, utils.INTERACT],
        [utils.STAY],
        [utils.LEFT, utils.LEFT, utils.INTERACT],
    ]

    print_policies("Original", policies)

    unique = utils.deduplicate_policies(policies)
    print_policies("Deduplicated", unique)

    padded, lengths = utils.pad_policies(unique, pad_action=utils.STAY)
    print("\nOriginal lengths:", lengths)
    print_policies("Padded", padded)

    assert len(unique) == 3
    assert all(len(p) == len(padded[0]) for p in padded)

    print("\nPASS: deduplicate and pad passed.")


# =============================================================================
# Semantic validity tests
# =============================================================================

def test_interaction_validity():
    print_header("TEST 8: interaction validity rules (diagnostic only)")

    for state in TEST_STATES:
        print_subheader(state["name"])
        print_state(state)

        for dest in utils.DESTINATIONS:
            valid = utils.interaction_is_valid(state, dest)
            print(f"  interact valid for {dest:>6}: {valid}")

    # This function is diagnostic only and does NOT prune policy generation.
    print("\nPASS: interaction validity inspection complete.")


# =============================================================================
# Semantic compilation tests
# =============================================================================

def test_compile_semantic_policy():
    print_header("TEST 9: compile_semantic_policy for all semantic pairs")

    total_semantics = len(utils.DESTINATIONS) * len(utils.MODES)

    for state in TEST_STATES:
        print_subheader(state["name"])
        print_state(state)

        compiled_entries = []

        for dest in utils.DESTINATIONS:
            for mode in utils.MODES:
                compiled = utils.compile_semantic_policy(state, dest, mode)
                compiled_entries.append(compiled)
                label = f"({dest}, {mode})"

                print(
                    f"  {label:20s} -> "
                    f"target={compiled['target_tile']} "
                    f"approach={compiled['approach_tile']} "
                    f"req_face={None if compiled['required_facing'] is None else utils.ACTION_NAMES[compiled['required_facing']]} "
                    f"final_face={None if compiled['final_facing'] is None else utils.ACTION_NAMES[compiled['final_facing']]} "
                    f"path={utils.action_seq_to_str(compiled['path'])} "
                    f"actions={utils.action_seq_to_str(compiled['actions'])} "
                    f"valid={compiled['valid']}"
                )

                assert compiled is not None
                assert len(compiled["actions"]) > 0

                if dest != "noop":
                    assert compiled["approach_tile"] in utils.adjacent_walkable_tiles(compiled["target_tile"])

        assert len(compiled_entries) == total_semantics

    print("\nPASS: semantic compilation inspection complete.")


def test_generate_semantic_policy_metadata():
    print_header("TEST 10: generate_semantic_policy_metadata")

    total_semantics = len(utils.DESTINATIONS) * len(utils.MODES)

    for state in TEST_STATES:
        print_subheader(state["name"])
        entries = utils.generate_semantic_policy_metadata(state)
        print_state(state)
        print_metadata(entries)

        assert len(entries) == total_semantics
        for m in entries:
            assert "destination" in m
            assert "mode" in m
            assert "actions" in m
            assert len(m["actions"]) > 0

    print("\nPASS: metadata generation passed.")


def test_generate_policies_from_state():
    print_header("TEST 11: generate_policies_from_state")

    total_semantics = len(utils.DESTINATIONS) * len(utils.MODES)

    for state in TEST_STATES:
        print_subheader(state["name"])
        print_state(state)

        policies, metadata = utils.generate_policies_from_state(
            state,
            deduplicate=False,
            pad=False,
            return_metadata=True,
        )

        print_policies("Generated policies", policies)
        print_metadata(metadata)

        assert len(policies) == total_semantics
        assert len(metadata) == total_semantics
        assert utils.validate_policies(policies) is True

        padded, lengths, metadata2 = utils.generate_policies_from_state(
            state,
            deduplicate=False,
            pad=True,
            return_metadata=True,
        )

        print("\nLengths:", lengths)
        print_policies("Generated padded policies", padded)

        assert len(padded) == total_semantics
        assert len(metadata2) == total_semantics
        assert all(len(p) == len(padded[0]) for p in padded)

    print("\nPASS: generate_policies_from_state passed.")


def test_deduplicated_generation_optional():
    print_header("TEST 12: optional deduplicated generation")

    state = TEST_STATES[0]
    print_state(state)

    policies, metadata = utils.generate_policies_from_state(
        state,
        deduplicate=True,
        pad=False,
        return_metadata=True,
    )

    print_policies("Deduplicated policies", policies)
    print_metadata(metadata)

    assert len(policies) <= len(utils.DESTINATIONS) * len(utils.MODES)
    assert len(metadata) == len(policies)

    print("\nPASS: optional deduplicated generation passed.")


# =============================================================================
# Focused expected behavior tests
# =============================================================================

def test_expected_semantic_behavior():
    print_header("TEST 13: focused behavioral expectations")

    # Case 1: empty-handed at (1,1), facing WEST, onion1 interact should be immediate interact
    state = TEST_STATES[0]
    compiled = utils.compile_semantic_policy(state, "onion1", "interact")
    print_subheader("Case 1 expectation: onion1 interact")
    print_state(state)
    print("Compiled:", compiled)
    assert compiled["actions"] == [utils.INTERACT]

    # Case 1b: same tile facing EAST — toward onion1 is WEST into station; one bump-turn, not EAST→WEST.
    state_east = {**TEST_STATES[0], "self_orient": "EAST", "name": "Case 1b: (1,1) facing EAST toward onion1"}
    compiled = utils.compile_semantic_policy(state_east, "onion1", "stay")
    print_subheader("Case 1b expectation: onion1 stay = one WEST bump-turn")
    print_state(state_east)
    print("Compiled:", compiled)
    assert compiled["actions"] == [utils.LEFT]
    compiled = utils.compile_semantic_policy(state_east, "onion1", "interact")
    print("Compiled interact:", compiled)
    assert compiled["actions"] == [utils.LEFT, utils.INTERACT]

    # Case 1: pot stay should still exist
    compiled = utils.compile_semantic_policy(state, "pot", "stay")
    print("\nCase 1 expectation: pot stay exists")
    print("Compiled:", compiled)
    assert compiled is not None
    assert len(compiled["actions"]) > 0

    # Case 2: holding onion with pot at 2 onions, pot interact exists
    state = TEST_STATES[1]
    compiled = utils.compile_semantic_policy(state, "pot", "interact")
    print_subheader("Case 2 expectation: pot interact exists")
    print_state(state)
    print("Compiled:", compiled)
    assert compiled is not None
    assert len(compiled["actions"]) > 0

    # Case 3: holding dish with ready soup, pot interact exists
    state = TEST_STATES[2]
    compiled = utils.compile_semantic_policy(state, "pot", "interact")
    print_subheader("Case 3 expectation: pot interact exists")
    print_state(state)
    print("Compiled:", compiled)
    assert compiled is not None
    assert len(compiled["actions"]) > 0

    # Case 4: holding soup, serve interact exists
    state = TEST_STATES[3]
    compiled = utils.compile_semantic_policy(state, "serve", "interact")
    print_subheader("Case 4 expectation: serve interact exists")
    print_state(state)
    print("Compiled:", compiled)
    assert compiled is not None
    assert len(compiled["actions"]) > 0

    # Case 5: cntr2 interact exists
    state = TEST_STATES[4]
    compiled = utils.compile_semantic_policy(state, "cntr2", "interact")
    print_subheader("Case 5 expectation: cntr2 interact exists")
    print_state(state)
    print("Compiled:", compiled)
    assert compiled is not None
    assert len(compiled["actions"]) > 0

    # Case 6: already adjacent to cntr5 but facing NORTH -> should still exist,
    # potentially via a reorientation loop or fallback.
    state = TEST_STATES[5]
    compiled = utils.compile_semantic_policy(state, "cntr5", "interact")
    print_subheader("Case 6 expectation: cntr5 interact still exists")
    print_state(state)
    print("Compiled:", compiled)
    assert compiled is not None
    assert len(compiled["actions"]) > 0

    print("\nPASS: focused behavioral expectations passed.")


# =============================================================================
# Sampling tests
# =============================================================================

def test_sampling_on_generated_policies():
    print_header("TEST 14: sampling on generated policies")

    state = TEST_STATES[4]
    print_state(state)

    padded_policies, lengths = utils.generate_policies_from_state(
        state,
        deduplicate=False,
        pad=True,
        return_metadata=False,
    )

    print("Lengths:", lengths)
    print_policies("Padded policies", padded_policies)

    assert len(padded_policies) == len(utils.DESTINATIONS) * len(utils.MODES)

    q_pi = np.ones(len(padded_policies), dtype=np.float64)
    q_pi /= q_pi.sum()
    print("\nUniform q_pi:", q_pi)

    det_action = utils.sample_action(
        q_pi=q_pi,
        policies=padded_policies,
        action_selection="deterministic",
    )
    print("Deterministic sample_action:", utils.ACTION_NAMES[det_action])

    det_policy_action = utils.sample_policy(
        q_pi=q_pi,
        policies=padded_policies,
        action_selection="deterministic",
    )
    print("Deterministic sample_policy:", utils.ACTION_NAMES[det_policy_action])

    np.random.seed(42)
    samples = []
    for _ in range(10):
        a = utils.sample_action(
            q_pi=q_pi,
            policies=padded_policies,
            action_selection="stochastic",
            alpha=8.0,
        )
        samples.append(utils.ACTION_NAMES[a])

    print("10 stochastic action samples:", samples)

    print("\nPASS: sampling test passed.")


# =============================================================================
# Conversion helpers
# =============================================================================

def test_qs_and_observation_helpers():
    print_header("TEST 15: qs conversion and observation helpers")

    qs_dict = {
        "self_pos": np.array([0.1, 0.7, 0.2]),
        "pot_state": np.array([0.8, 0.2]),
    }
    factor_order = ["self_pos", "pot_state"]

    print("Original qs_dict:")
    for k, v in qs_dict.items():
        print(f"  {k}: {v}")

    qs_list = utils.qs_dict_to_list(qs_dict, factor_order=factor_order)
    print("\nqs_list:", qs_list)

    qs_back = utils.qs_list_to_dict(qs_list, factor_order=factor_order)
    print("\nqs_back:", qs_back)

    assert np.allclose(qs_back["self_pos"], qs_dict["self_pos"])
    assert np.allclose(qs_back["pot_state"], qs_dict["pot_state"])

    obs_dict = {
        "self_pos_obs": 2,
        "pot_state_obs": 1,
    }
    observation_labels = {
        "self_pos_obs": [0, 1, 2, 3],
        "pot_state_obs": [0, 1],
    }

    one_hot = utils.observations_to_one_hot(obs_dict, observation_labels)
    print("\nOne-hot observations:")
    for k, v in one_hot.items():
        print(f"  {k}: {v}")

    assert np.allclose(one_hot["self_pos_obs"], np.array([0.0, 0.0, 1.0, 0.0]))
    assert np.allclose(one_hot["pot_state_obs"], np.array([0.0, 1.0]))

    print("\nPASS: qs and observation helpers passed.")


# =============================================================================
# Main
# =============================================================================

def main():
    np.set_printoptions(suppress=True, precision=4)

    print_header("RUNNING ALL utils.py TESTS")

    test_layout_sanity()
    test_adjacent_walkable_tiles()
    test_required_facing()
    test_bfs_paths()
    test_oriented_bfs()
    test_validate_policies()
    test_deduplicate_and_pad()
    test_interaction_validity()
    test_compile_semantic_policy()
    test_generate_semantic_policy_metadata()
    test_generate_policies_from_state()
    test_deduplicated_generation_optional()
    test_expected_semantic_behavior()
    test_sampling_on_generated_policies()
    test_qs_and_observation_helpers()

    print_header("ALL TESTS PASSED")


if __name__ == "__main__":
    main()