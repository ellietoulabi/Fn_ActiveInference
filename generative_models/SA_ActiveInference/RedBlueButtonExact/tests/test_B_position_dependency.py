"""
Test suite specifically for position-dependent button pressing.

This test validates that button states ONLY change when:
1. Agent is at the button position
2. OPEN action is taken
3. Button is not already pressed

It tests various scenarios with different position alignments.
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from generative_models.SA_ActiveInference.RedBlueButton.B import B_fn


def test_position_dependency_red_button():
    """Test that red button pressing depends on agent position."""
    
    print("\n" + "="*70)
    print("TEST: Red Button Position Dependency")
    print("="*70)
    
    width = 3
    height = 3
    S = width * height
    noise = 0.05
    
    # Test all 9 agent positions against red button at position 5
    red_button_pos = 5
    
    print(f"\nRed button fixed at position {red_button_pos}")
    print(f"Testing agent at all {S} positions with OPEN action")
    print(f"Expected: Button presses ONLY when agent_pos == {red_button_pos}\n")
    
    results = []
    
    for agent_pos in range(S):
        # Setup: agent at specific position, red button at pos 5
        qs = {
            "agent_pos": np.zeros(S),
            "red_button_pos": np.zeros(S),
            "blue_button_pos": np.zeros(S),
            "red_button_state": np.array([1.0, 0.0]),  # not pressed
            "blue_button_state": np.array([1.0, 0.0]),  # not pressed
        }
        qs["agent_pos"][agent_pos] = 1.0
        qs["red_button_pos"][red_button_pos] = 1.0
        qs["blue_button_pos"][8] = 1.0  # blue button elsewhere
        
        # Apply OPEN action
        qs_next = B_fn(qs, action=4, width=width, height=height, B_NOISE_LEVEL=noise)
        
        # Check if button was pressed
        pressed_prob = qs_next["red_button_state"][1]
        at_button = (agent_pos == red_button_pos)
        
        results.append({
            "agent_pos": agent_pos,
            "at_button": at_button,
            "pressed_prob": pressed_prob,
        })
        
        # Validate
        if at_button:
            # Should press with 80% probability
            expected = 0.8
            if abs(pressed_prob - expected) < 0.01:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            print(f"  Agent pos {agent_pos}: AT button     → pressed={pressed_prob:.3f} (expected ~{expected:.1f}) {status}")
        else:
            # Should NOT press (stay at ~0.0, only noise)
            expected = 0.0
            if pressed_prob < 0.01:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            print(f"  Agent pos {agent_pos}: NOT at button → pressed={pressed_prob:.3f} (expected ~{expected:.1f}) {status}")
    
    # Summary
    print(f"\n{'='*70}")
    at_button_results = [r for r in results if r["at_button"]]
    not_at_button_results = [r for r in results if not r["at_button"]]
    
    at_button_pass = all(abs(r["pressed_prob"] - 0.8) < 0.01 for r in at_button_results)
    not_at_button_pass = all(r["pressed_prob"] < 0.01 for r in not_at_button_results)
    
    print(f"At button (n={len(at_button_results)}): {'✓ ALL PASS' if at_button_pass else '✗ SOME FAIL'}")
    print(f"Not at button (n={len(not_at_button_results)}): {'✓ ALL PASS' if not_at_button_pass else '✗ SOME FAIL'}")
    
    return at_button_pass and not_at_button_pass


def test_position_dependency_blue_button():
    """Test that blue button pressing depends on agent position."""
    
    print("\n" + "="*70)
    print("TEST: Blue Button Position Dependency")
    print("="*70)
    
    width = 3
    height = 3
    S = width * height
    noise = 0.05
    
    # Test all 9 agent positions against blue button at position 2
    blue_button_pos = 2
    
    print(f"\nBlue button fixed at position {blue_button_pos}")
    print(f"Testing agent at all {S} positions with OPEN action")
    print(f"Expected: Button presses ONLY when agent_pos == {blue_button_pos}\n")
    
    results = []
    
    for agent_pos in range(S):
        # Setup: agent at specific position, blue button at pos 2
        qs = {
            "agent_pos": np.zeros(S),
            "red_button_pos": np.zeros(S),
            "blue_button_pos": np.zeros(S),
            "red_button_state": np.array([1.0, 0.0]),  # not pressed
            "blue_button_state": np.array([1.0, 0.0]),  # not pressed
        }
        qs["agent_pos"][agent_pos] = 1.0
        qs["red_button_pos"][8] = 1.0  # red button elsewhere
        qs["blue_button_pos"][blue_button_pos] = 1.0
        
        # Apply OPEN action
        qs_next = B_fn(qs, action=4, width=width, height=height, B_NOISE_LEVEL=noise)
        
        # Check if button was pressed
        pressed_prob = qs_next["blue_button_state"][1]
        at_button = (agent_pos == blue_button_pos)
        
        results.append({
            "agent_pos": agent_pos,
            "at_button": at_button,
            "pressed_prob": pressed_prob,
        })
        
        # Validate
        if at_button:
            # Should press with 80% probability
            expected = 0.8
            if abs(pressed_prob - expected) < 0.01:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            print(f"  Agent pos {agent_pos}: AT button     → pressed={pressed_prob:.3f} (expected ~{expected:.1f}) {status}")
        else:
            # Should NOT press (stay at ~0.0, only noise)
            expected = 0.0
            if pressed_prob < 0.01:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            print(f"  Agent pos {agent_pos}: NOT at button → pressed={pressed_prob:.3f} (expected ~{expected:.1f}) {status}")
    
    # Summary
    print(f"\n{'='*70}")
    at_button_results = [r for r in results if r["at_button"]]
    not_at_button_results = [r for r in results if not r["at_button"]]
    
    at_button_pass = all(abs(r["pressed_prob"] - 0.8) < 0.01 for r in at_button_results)
    not_at_button_pass = all(r["pressed_prob"] < 0.01 for r in not_at_button_results)
    
    print(f"At button (n={len(at_button_results)}): {'✓ ALL PASS' if at_button_pass else '✗ SOME FAIL'}")
    print(f"Not at button (n={len(not_at_button_results)}): {'✓ ALL PASS' if not_at_button_pass else '✗ SOME FAIL'}")
    
    return at_button_pass and not_at_button_pass


def test_uncertain_position():
    """Test button pressing with uncertain agent position."""
    
    print("\n" + "="*70)
    print("TEST: Uncertain Agent Position")
    print("="*70)
    
    width = 3
    height = 3
    S = width * height
    noise = 0.05
    
    test_cases = [
        {
            "name": "50-50 at button vs not",
            "agent_dist": [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            "button_pos": 2,
            "expected_pressed": 0.5 * 0.8,  # 50% at button * 80% success
        },
        {
            "name": "25-75 at button vs not",
            "agent_dist": [0.0, 0.0, 0.25, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0],
            "button_pos": 2,
            "expected_pressed": 0.25 * 0.8,  # 25% at button * 80% success
        },
        {
            "name": "Uniform distribution",
            "agent_dist": [1/9] * 9,
            "button_pos": 4,
            "expected_pressed": (1/9) * 0.8,  # 1/9 at button * 80% success
        },
        {
            "name": "100% at button",
            "agent_dist": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "button_pos": 2,
            "expected_pressed": 1.0 * 0.8,  # 100% at button * 80% success
        },
    ]
    
    all_pass = True
    
    for tc in test_cases:
        qs = {
            "agent_pos": np.array(tc["agent_dist"]),
            "red_button_pos": np.zeros(S),
            "blue_button_pos": np.zeros(S),
            "red_button_state": np.array([1.0, 0.0]),
            "blue_button_state": np.array([1.0, 0.0]),
        }
        qs["red_button_pos"][tc["button_pos"]] = 1.0
        qs["blue_button_pos"][8] = 1.0
        
        # Apply OPEN action
        qs_next = B_fn(qs, action=4, width=width, height=height, B_NOISE_LEVEL=noise)
        
        pressed_prob = qs_next["red_button_state"][1]
        expected = tc["expected_pressed"]
        diff = abs(pressed_prob - expected)
        
        # Allow 2% tolerance
        if diff < 0.02:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_pass = False
        
        print(f"\n  {tc['name']}:")
        print(f"    Button pos: {tc['button_pos']}")
        print(f"    Agent dist: {[f'{p:.3f}' for p in tc['agent_dist'] if p > 0.01]}")
        print(f"    Pressed prob: {pressed_prob:.4f}")
        print(f"    Expected:     {expected:.4f}")
        print(f"    Difference:   {diff:.4f}")
        print(f"    Status: {status}")
    
    print(f"\n{'='*70}")
    print(f"Overall: {'✓ ALL PASS' if all_pass else '✗ SOME FAIL'}")
    
    return all_pass


def test_non_open_actions_dont_press():
    """Test that non-OPEN actions never press buttons regardless of position."""
    
    print("\n" + "="*70)
    print("TEST: Non-OPEN Actions Don't Press Buttons")
    print("="*70)
    
    width = 3
    height = 3
    S = width * height
    noise = 0.05
    
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 5: "NOOP"}
    
    # Agent at button position
    qs = {
        "agent_pos": np.zeros(S),
        "red_button_pos": np.zeros(S),
        "blue_button_pos": np.zeros(S),
        "red_button_state": np.array([1.0, 0.0]),
        "blue_button_state": np.array([1.0, 0.0]),
    }
    qs["agent_pos"][4] = 1.0
    qs["red_button_pos"][4] = 1.0  # Same position!
    qs["blue_button_pos"][4] = 1.0  # Same position!
    
    print("\nAgent AT button positions (pos 4)")
    print("Testing that non-OPEN actions don't press buttons\n")
    
    all_pass = True
    
    for action in [0, 1, 2, 3, 5]:
        qs_next = B_fn(qs, action, width, height, B_NOISE_LEVEL=noise)
        
        red_pressed = qs_next["red_button_state"][1]
        blue_pressed = qs_next["blue_button_state"][1]
        
        # Should remain not pressed (< 1% due to noise only)
        if red_pressed < 0.01 and blue_pressed < 0.01:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_pass = False
        
        print(f"  Action {action} ({action_names[action]:5s}): " +
              f"red={red_pressed:.4f}, blue={blue_pressed:.4f} {status}")
    
    print(f"\n{'='*70}")
    print(f"Overall: {'✓ ALL PASS' if all_pass else '✗ SOME FAIL'}")
    
    return all_pass


def test_already_pressed_stays_pressed():
    """Test that already-pressed buttons stay pressed."""
    
    print("\n" + "="*70)
    print("TEST: Already Pressed Buttons Stay Pressed")
    print("="*70)
    
    width = 3
    height = 3
    S = width * height
    noise = 0.05
    
    # Test with agent at different positions
    test_positions = [
        {"agent": 0, "button": 0, "at_button": True},
        {"agent": 4, "button": 0, "at_button": False},
    ]
    
    all_pass = True
    
    for tp in test_positions:
        qs = {
            "agent_pos": np.zeros(S),
            "red_button_pos": np.zeros(S),
            "blue_button_pos": np.zeros(S),
            "red_button_state": np.array([0.0, 1.0]),  # Already pressed!
            "blue_button_state": np.array([1.0, 0.0]),
        }
        qs["agent_pos"][tp["agent"]] = 1.0
        qs["red_button_pos"][tp["button"]] = 1.0
        qs["blue_button_pos"][8] = 1.0
        
        # Apply OPEN action
        qs_next = B_fn(qs, action=4, width=width, height=height, B_NOISE_LEVEL=noise)
        
        pressed_prob = qs_next["red_button_state"][1]
        
        # Should stay pressed (close to 1.0)
        if pressed_prob > 0.99:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_pass = False
        
        location = "at button" if tp["at_button"] else "NOT at button"
        print(f"\n  Agent {location} (agent={tp['agent']}, button={tp['button']}):")
        print(f"    Pressed prob: {pressed_prob:.4f} (expected ~1.0) {status}")
    
    print(f"\n{'='*70}")
    print(f"Overall: {'✓ ALL PASS' if all_pass else '✗ SOME FAIL'}")
    
    return all_pass


if __name__ == "__main__":
    print("\n" + "="*70)
    print("POSITION-DEPENDENT BUTTON PRESSING TEST SUITE")
    print("="*70)
    
    results = {}
    
    results["red_button"] = test_position_dependency_red_button()
    results["blue_button"] = test_position_dependency_blue_button()
    results["uncertain_position"] = test_uncertain_position()
    results["non_open_actions"] = test_non_open_actions_dont_press()
    results["already_pressed"] = test_already_pressed_stays_pressed()
    
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
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("="*70 + "\n")

