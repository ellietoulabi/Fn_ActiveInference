"""
Test suite for A (observation model).

Tests the functional A implementation to ensure observation distributions
are computed correctly based on state factor dependencies.
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from generative_models.SA_ActiveInference.RedBlueButton.A import (
    A_fn, A_agent_pos, A_on_red_button, A_on_blue_button,
    A_red_button_state, A_blue_button_state, A_game_result,
    A_button_just_pressed
)


def test_agent_pos_observation():
    """Test agent position observation with noise."""
    print("\n" + "="*70)
    print("TEST: Agent Position Observation")
    print("="*70)
    
    S = 9
    
    # Test 1: Agent certain at position 4
    state = {"agent_pos": np.zeros(S)}
    state["agent_pos"][4] = 1.0
    state["agent_pos"] = jnp.array(state["agent_pos"])
    
    obs_dist = A_agent_pos(state, num_obs=S)
    
    print("\nTest 1: Agent certain at position 4")
    print(f"  Observation distribution:")
    for i, p in enumerate(obs_dist):
        if p > 0.001:
            print(f"    obs {i}: {p:.4f}")
    
    # Should be mostly at position 4
    assert obs_dist[4] > 0.9, "Position 4 should have highest probability"
    assert np.sum(obs_dist) - 1.0 < 1e-6, "Should sum to 1"
    print("  ✓ PASS")
    
    # Test 2: Agent uniform
    state = {"agent_pos": jnp.ones(S) / S}
    obs_dist = A_agent_pos(state, num_obs=S)
    
    print("\nTest 2: Agent uniform over all positions")
    print(f"  All observations should be roughly uniform: ~{1/S:.3f}")
    print(f"  Max deviation: {np.max(np.abs(obs_dist - 1/S)):.4f}")
    assert np.max(np.abs(obs_dist - 1/S)) < 0.02, "Should be roughly uniform"
    print("  ✓ PASS")
    
    return True


def test_on_button_observations():
    """Test on_red_button and on_blue_button observations."""
    print("\n" + "="*70)
    print("TEST: On Button Observations")
    print("="*70)
    
    S = 9
    
    # Test 1: Agent at red button (both at position 2)
    state = {
        "agent_pos": np.zeros(S),
        "red_button_pos": np.zeros(S),
    }
    state["agent_pos"][2] = 1.0
    state["red_button_pos"][2] = 1.0
    for k in state:
        state[k] = jnp.array(state[k])
    
    obs_dist = A_on_red_button(state, num_obs=2)
    
    print("\nTest 1: Agent at red button (both at pos 2)")
    print(f"  On button: FALSE={obs_dist[0]:.4f}, TRUE={obs_dist[1]:.4f}")
    assert obs_dist[1] > 0.99, "Should observe TRUE with high probability"
    print("  ✓ PASS")
    
    # Test 2: Agent NOT at red button
    state["agent_pos"] = jnp.zeros(S).at[5].set(1.0)
    obs_dist = A_on_red_button(state, num_obs=2)
    
    print("\nTest 2: Agent NOT at red button (agent=5, button=2)")
    print(f"  On button: FALSE={obs_dist[0]:.4f}, TRUE={obs_dist[1]:.4f}")
    assert obs_dist[0] > 0.99, "Should observe FALSE with high probability"
    print("  ✓ PASS")
    
    # Test 3: Uncertain position (50-50)
    state["agent_pos"] = jnp.zeros(S).at[2].set(0.5).at[5].set(0.5)
    obs_dist = A_on_red_button(state, num_obs=2)
    
    print("\nTest 3: Agent 50% at button, 50% not")
    print(f"  On button: FALSE={obs_dist[0]:.4f}, TRUE={obs_dist[1]:.4f}")
    assert abs(obs_dist[1] - 0.5) < 0.01, "Should be 50-50"
    print("  ✓ PASS")
    
    return True


def test_button_state_observations():
    """Test direct button state observations."""
    print("\n" + "="*70)
    print("TEST: Button State Observations")
    print("="*70)
    
    # Test 1: Red button not pressed
    state = {"red_button_state": jnp.array([1.0, 0.0])}
    obs_dist = A_red_button_state(state, num_obs=2)
    
    print("\nTest 1: Red button not pressed")
    print(f"  Obs: not_pressed={obs_dist[0]:.4f}, pressed={obs_dist[1]:.4f}")
    assert obs_dist[0] > 0.99, "Should observe not_pressed"
    print("  ✓ PASS")
    
    # Test 2: Red button pressed
    state = {"red_button_state": jnp.array([0.0, 1.0])}
    obs_dist = A_red_button_state(state, num_obs=2)
    
    print("\nTest 2: Red button pressed")
    print(f"  Obs: not_pressed={obs_dist[0]:.4f}, pressed={obs_dist[1]:.4f}")
    assert obs_dist[1] > 0.99, "Should observe pressed"
    print("  ✓ PASS")
    
    # Test 3: Uncertain button state
    state = {"red_button_state": jnp.array([0.3, 0.7])}
    obs_dist = A_red_button_state(state, num_obs=2)
    
    print("\nTest 3: Uncertain button state (30% not pressed, 70% pressed)")
    print(f"  Obs: not_pressed={obs_dist[0]:.4f}, pressed={obs_dist[1]:.4f}")
    assert abs(obs_dist[0] - 0.3) < 0.01, "Should match belief"
    assert abs(obs_dist[1] - 0.7) < 0.01, "Should match belief"
    print("  ✓ PASS")
    
    return True


def test_game_result_observation():
    """Test game result observations."""
    print("\n" + "="*70)
    print("TEST: Game Result Observation")
    print("="*70)
    
    # Test 1: Both buttons pressed (WIN)
    state = {
        "red_button_state": jnp.array([0.0, 1.0]),
        "blue_button_state": jnp.array([0.0, 1.0]),
    }
    obs_dist = A_game_result(state, num_obs=3)
    
    print("\nTest 1: Both buttons pressed")
    print(f"  Result: neutral={obs_dist[0]:.4f}, win={obs_dist[1]:.4f}, lose={obs_dist[2]:.4f}")
    assert obs_dist[1] > 0.99, "Should observe WIN"
    print("  ✓ PASS")
    
    # Test 2: Only blue pressed (LOSE)
    state = {
        "red_button_state": jnp.array([1.0, 0.0]),
        "blue_button_state": jnp.array([0.0, 1.0]),
    }
    obs_dist = A_game_result(state, num_obs=3)
    
    print("\nTest 2: Only blue pressed")
    print(f"  Result: neutral={obs_dist[0]:.4f}, win={obs_dist[1]:.4f}, lose={obs_dist[2]:.4f}")
    assert obs_dist[2] > 0.99, "Should observe LOSE"
    print("  ✓ PASS")
    
    # Test 3: Neither pressed (NEUTRAL)
    state = {
        "red_button_state": jnp.array([1.0, 0.0]),
        "blue_button_state": jnp.array([1.0, 0.0]),
    }
    obs_dist = A_game_result(state, num_obs=3)
    
    print("\nTest 3: Neither button pressed")
    print(f"  Result: neutral={obs_dist[0]:.4f}, win={obs_dist[1]:.4f}, lose={obs_dist[2]:.4f}")
    assert obs_dist[0] > 0.99, "Should observe NEUTRAL"
    print("  ✓ PASS")
    
    # Test 4: Only red pressed (NEUTRAL)
    state = {
        "red_button_state": jnp.array([0.0, 1.0]),
        "blue_button_state": jnp.array([1.0, 0.0]),
    }
    obs_dist = A_game_result(state, num_obs=3)
    
    print("\nTest 4: Only red pressed")
    print(f"  Result: neutral={obs_dist[0]:.4f}, win={obs_dist[1]:.4f}, lose={obs_dist[2]:.4f}")
    assert obs_dist[0] > 0.99, "Should observe NEUTRAL"
    print("  ✓ PASS")
    
    return True


def test_full_A_fn():
    """Test the full A_fn that computes all modalities."""
    print("\n" + "="*70)
    print("TEST: Full A_fn")
    print("="*70)
    
    S = 9
    
    # Setup state
    state_factors = {
        "agent_pos": np.zeros(S),
        "red_button_pos": np.zeros(S),
        "blue_button_pos": np.zeros(S),
        "red_button_state": np.array([1.0, 0.0]),
        "blue_button_state": np.array([1.0, 0.0]),
    }
    state_factors["agent_pos"][4] = 1.0
    state_factors["red_button_pos"][2] = 1.0
    state_factors["blue_button_pos"][6] = 1.0
    
    # Convert to JAX
    for k in state_factors:
        state_factors[k] = jnp.array(state_factors[k])
    
    # Apply A_fn
    obs_dists = A_fn(state_factors)
    
    print("\nState configuration:")
    print(f"  Agent at pos 4")
    print(f"  Red button at pos 2")
    print(f"  Blue button at pos 6")
    print(f"  Both buttons not pressed")
    
    print("\nObservation distributions:")
    print(f"  agent_pos: mode={np.argmax(obs_dists['agent_pos'])}")
    print(f"  on_red_button: FALSE={obs_dists['on_red_button'][0]:.3f}, TRUE={obs_dists['on_red_button'][1]:.3f}")
    print(f"  on_blue_button: FALSE={obs_dists['on_blue_button'][0]:.3f}, TRUE={obs_dists['on_blue_button'][1]:.3f}")
    print(f"  red_button_state: not_pressed={obs_dists['red_button_state'][0]:.3f}")
    print(f"  blue_button_state: not_pressed={obs_dists['blue_button_state'][0]:.3f}")
    print(f"  game_result: neutral={obs_dists['game_result'][0]:.3f}, win={obs_dists['game_result'][1]:.3f}, lose={obs_dists['game_result'][2]:.3f}")
    
    # Validate
    assert np.argmax(obs_dists['agent_pos']) == 4, "Agent pos should be 4"
    assert obs_dists['on_red_button'][0] > 0.9, "Should NOT be on red button"
    assert obs_dists['on_blue_button'][0] > 0.9, "Should NOT be on blue button"
    assert obs_dists['game_result'][0] > 0.9, "Should be NEUTRAL"
    
    print("\n  ✓ ALL CHECKS PASS")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FUNCTIONAL A TEST SUITE")
    print("="*70)
    
    results = {}
    
    results["agent_pos"] = test_agent_pos_observation()
    results["on_button"] = test_on_button_observations()
    results["button_state"] = test_button_state_observations()
    results["game_result"] = test_game_result_observation()
    results["full_A_fn"] = test_full_A_fn()
    
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

