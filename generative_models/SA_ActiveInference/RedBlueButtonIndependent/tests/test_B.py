"""
Test suite for B (transition model) - comparing matrix vs functional implementations.

This test builds a matrix B from the functional B implementation and verifies
that both produce identical results under the same actions.
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from generative_models.SA_ActiveInference.RedBlueButton.B import (
    B_fn, B_agent_pos, B_static_with_noise, 
    B_red_button_state, B_blue_button_state
)
from generative_models.SA_ActiveInference.RedBlueButton.model_init import (
    state_state_dependencies, states
)


def normalize(p):
    """Normalize a probability distribution."""
    return p / np.maximum(np.sum(p), 1e-8)


def build_matrix_B_for_factor(factor_name, factor_func, state_sizes, action, 
                                width, height, noise=0.05):
    """
    Build a transition matrix B[s_next, s_prev] for a single factor and action.
    
    For factors with dependencies, this marginalizes over the other factors'
    current beliefs (mean-field approximation).
    """
    deps = state_state_dependencies[factor_name]
    factor_size = state_sizes[factor_name]
    
    # Build transition matrix
    B_matrix = np.zeros((factor_size, factor_size))
    
    if len(deps) == 1 and deps[0] == factor_name:
        # No cross-factor dependencies - simple case
        for s_prev in range(factor_size):
            # Create delta distribution at s_prev
            parents = {factor_name: np.zeros(factor_size)}
            parents[factor_name][s_prev] = 1.0
            parents[factor_name] = jnp.array(parents[factor_name])
            
            # Apply functional update
            if factor_name == "agent_pos":
                result = factor_func(parents, action, width, height, noise)
            elif factor_name in ["red_button_pos", "blue_button_pos"]:
                result = B_static_with_noise(parents, factor_name, noise)
            else:
                result = parents[factor_name]  # fallback
            
            B_matrix[:, s_prev] = np.array(result)
    else:
        # Has dependencies - need to marginalize
        # For button states, they depend on agent_pos and button_pos
        # We'll use uniform distributions for dependencies
        for s_prev in range(factor_size):
            # Create belief state
            parents = {}
            for dep in deps:
                if dep == factor_name:
                    # Delta at s_prev for this factor
                    parents[dep] = jnp.zeros(state_sizes[dep])
                    parents[dep] = parents[dep].at[s_prev].set(1.0)
                else:
                    # Uniform over other factors (mean-field assumption)
                    parents[dep] = jnp.ones(state_sizes[dep]) / state_sizes[dep]
            
            # Apply functional update
            if factor_name == "red_button_state":
                result = B_red_button_state(parents, action, noise)
            elif factor_name == "blue_button_state":
                result = B_blue_button_state(parents, action, noise)
            elif factor_name == "agent_pos":
                result = B_agent_pos(parents, action, width, height, noise)
            else:
                result = B_static_with_noise(parents, factor_name, noise)
            
            B_matrix[:, s_prev] = np.array(result)
    
    return B_matrix


def test_single_factor_comparison(factor_name, state_sizes, width, height, noise=0.05):
    """
    Test a single factor by comparing matrix and functional B.
    """
    print(f"\n{'='*70}")
    print(f"Testing factor: {factor_name}")
    print(f"{'='*70}")
    
    factor_size = state_sizes[factor_name]
    deps = state_state_dependencies[factor_name]
    
    print(f"Factor size: {factor_size}")
    print(f"Dependencies: {deps}")
    
    # Test for each action
    for action in range(6):
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "OPEN", 5: "NOOP"}
        
        # Build matrix B for this action
        if factor_name == "agent_pos":
            B_matrix = build_matrix_B_for_factor(
                factor_name, B_agent_pos, state_sizes, action, width, height, noise
            )
        elif factor_name in ["red_button_pos", "blue_button_pos"]:
            B_matrix = build_matrix_B_for_factor(
                factor_name, B_static_with_noise, state_sizes, action, width, height, noise
            )
        else:
            B_matrix = build_matrix_B_for_factor(
                factor_name, None, state_sizes, action, width, height, noise
            )
        
        # Test with a few different initial beliefs
        test_beliefs = [
            np.zeros(factor_size),  # delta at 0
            np.ones(factor_size) / factor_size,  # uniform
        ]
        test_beliefs[0][0] = 1.0
        
        # Add delta at middle position if size > 2
        if factor_size > 2:
            middle_belief = np.zeros(factor_size)
            middle_belief[factor_size // 2] = 1.0
            test_beliefs.append(middle_belief)
        
        all_match = True
        max_diff = 0.0
        
        for test_idx, qs_factor in enumerate(test_beliefs):
            # Matrix multiplication
            qs_matrix = B_matrix.dot(qs_factor)
            
            # Functional application
            # Build full qs dict with this factor and uniform for others
            qs_full = {}
            for dep in deps:
                if dep == factor_name:
                    qs_full[dep] = jnp.array(qs_factor)
                else:
                    qs_full[dep] = jnp.ones(state_sizes[dep]) / state_sizes[dep]
            
            # Apply functional B
            if factor_name == "agent_pos":
                qs_functional = np.array(B_agent_pos(qs_full, action, width, height, noise))
            elif factor_name in ["red_button_pos", "blue_button_pos"]:
                qs_functional = np.array(B_static_with_noise(qs_full, factor_name, noise))
            elif factor_name == "red_button_state":
                qs_functional = np.array(B_red_button_state(qs_full, action, noise))
            elif factor_name == "blue_button_state":
                qs_functional = np.array(B_blue_button_state(qs_full, action, noise))
            else:
                continue
            
            # Compare
            diff = np.max(np.abs(qs_matrix - qs_functional))
            max_diff = max(max_diff, diff)
            
            if diff > 1e-6:
                all_match = False
                print(f"\n  ❌ Action {action} ({action_names[action]}), test {test_idx}: MISMATCH")
                print(f"     Max diff: {diff:.6e}")
                print(f"     Matrix:     {qs_matrix}")
                print(f"     Functional: {qs_functional}")
        
        if all_match:
            print(f"  ✓ Action {action} ({action_names[action]}): MATCH (max diff: {max_diff:.6e})")
    
    print(f"\n{'='*70}\n")


def test_full_B_comparison():
    """
    Test the full B_fn by comparing with per-factor matrix multiplications.
    """
    print("\n" + "="*70)
    print("FULL B_fn COMPARISON TEST")
    print("="*70)
    
    width = 3
    height = 3
    S = width * height
    noise = 0.05
    
    # State sizes
    state_sizes = {
        "agent_pos": S,
        "red_button_pos": S,
        "blue_button_pos": S,
        "red_button_state": 2,
        "blue_button_state": 2,
    }
    
    # Test initial state
    qs_init = {
        "agent_pos": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        "red_button_pos": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "blue_button_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        "red_button_state": np.array([1.0, 0.0]),
        "blue_button_state": np.array([1.0, 0.0]),
    }
    
    for action in range(6):
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "OPEN", 5: "NOOP"}
        
        # Apply functional B_fn
        qs_functional = B_fn(qs_init, action, width, height, B_NOISE_LEVEL=noise)
        
        # Apply matrix B (independent per factor - mean field)
        qs_matrix = {}
        for factor in state_sizes.keys():
            # Build matrix for this specific factor's function
            if factor == "agent_pos":
                func = B_agent_pos
            elif factor in ["red_button_pos", "blue_button_pos"]:
                func = B_static_with_noise
            elif factor == "red_button_state":
                func = B_red_button_state
            elif factor == "blue_button_state":
                func = B_blue_button_state
            else:
                func = None
            
            B_matrix = build_matrix_B_for_factor(
                factor, func, state_sizes, action, width, height, noise
            )
            qs_matrix[factor] = B_matrix.dot(qs_init[factor])
        
        # Compare each factor
        print(f"\nAction {action} ({action_names[action]}):")
        all_match = True
        for factor in state_sizes.keys():
            diff = np.max(np.abs(np.array(qs_functional[factor]) - qs_matrix[factor]))
            if diff > 1e-6:
                all_match = False
                print(f"  ❌ {factor}: MISMATCH (diff={diff:.6e})")
            else:
                print(f"  ✓ {factor}: MATCH (diff={diff:.6e})")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    width = 3
    height = 3
    S = width * height
    noise = 0.05
    
    state_sizes = {
        "agent_pos": S,
        "red_button_pos": S,
        "blue_button_pos": S,
        "red_button_state": 2,
        "blue_button_state": 2,
    }
    
    print("\n" + "="*70)
    print("MATRIX B vs FUNCTIONAL B COMPARISON TEST")
    print("="*70)
    
    # Test individual factors
    for factor in state_sizes.keys():
        test_single_factor_comparison(factor, state_sizes, width, height, noise)
    
    # Test full B_fn
    test_full_B_comparison()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)

