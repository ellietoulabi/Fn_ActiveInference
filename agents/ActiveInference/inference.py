"""
State inference for Active Inference with functional generative model.

This module implements belief updating algorithms using functional A_fn.
Main algorithm: Variational Fixed-Point Iteration (FPI) for mean-field inference.
"""

import numpy as np
from . import maths


def vanilla_fpi_update_posterior_states(
    A_fn,
    obs_dict,
    prior_dict,
    state_factors,
    state_sizes,
    num_iter=16,
    dF_tol=0.001,
    debug=False,
):
    """
    Mean-field variational inference to update posterior beliefs over hidden states.
    
    Uses coordinate ascent to minimize variational free energy, iteratively updating
    each factor's marginal belief until convergence.

    Args:
        A_fn: functional observation model (takes state_indices → obs likelihoods)
        obs_dict: dict of observed indices per modality
            e.g., {'agent_pos': 3, 'on_red_button': 1, ...}
        prior_dict: dict of prior beliefs per factor
            e.g., {'agent_pos': array([...]), 'red_button_pos': array([...]), ...}
        state_factors: list of state factor names (defines order)
            e.g., ['agent_pos', 'red_button_pos', 'blue_button_pos', ...]
        state_sizes: dict mapping factor names to sizes
            e.g., {'agent_pos': 9, 'red_button_pos': 9, ...}
        num_iter: maximum number of iterations
        dF_tol: convergence tolerance for free energy change
        debug: if True, print debug information

    Returns:
        qs_dict: dict of posterior beliefs per factor (same structure as prior_dict)
    
    Algorithm:
        1. Initialize qs to uniform (or prior)
        2. For each iteration:
            a. For each factor f:
                - Marginalize likelihood over other factors
                - Update q(s_f) ∝ p(s_f) * Σ_{s_-f} q(s_-f) p(o | s_f, s_-f)
            b. Check convergence
        3. Return converged beliefs
    
    Notes:
        - Uses mean-field approximation: q(s) = Π_f q(s_f)
        - Works directly with functional A, no matrix construction
        - More memory efficient than matrix-based inference for large state spaces
    """
    num_factors = len(state_factors)
    
    # Debug: print observations
    if debug:
        print("\n" + "="*60)
        print("INFERENCE DEBUG")
        print("="*60)
        print("Observations:", obs_dict)
        if obs_dict.get('on_blue_button') == 1:
            print(">>> AGENT IS ON BLUE BUTTON! <<<")
        if obs_dict.get('on_red_button') == 1:
            print(">>> AGENT IS ON RED BUTTON! <<<")
    
    # Initialize posterior beliefs (uniform or from prior)
    qs_dict = {}
    for factor in state_factors:
        if factor in prior_dict:
            qs_dict[factor] = prior_dict[factor].copy()
        else:
            # Uniform initialization
            size = state_sizes[factor]
            qs_dict[factor] = np.ones(size) / size
    
    # Convert priors to log space for numerical stability
    log_prior_dict = {f: maths.log_stable(prior_dict[f]) for f in state_factors}
    
    # Initialize free energy tracking
    prev_vfe = maths.calc_variational_free_energy(qs_dict, prior_dict)
    
    # Coordinate ascent loop
    for iteration in range(num_iter):
        # Update each factor in turn
        for factor_idx, factor in enumerate(state_factors):
            # Get size of this factor
            factor_size = state_sizes[factor]
            
            # Compute marginal likelihood for each value of this factor
            # by marginalizing over other factors using current beliefs
            log_likelihood_factor = np.zeros(factor_size)
            
            for s_f in range(factor_size):
                # Get all plausible states where this factor = s_f
                # and marginalize over other factors
                
                # Build list of states to marginalize over
                other_factors = [f for f in state_factors if f != factor]
                
                # Get plausible values for other factors (to reduce computation)
                # For directly observed factors, clamp to observed value
                plausible_combos = _get_plausible_combinations(
                    qs_dict, other_factors, obs_dict, threshold=1e-10
                )
                
                # Marginalize: sum over other factors weighted by their beliefs
                marginal_likelihood = 0.0
                for other_indices, joint_prob_others in plausible_combos:
                    # Build full state indices with this factor fixed at s_f
                    state_indices = other_indices.copy()
                    state_indices[factor] = s_f
                    
                    # Compute likelihood p(o | s) for this state configuration
                    likelihood = maths.compute_likelihood_for_state(
                        A_fn, obs_dict, state_indices
                    )
                    
                    # Debug: check what A_fn returns for blue button
                    if debug and factor == 'blue_button_pos' and obs_dict.get('on_blue_button') == 1 and iteration == 0 and s_f == 2:
                        obs_liks = A_fn(state_indices)
                        print(f"    s_f={s_f}, state={state_indices}, likelihood={likelihood:.6e}")
                        for mod, obs_idx in obs_dict.items():
                            if mod in obs_liks:
                                lik_val = obs_liks[mod][obs_idx]
                                print(f"      {mod}: obs_lik[{obs_idx}] = {lik_val:.6e}")
                    
                    # Accumulate weighted by beliefs over other factors
                    marginal_likelihood += joint_prob_others * likelihood
                
                # Store log likelihood
                log_likelihood_factor[s_f] = maths.log_stable(marginal_likelihood)
            
            # Update factor belief: q(s_f) ∝ p(s_f) * Σ_{s_-f} q(s_-f) p(o|s)
            log_posterior = log_prior_dict[factor] + log_likelihood_factor
            qs_dict[factor] = maths.softmax(log_posterior)
            
            # Debug button position updates
            if debug and factor == 'blue_button_pos' and obs_dict.get('on_blue_button') == 1:
                print(f"\n[Iter {iteration}] Updating blue_button_pos:")
                print(f"  Agent is at position: {obs_dict.get('agent_pos', '?')}")
                print(f"  Log likelihood: {log_likelihood_factor}")
                print(f"  Updated belief (top 3): {np.argsort(qs_dict[factor])[-3:][::-1]} = {np.sort(qs_dict[factor])[-3:][::-1]}")
                print(f"  Belief at agent position {obs_dict.get('agent_pos')}: {qs_dict[factor][obs_dict.get('agent_pos', 0)]:.4f}")
        
        # Check convergence
        vfe = maths.calc_variational_free_energy(qs_dict, prior_dict)
        dF = abs(prev_vfe - vfe)
        
        if dF < dF_tol:
            break
        
        prev_vfe = vfe
    
    return qs_dict


def _get_plausible_combinations(qs_dict, factors, obs_dict=None, threshold=1e-10):
    """
    Get plausible combinations of factor values with their joint probabilities.
    
    Helper for efficient marginalization during inference.

    Args:
        qs_dict: dict of current beliefs
        factors: list of factor names to combine
        obs_dict: dict of observations (to clamp observed state factors)
        threshold: minimum probability threshold

    Returns:
        list of (indices_dict, joint_prob) tuples
    """
    import itertools
    
    if len(factors) == 0:
        return [({}, 1.0)]
    
    # Get plausible indices for each factor
    plausible_indices = {}
    for factor in factors:
        # Check if this factor corresponds to a direct state observation
        # For 'agent_pos', the observation is directly the state
        if obs_dict is not None and factor in obs_dict:
            # Clamp to observed value
            plausible_indices[factor] = [obs_dict[factor]]
        else:
            # Use belief distribution
            qs_f = qs_dict[factor]
            plausible_indices[factor] = np.where(qs_f > threshold)[0]
    
    # Generate combinations
    combinations = []
    for indices in itertools.product(*[plausible_indices[f] for f in factors]):
        indices_dict = {factor: idx for factor, idx in zip(factors, indices)}
        
        # Compute joint probability (mean-field assumption)
        joint_prob = 1.0
        for factor, idx in indices_dict.items():
            # If factor is observed, use probability 1.0
            if obs_dict is not None and factor in obs_dict and obs_dict[factor] == idx:
                joint_prob *= 1.0
            else:
                joint_prob *= qs_dict[factor][idx]
        
        if joint_prob > threshold:
            combinations.append((indices_dict, joint_prob))
    
    return combinations


# =============================================================================
# Alternative: Simplified Single-Factor Inference
# =============================================================================

def update_single_factor_belief(
    A_fn,
    obs_dict,
    prior_factor,
    factor_name,
    factor_size,
    other_beliefs_dict,
    state_factors,
):
    """
    Update belief for a single factor given fixed beliefs over others.
    
    This is a building block for coordinate ascent algorithms.
    
    Args:
        A_fn: functional observation model
        obs_dict: observed indices
        prior_factor: prior belief for this factor
        factor_name: name of factor to update
        factor_size: number of states for this factor
        other_beliefs_dict: fixed beliefs for other factors
        state_factors: list of all factor names
    
    Returns:
        updated belief for this factor
    """
    log_likelihood = np.zeros(factor_size)
    
    for s_f in range(factor_size):
        # Build state with this factor at s_f
        state_indices = other_beliefs_dict.copy()
        state_indices[factor_name] = s_f
        
        # For other factors, we need to marginalize
        # This is simplified - assumes point estimates for other factors
        likelihood = maths.compute_likelihood_for_state(A_fn, obs_dict, state_indices)
        log_likelihood[s_f] = maths.log_stable(likelihood)
    
    # Combine with prior
    log_prior = maths.log_stable(prior_factor)
    log_posterior = log_prior + log_likelihood
    
    return maths.softmax(log_posterior)


# =============================================================================
# Utility: Get Most Likely State Configuration
# =============================================================================

def get_map_state(qs_dict):
    """
    Get Maximum A Posteriori (MAP) state configuration.
    
    Returns the most likely value for each factor under current beliefs.
    
    Args:
        qs_dict: dict of posterior beliefs
    
    Returns:
        map_state: dict mapping factor names to most likely indices
    
    Examples:
        >>> qs = {'agent_pos': np.array([0.1, 0.7, 0.2]), 
        ...       'button_state': np.array([0.3, 0.7])}
        >>> get_map_state(qs)
        {'agent_pos': 1, 'button_state': 1}
    """
    return {factor: int(np.argmax(belief)) for factor, belief in qs_dict.items()}


def get_expected_state_indices(qs_dict):
    """
    Get expected (mean) state indices.
    
    Returns expected value for each factor (not necessarily integer).
    
    Args:
        qs_dict: dict of posterior beliefs
    
    Returns:
        expected_state: dict mapping factor names to expected indices
    """
    expected_state = {}
    for factor, belief in qs_dict.items():
        indices = np.arange(len(belief))
        expected_state[factor] = float(np.sum(indices * belief))
    return expected_state


# =============================================================================
# Debugging/Monitoring Utilities
# =============================================================================

def compute_inference_diagnostics(qs_dict, prior_dict, obs_dict, A_fn, state_factors, state_sizes):
    """
    Compute diagnostic metrics for inference quality.
    
    Args:
        qs_dict: posterior beliefs
        prior_dict: prior beliefs
        obs_dict: observations
        A_fn: observation model
        state_factors: list of factor names
        state_sizes: dict of factor sizes
    
    Returns:
        diagnostics: dict with various metrics
    """
    diagnostics = {}
    
    # Variational free energy
    diagnostics['vfe'] = maths.calc_variational_free_energy(qs_dict, prior_dict)
    
    # Entropy of each factor
    diagnostics['entropy'] = {}
    for factor, belief in qs_dict.items():
        H = -np.sum(belief * maths.log_stable(belief))
        diagnostics['entropy'][factor] = float(H)
    
    # MAP state
    diagnostics['map_state'] = get_map_state(qs_dict)
    
    # Concentration (how peaked is the belief?)
    diagnostics['concentration'] = {}
    for factor, belief in qs_dict.items():
        max_prob = np.max(belief)
        diagnostics['concentration'][factor] = float(max_prob)
    
    return diagnostics
