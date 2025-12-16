"""
Policy evaluation and action selection for Active Inference with functional generative model.

This module implements:
- Expected state computation using B_fn
- Expected observation prediction using A_fn
- Expected Free Energy (EFE) calculation
- Policy posterior inference
- Action selection
"""

import numpy as np
from . import maths
from . import utils


# =============================================================================
# Expected State Prediction (using B_fn)
# =============================================================================

def get_expected_state(B_fn, qs_current, action, env_params):
    """
    Compute expected next state distribution given current beliefs and action.
    
    Uses functional B to propagate beliefs forward one step.

    Args:
        B_fn: functional transition model (qs, action) → next_qs
        qs_current: dict of current belief distributions per factor
        action: int, action to take
        env_params: dict with environment parameters (width, height, etc.)

    Returns:
        qs_next: dict of predicted next state beliefs
    
    Examples:
        >>> qs = {'agent_pos': np.array([1.0, 0, 0, ...]), ...}
        >>> action = 1  # DOWN
        >>> qs_next = get_expected_state(B_fn, qs, action, env_params)
    """
    # B_fn directly handles belief propagation
    return B_fn(qs_current, action, **env_params)


def get_expected_states(B_fn, qs_current, policy, env_params):
    """
    Roll out expected states under a sequence of actions (policy).

    Args:
        B_fn: functional transition model
        qs_current: dict of current beliefs
        policy: list of actions [a_0, a_1, ..., a_T]
        env_params: dict with environment parameters

    Returns:
        qs_pred: list of dicts, one per timestep
            qs_pred[t] is the predicted belief at step t+1
    
    Examples:
        >>> policy = [1, 3, 4]  # DOWN, RIGHT, OPEN
        >>> qs_pred = get_expected_states(B_fn, qs, policy, env_params)
        >>> len(qs_pred)
        3
    """
    if np.isscalar(policy):
        policy = [int(policy)]

    qs_pred = []
    qs_t = qs_current

    for action in policy:
        qs_next = B_fn(qs_t, int(action), **env_params)
        qs_pred.append(qs_next)
        qs_t = qs_next

    return qs_pred


# =============================================================================
# Expected Observation Prediction (using A_fn)
# =============================================================================

def get_expected_obs_from_beliefs(A_fn, qs_dict, state_factors, state_sizes,
                                   observation_labels=None, observation_state_dependencies=None):
    """
    Compute expected observation distributions from belief over states.
    
    Optimized: enumerate unique state configurations once, reuse across modalities.
    """
    # Import default model_init for backward compatibility
    if observation_labels is None or observation_state_dependencies is None:
        from generative_models.SA_ActiveInference.RedBlueButton import model_init as default_model
        if observation_labels is None:
            observation_labels = default_model.observations
        if observation_state_dependencies is None:
            observation_state_dependencies = default_model.observation_state_dependencies
    import itertools
    
    # Convert JAX arrays to numpy to avoid compilation overhead
    qs_dict_np = {f: np.array(qs_dict[f]) for f in state_factors}
    
    map_indices = {f: int(np.argmax(qs_dict_np[f])) for f in state_factors}
    SKIP_MODALITIES = {'button_just_pressed'}
    
    # Adaptive entropy threshold based on belief concentration
    max_entropy_observed = max(
        -np.sum(qs_dict_np[f] * np.log(qs_dict_np[f] + 1e-16))
        for f in state_factors
    )
    ENTROPY_THRESHOLD = min(0.1, max(0.01, max_entropy_observed * 0.1))
    
    dynamic_factors = set()
    for f in state_factors:
        q_f = qs_dict_np[f]  # Use numpy version
        entropy = -np.sum(q_f * np.log(q_f + 1e-16))
        if entropy > ENTROPY_THRESHOLD:
            dynamic_factors.add(f)
    
    # Find deps that are actually dynamic
    all_deps = set()
    for modality, deps in observation_state_dependencies.items():
        if modality not in SKIP_MODALITIES:
            for dep in deps:
                if dep in dynamic_factors:
                    all_deps.add(dep)
    
    # Enumerate combinations of DYNAMIC factors only
    dep_list = sorted(all_deps)
    dep_ranges = [range(len(qs_dict_np[dep])) for dep in dep_list]
    
    # Precompute likelihoods for all state combinations
    likelihood_cache = []
    prob_cache = []
    
    for combo in itertools.product(*dep_ranges):
        # Compute joint prob
        joint_prob = 1.0
        state_indices = map_indices.copy()
        for dep, idx in zip(dep_list, combo):
            joint_prob *= qs_dict_np[dep][idx]  # Use numpy version
            state_indices[dep] = int(idx)
        
        if joint_prob <= 1e-16:
            continue
        
        # Call A_fn ONCE for this state
        obs_likelihoods = A_fn(state_indices)
        likelihood_cache.append(obs_likelihoods)
        prob_cache.append(joint_prob)
    
    # Now marginalize each modality using cached likelihoods
    qo_dict = {}
    for modality, deps in observation_state_dependencies.items():
        if modality in SKIP_MODALITIES:
            continue
        
        num_obs = len(observation_labels[modality])
        qo_m = np.zeros(num_obs)
        
        for obs_lik, joint_prob in zip(likelihood_cache, prob_cache):
            p_o_m = obs_lik[modality]
            qo_m += joint_prob * p_o_m
        
        qo_dict[modality] = maths.normalize(qo_m)
    
    # Approximate button_just_pressed (works with both SA and MA naming)
    if 'button_just_pressed' in observation_state_dependencies:
        if 'on_red_button' in qo_dict:
            p_on_red = qo_dict['on_red_button'][1]
            p_on_blue = qo_dict['on_blue_button'][1]
        elif 'my_on_red_button' in qo_dict:
            p_on_red = qo_dict['my_on_red_button'][1]
            p_on_blue = qo_dict['my_on_blue_button'][1]
        else:
            p_on_red = 0.0
            p_on_blue = 0.0
        p_just_pressed = min(1.0, p_on_red + p_on_blue)
        qo_dict['button_just_pressed'] = np.array([1.0 - p_just_pressed, p_just_pressed])
    
    return qo_dict


def get_expected_obs_sequence(A_fn, qs_pi, state_factors, state_sizes):
    """
    Compute expected observations over time under a policy.
    
    Args:
        A_fn: functional observation model
        qs_pi: list of belief dicts over time
        state_factors: list of factor names
        state_sizes: dict of factor sizes
    
    Returns:
        qo_pi: list of dicts, one per timestep
            qo_pi[t][modality] is predicted observation distribution at step t
    """
    qo_pi = []
    
    for qs_t in qs_pi:
        qo_t = get_expected_obs_from_beliefs(A_fn, qs_t, state_factors, state_sizes)
        qo_pi.append(qo_t)
    
    return qo_pi


def get_expected_obs_and_info_gain_unified(A_fn, qs_pi, state_factors, state_sizes, observation_labels, 
                                            observation_state_dependencies=None, debug=False):
    """
    UNIFIED: Compute BOTH expected observations AND info gain in ONE pass.
    
    This avoids redundant state enumeration by computing both metrics from the
    same cached A_fn calls.
    
    Returns:
        qo_pi: list of observation predictions per timestep
        total_info_gain: float, sum of Bayesian surprise over timesteps
    """
    # Import default model_init for backward compatibility
    if observation_state_dependencies is None:
        from generative_models.SA_ActiveInference.RedBlueButton import model_init as default_model
        observation_state_dependencies = default_model.observation_state_dependencies
    import itertools
    
    qo_pi = []
    total_info_gain = 0.0
    
    for t_idx, qs_t in enumerate(qs_pi):
        # Convert to numpy once
        qs_dict_np = {f: np.array(qs_t[f]) for f in state_factors}
        map_indices = {f: int(np.argmax(qs_dict_np[f])) for f in state_factors}
        
        # Find dynamic factors (adaptive entropy threshold)
        # When beliefs are very concentrated, use lower threshold
        # When beliefs are uncertain, use higher threshold to reduce computation
        max_entropy_observed = max(
            -np.sum(qs_dict_np[f] * np.log(qs_dict_np[f] + 1e-16))
            for f in state_factors
        )
        # Adaptive threshold: 0.01 when concentrated, 0.1 when very uncertain
        ENTROPY_THRESHOLD = min(0.1, max(0.01, max_entropy_observed * 0.1))
        
        dynamic_factors = set()
        for f in state_factors:
            q_f = qs_dict_np[f]
            entropy = -np.sum(q_f * np.log(q_f + 1e-16))
            if entropy > ENTROPY_THRESHOLD:
                dynamic_factors.add(f)
        
        # Find dynamic deps
        SKIP_MODALITIES = {'button_just_pressed'}
        all_deps = set()
        for modality, deps in observation_state_dependencies.items():
            if modality not in SKIP_MODALITIES:
                for dep in deps:
                    if dep in dynamic_factors:
                        all_deps.add(dep)
        
        # Enumerate states ONCE, cache A_fn results
        dep_list = sorted(all_deps)
        dep_ranges = [range(len(qs_dict_np[dep])) for dep in dep_list]
        
        likelihood_cache = []
        prob_cache = []
        
        for combo in itertools.product(*dep_ranges):
            joint_prob = 1.0
            state_indices = map_indices.copy()
            for dep, idx in zip(dep_list, combo):
                joint_prob *= qs_dict_np[dep][idx]
                state_indices[dep] = int(idx)
            
            if joint_prob <= 1e-16:
                continue
            
            # Call A_fn ONCE per state
            obs_likelihoods = A_fn(state_indices)
            likelihood_cache.append(obs_likelihoods)
            prob_cache.append(joint_prob)
        
        # --- EXPECTED OBSERVATIONS ---
        qo_t = {}
        for modality, deps in observation_state_dependencies.items():
            if modality in SKIP_MODALITIES:
                continue
            
            num_obs = len(observation_labels[modality])
            qo_m = np.zeros(num_obs)
            
            for obs_lik, joint_prob in zip(likelihood_cache, prob_cache):
                qo_m += joint_prob * obs_lik[modality]
            
            qo_t[modality] = maths.normalize(qo_m)
        
        # Approximate button_just_pressed (works with both SA and MA naming)
        if 'button_just_pressed' in observation_state_dependencies:
            # Try both naming conventions
            if 'on_red_button' in qo_t:
                p_on_red = qo_t['on_red_button'][1]
                p_on_blue = qo_t['on_blue_button'][1]
            elif 'my_on_red_button' in qo_t:
                p_on_red = qo_t['my_on_red_button'][1]
                p_on_blue = qo_t['my_on_blue_button'][1]
            else:
                p_on_red = 0.0
                p_on_blue = 0.0
            p_just_pressed = min(1.0, p_on_red + p_on_blue)
            qo_t['button_just_pressed'] = np.array([1.0 - p_just_pressed, p_just_pressed])
        
        qo_pi.append(qo_t)
        
        # --- INFO GAIN (using JOINT observations, not sum of modalities) ---
        # Compute I(s; o_joint) = H[Q(o_joint)] - E_Q(s)[H[p(o_joint|s)]]
        
        # Get modalities to include (skip button_just_pressed)
        active_modalities = [m for m in observation_state_dependencies.keys() 
                            if m not in SKIP_MODALITIES]
        
        if len(active_modalities) == 0:
            timestep_info_gain = 0.0
        else:
            # Build joint observation space
            obs_sizes = [len(observation_labels[m]) for m in active_modalities]
            total_joint_obs = np.prod(obs_sizes)
            
            # Initialize joint distributions
            qo_joint = np.zeros(total_joint_obs)  # P(o_joint) under beliefs
            cond_entropy_joint = 0.0  # E_Q(s)[H[p(o_joint|s)]]
            
            # For each plausible state, compute joint observation distribution
            for obs_lik, joint_prob in zip(likelihood_cache, prob_cache):
                # Create joint observation distribution for this state
                # po_joint = p(o1|s) ⊗ p(o2|s) ⊗ ... (outer product)
                po_joint = np.array([1.0])
                for m in active_modalities:
                    p_o_m = obs_lik[m]
                    po_joint = np.outer(po_joint, p_o_m).ravel()
                
                # Accumulate predicted joint observation distribution
                qo_joint += joint_prob * po_joint
                
                # Accumulate conditional entropy: H[p(o_joint|s)]
                H_o_given_s = -np.sum(po_joint * maths.log_stable(po_joint))
                cond_entropy_joint += joint_prob * H_o_given_s
            
            # Normalize joint observation distribution
            qo_joint = maths.normalize(qo_joint)
            
            # Compute predicted entropy: H[Q(o_joint)]
            pred_entropy_joint = -np.sum(qo_joint * maths.log_stable(qo_joint))
            
            # Info gain = H[Q(o_joint)] - E_Q(s)[H[p(o_joint|s)]]
            timestep_info_gain = pred_entropy_joint - cond_entropy_joint
        
        total_info_gain += timestep_info_gain
        
        if debug and t_idx < 3:  # Show first 3 timesteps
            if len(active_modalities) > 0:
                print(f"      t={t_idx}: pred_H={pred_entropy_joint:.4f}, cond_H={cond_entropy_joint:.4f}, IG={timestep_info_gain:.4f}")
            else:
                print(f"      t={t_idx}: IG={timestep_info_gain:.4f} (no active modalities)")
    
    return qo_pi, float(total_info_gain)


# =============================================================================
# Expected Free Energy Components
# =============================================================================

def calc_expected_utility(qo_pi, C_fn, observation_labels):
    """
    Calculate expected utility (preference satisfaction) over time.
    
    U = Σ_t Σ_m Σ_o q(o_m^t) * C_m(o)
    
    Args:
        qo_pi: list of observation prediction dicts over time
        C_fn: functional preference model (obs_indices) → preferences
        observation_labels: dict mapping modality names to label lists
    
    Returns:
        expected_utility: float, sum of expected preferences
    
    Notes:
        Higher utility = observations align better with preferences
    """
    total_utility = 0.0
    
    for qo_t in qo_pi:
        # For each timestep, compute expected utility
        for modality, qo_m in qo_t.items():
            # Get number of observations for this modality
            num_obs = len(observation_labels[modality])
            
            # Sum over observations: Σ_o q(o) * C(o)
            for obs_idx in range(num_obs):
                # Get preference for this observation
                obs_indices = {modality: obs_idx}
                prefs = C_fn(obs_indices)
                pref_value = prefs.get(modality, 0.0)
                
                # Weight by probability of observing it
                total_utility += qo_m[obs_idx] * pref_value
    
    return float(total_utility)


def calc_states_info_gain(A_fn, qs_pi, state_factors, state_sizes):
    """
    Sum Bayesian surprise over time using full marginalization.
    """
    total_info_gain = 0.0
    for qs_t in qs_pi:
        G_t = maths.calc_surprise_functional(A_fn, qs_t, state_factors, state_sizes)
        total_info_gain += G_t
    return float(total_info_gain)


# =============================================================================
# Policy Posterior Inference
# =============================================================================

def vanilla_fpi_update_posterior_policies(
    qs,
    A_fn,
    B_fn,
    C_fn,
    policies,
    env_params,
    state_factors,
    state_sizes,
    observation_labels,
    observation_state_dependencies=None,
    use_utility=True,
    use_states_info_gain=True,
    E=None,
    gamma=16.0,
):
    """
    Update posterior over policies by computing Expected Free Energy (EFE).
    
    For each policy π:
        G(π) = -E_π[U] - E_π[G_states]
             = -(expected utility) - (expected information gain)
    
    Then: q(π) ∝ exp(-γ * G(π)) * p(π)
    
    Args:
        qs: dict of current state beliefs
        A_fn: functional observation model
        B_fn: functional transition model
        C_fn: functional preference model
        policies: list of policies (each policy is list of actions)
        env_params: dict with environment parameters
        state_factors: list of state factor names
        state_sizes: dict mapping factor names to sizes
        observation_labels: dict mapping modality names to observation labels
        use_utility: whether to include utility term
        use_states_info_gain: whether to include info gain term
        E: prior over policies (if None, uniform)
        gamma: precision parameter (inverse temperature)
    
    Returns:
        q_pi: array of policy posterior probabilities
        G: array of expected free energies per policy
    
    Notes:
        - Lower G = better policy (more utility and/or more info gain)
        - Gamma controls how deterministic policy selection is
    """
    num_policies = len(policies)
    G = np.zeros(num_policies)

    # Prior over policies (log space)
    if E is None:
        lnE = np.log(np.ones(num_policies) / num_policies)
    else:
        lnE = maths.log_stable(E)

    # Evaluate each policy (store details for later debug)
    policy_details = []  # Store (policy_idx, policy, qs_pi, utility, info_gain) for debugging
    
    for policy_idx, policy in enumerate(policies):
        # Predict future states under this policy
        qs_pi = get_expected_states(B_fn, qs, policy, env_params)
        
        # UNIFIED: Compute expected obs AND info gain in one pass (30-40% speedup)
        if use_utility and use_states_info_gain:
            qo_pi, info_gain = get_expected_obs_and_info_gain_unified(
                A_fn, qs_pi, state_factors, state_sizes, observation_labels, 
                observation_state_dependencies=observation_state_dependencies, debug=False
            )
            utility = calc_expected_utility(qo_pi, C_fn, observation_labels)
            G[policy_idx] -= utility
            G[policy_idx] -= info_gain
            policy_details.append((policy_idx, policy, qs_pi, utility, info_gain))
        elif use_utility:
            # Only utility needed
            qo_pi = get_expected_obs_sequence(A_fn, qs_pi, state_factors, state_sizes)
            utility = calc_expected_utility(qo_pi, C_fn, observation_labels)
            G[policy_idx] -= utility
            info_gain = 0.0
            policy_details.append((policy_idx, policy, qs_pi, utility, info_gain))
        elif use_states_info_gain:
            # Only info gain needed (rare case)
            _, info_gain = get_expected_obs_and_info_gain_unified(
                A_fn, qs_pi, state_factors, state_sizes, observation_labels,
                observation_state_dependencies=observation_state_dependencies
            )
            G[policy_idx] -= info_gain
            utility = 0.0
            policy_details.append((policy_idx, policy, qs_pi, utility, info_gain))
    
    # Compute policy posterior: q(π) ∝ exp(-γ * G(π)) * p(π)
    log_q_pi = -gamma * G + lnE
    q_pi = maths.softmax(log_q_pi)
    
    # DEBUG: Show top 5 most probable policies with detailed breakdown
    # Commented out for cleaner output
    # top_k = 5
    # top_indices = np.argsort(q_pi)[-top_k:][::-1]  # Highest probability first
    # 
    # print(f"\n🔍 Top {top_k} Most Probable Policies:")
    # action_names = {0: 'UP', 1: 'DO', 2: 'LE', 3: 'RI', 4: 'OP', 5: 'NO'}
    # 
    # for rank, idx in enumerate(top_indices, 1):
    #     policy_idx, policy, qs_pi, utility, info_gain = policy_details[idx]
    #     policy_str = '→'.join([action_names[a] for a in policy])
    #     print(f"  #{rank} Policy {policy_idx:2d} [{policy_str:11s}]: prob={q_pi[idx]:.4f}, util={utility:7.4f}, info={info_gain:7.4f}, G={G[idx]:7.4f}")
    #     
    #     # Show predicted agent positions (entropy of belief)
    #     if rank <= 5 and len(qs_pi) > 0:
    #         print(f"      Predicted agent position entropy at each step:")
    #         for t_idx, qs_t in enumerate(qs_pi):
    #             agent_pos_belief = np.array(qs_t['agent_pos'])
    #             entropy = -np.sum(agent_pos_belief * np.log(agent_pos_belief + 1e-16))
    #             most_likely_pos = int(np.argmax(agent_pos_belief))
    #             max_prob = agent_pos_belief[most_likely_pos]
    #             print(f"        t={t_idx}: pos*={most_likely_pos}, prob={max_prob:.3f}, H={entropy:.4f}")
    #     
    #     # Show per-timestep breakdown for top 2
    #     if rank <= 2:
    #         # Re-compute with debug enabled for detailed view
    #         qo_pi_debug, info_gain_debug = get_expected_obs_and_info_gain_unified(
    #             A_fn, qs_pi, state_factors, state_sizes, observation_labels, debug=True
    #         )
    #         
    #         # Show what observations contribute to utility
    #         print(f"      Utility breakdown per timestep:")
    #         for t, qo_t in enumerate(qo_pi_debug):
    #             # Compute utility for this timestep
    #             timestep_util = 0.0
    #             for modality, qo_m in qo_t.items():
    #                 for obs_idx in range(len(qo_m)):
    #                     obs_indices = {modality: obs_idx}
    #                     prefs = C_fn(obs_indices)
    #                     pref_value = prefs.get(modality, 0.0)
    #                     contribution = qo_m[obs_idx] * pref_value
    #                     if abs(contribution) > 0.01:  # Only show significant contributions
    #                         obs_labels = observation_labels.get(modality, [])
    #                         obs_name = obs_labels[obs_idx] if obs_idx < len(obs_labels) else str(obs_idx)
    #                         timestep_util += contribution
    #                         print(f"        t={t} {modality}[{obs_name}]: p={qo_m[obs_idx]:.3f} × pref={pref_value:.2f} = {contribution:.4f}")
    #         print(f"        t={t} TOTAL: {timestep_util:.4f}")
    
    return q_pi, G


# =============================================================================
# Action Selection
# =============================================================================

def sample_action(q_pi, policies, action_selection="deterministic", alpha=16.0, actions=None):
    """
    Sample action from policy posterior (marginal over first actions).
    
    Wrapper around utils.sample_action for consistency.
    
    Args:
        q_pi: policy posterior
        policies: list of policies
        action_selection: "deterministic" or "stochastic"
        alpha: precision parameter
        actions: list of available actions
    
    Returns:
        selected action (int)
    """
    return utils.sample_action(q_pi, policies, action_selection, alpha, actions)


def sample_policy(q_pi, policies, action_selection="deterministic", alpha=16.0):
    """
    Sample policy from policy posterior and return first action.
    
    Wrapper around utils.sample_policy for consistency.
    
    Args:
        q_pi: policy posterior
        policies: list of policies
        action_selection: "deterministic" or "stochastic"
        alpha: precision parameter
    
    Returns:
        selected action (int)
    """
    return utils.sample_policy(q_pi, policies, action_selection, alpha)


# =============================================================================
# Debugging/Analysis Utilities
# =============================================================================

def evaluate_policy_components(
    policy,
    qs,
    A_fn,
    B_fn,
    C_fn,
    env_params,
    state_factors,
    state_sizes,
    observation_labels,
):
    """
    Evaluate individual components of a single policy's EFE.
    
    Useful for debugging and understanding agent behavior.
    
    Args:
        policy: single policy (list of actions)
        qs: current beliefs
        A_fn, B_fn, C_fn: generative model functions
        env_params: environment parameters
        state_factors: list of factor names
        state_sizes: dict of factor sizes
        observation_labels: dict of observation labels
    
    Returns:
        components: dict with 'utility', 'info_gain', 'G_total'
    """
    # Predict states
    qs_pi = get_expected_states(B_fn, qs, policy, env_params)
    
    # Predict observations
    qo_pi = get_expected_obs_sequence(A_fn, qs_pi, state_factors, state_sizes)
    
    # Compute components
    utility = calc_expected_utility(qo_pi, C_fn, observation_labels)
    info_gain = calc_states_info_gain(A_fn, qs_pi, state_factors, state_sizes)
    
    G_total = -utility - info_gain
    
    return {
        'utility': float(utility),
        'info_gain': float(info_gain),
        'G_total': float(G_total),
        'predicted_states': qs_pi,
        'predicted_observations': qo_pi,
    }


def get_top_policies(q_pi, policies, top_k=5):
    """
    Get top-k most likely policies.
    
    Args:
        q_pi: policy posterior
        policies: list of policies
        top_k: number of top policies to return
    
    Returns:
        top_policies: list of (policy, probability, index) tuples
    """
    top_indices = np.argsort(q_pi)[-top_k:][::-1]
    
    top_policies = []
    for idx in top_indices:
        policy = policies[idx]
        prob = q_pi[idx]
        top_policies.append((policy, float(prob), int(idx)))
    
    return top_policies
