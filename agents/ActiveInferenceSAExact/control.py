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
# Exact per-group decomposition (replaces the adaptive entropy threshold)
# =============================================================================
#
# Ported verbatim from agents/ActiveInferenceRedBlueButtonExact/control.py
# (built and multiply-verified for MA Red-Blue-Button, 2026-08-13). The
# original get_expected_obs_from_beliefs / get_expected_obs_and_info_gain_unified
# picked which state factors to enumerate via an adaptive entropy threshold: any
# factor whose belief entropy fell below the threshold was dropped entirely from
# enumeration. When ALL factors are simultaneously below threshold (e.g. once
# position and button states are all fairly confident), dep_list becomes empty,
# itertools.product() over zero ranges yields exactly one (MAP) combo, and the
# predicted/conditional joint-observation entropy become identical by
# construction -- info gain is then IDENTICALLY zero regardless of how much real
# residual uncertainty remains (see ai/02-debug.md, "Deeper issue: flat EFE").
#
# Fix: no threshold at all. Group state factors into connected components via
# the modality<->factor dependency graph (union-find) -- two factors land in the
# same group iff some active modality depends on both. Within a group, states
# are enumerated exactly (every combo, no threshold-based dropping). Because the
# mean-field belief factorizes q(s) = prod_f q(s_f), and no modality spans two
# different groups, the joint predictive entropy is the exact SUM of each
# group's own joint entropy -- provably exact, not an approximation, for any
# dependency structure.
#
# This is the SAME generic decomposition used by MA Red-Blue-Button's Exact
# package; it is not RedBlueButton-Exact-specific in structure, so it applies
# directly to this package's (single-agent) state_factors/observation_state_dependencies.

def _group_modalities_by_shared_factors(state_factors, observation_state_dependencies, skip_modalities):
    """
    Union-find over the modality<->factor dependency graph. Returns a list of
    (group_factors: sorted list[str], group_modalities: list[str]) tuples,
    covering every active (non-skipped) modality exactly once and every state
    factor exactly once (a factor with no active-modality dependency at all
    gets its own singleton group with an empty modality list).
    """
    parent = {f: f for f in state_factors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    active_modalities = [m for m in observation_state_dependencies.keys() if m not in skip_modalities]
    for m in active_modalities:
        deps = [d for d in observation_state_dependencies[m] if d in parent]
        for d in deps[1:]:
            union(deps[0], d)

    groups_factors = {}
    for f in state_factors:
        groups_factors.setdefault(find(f), set()).add(f)

    groups_modalities = {root: [] for root in groups_factors}
    for m in active_modalities:
        deps = [d for d in observation_state_dependencies[m] if d in parent]
        if not deps:
            continue
        root = find(deps[0])
        groups_modalities[root].append(m)

    return [
        (sorted(groups_factors[root]), groups_modalities[root])
        for root in groups_factors
    ]


def get_expected_obs_from_beliefs(A_fn, qs_dict, state_factors, state_sizes,
                                   observation_labels=None, observation_state_dependencies=None):
    """
    Compute expected observation distributions from belief over states.

    Exact per-group decomposition (see module docstring above) -- no adaptive
    entropy threshold, no factor ever dropped from enumeration.
    """
    if observation_labels is None or observation_state_dependencies is None:
        from generative_models.SA_ActiveInference.RedBlueButtonExact import model_init as default_model
        if observation_labels is None:
            observation_labels = default_model.observations
        if observation_state_dependencies is None:
            observation_state_dependencies = default_model.observation_state_dependencies
    import itertools

    qs_dict_np = {f: np.array(qs_dict[f]) for f in state_factors}
    map_indices = {f: int(np.argmax(qs_dict_np[f])) for f in state_factors}

    SKIP_MODALITIES = {'button_just_pressed'}
    if observation_state_dependencies is not None:
        SKIP_MODALITIES |= {m for m in observation_state_dependencies.keys() if m.startswith("counter_") or m.startswith("ctr_")}
    SKIP_MODALITIES.add("agent_held_obs")

    groups = _group_modalities_by_shared_factors(state_factors, observation_state_dependencies, SKIP_MODALITIES)

    qo_dict = {}
    for group_factors, group_modalities in groups:
        if not group_modalities:
            continue
        dep_ranges = [range(state_sizes[f]) for f in group_factors]

        qo_ms = {m: np.zeros(len(observation_labels[m])) for m in group_modalities}
        for combo in itertools.product(*dep_ranges):
            joint_prob = 1.0
            state_indices = map_indices.copy()
            for f, idx in zip(group_factors, combo):
                joint_prob *= qs_dict_np[f][idx]
                state_indices[f] = int(idx)
            if joint_prob <= 1e-16:
                continue
            obs_likelihoods = A_fn(state_indices)
            for m in group_modalities:
                qo_ms[m] += joint_prob * obs_likelihoods[m]

        for m in group_modalities:
            qo_dict[m] = maths.normalize(qo_ms[m])

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


def get_expected_obs_sequence(
    A_fn,
    qs_pi,
    state_factors,
    state_sizes,
    observation_labels=None,
    observation_state_dependencies=None,
):
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
        qo_t = get_expected_obs_from_beliefs(
            A_fn,
            qs_t,
            state_factors,
            state_sizes,
            observation_labels=observation_labels,
            observation_state_dependencies=observation_state_dependencies,
        )
        qo_pi.append(qo_t)

    return qo_pi


def get_expected_obs_and_info_gain_unified(A_fn, qs_pi, state_factors, state_sizes, observation_labels,
                                            observation_state_dependencies=None, debug=False):
    """
    UNIFIED: Compute BOTH expected observations AND info gain in ONE pass.

    Exact per-group decomposition (see module docstring above). Total info
    gain per timestep is the exact sum of each connected group's own joint
    info gain -- provably exact under mean-field factorization + conditionally
    independent-given-state modalities (both already assumed throughout this
    codebase), not an approximation, and never drops a factor from
    enumeration regardless of how confident its belief is.

    Returns:
        qo_pi: list of observation predictions per timestep
        total_info_gain: float, sum of Bayesian surprise over timesteps
    """
    if observation_state_dependencies is None:
        from generative_models.SA_ActiveInference.RedBlueButtonExact import model_init as default_model
        observation_state_dependencies = default_model.observation_state_dependencies
    import itertools

    SKIP_MODALITIES = {'button_just_pressed'}
    if observation_state_dependencies is not None:
        SKIP_MODALITIES |= {m for m in observation_state_dependencies.keys() if m.startswith("counter_") or m.startswith("ctr_")}
    SKIP_MODALITIES.add("agent_held_obs")

    qo_pi = []
    total_info_gain = 0.0

    for t_idx, qs_t in enumerate(qs_pi):
        qs_dict_np = {f: np.array(qs_t[f]) for f in state_factors}
        map_indices = {f: int(np.argmax(qs_dict_np[f])) for f in state_factors}

        groups = _group_modalities_by_shared_factors(state_factors, observation_state_dependencies, SKIP_MODALITIES)

        qo_t = {}
        timestep_info_gain = 0.0

        for group_factors, group_modalities in groups:
            if not group_modalities:
                continue
            dep_ranges = [range(state_sizes[f]) for f in group_factors]

            qo_ms = {m: np.zeros(len(observation_labels[m])) for m in group_modalities}
            obs_sizes = [len(observation_labels[m]) for m in group_modalities]
            total_joint_obs = int(np.prod(obs_sizes))
            qo_joint_group = np.zeros(total_joint_obs)
            cond_entropy_group = 0.0

            for combo in itertools.product(*dep_ranges):
                joint_prob = 1.0
                state_indices = map_indices.copy()
                for f, idx in zip(group_factors, combo):
                    joint_prob *= qs_dict_np[f][idx]
                    state_indices[f] = int(idx)
                if joint_prob <= 1e-16:
                    continue

                obs_likelihoods = A_fn(state_indices)

                po_joint = np.array([1.0])
                for m in group_modalities:
                    p_o_m = obs_likelihoods[m]
                    qo_ms[m] += joint_prob * p_o_m
                    po_joint = np.outer(po_joint, p_o_m).ravel()

                qo_joint_group += joint_prob * po_joint
                H_o_given_s = -np.sum(po_joint * maths.log_stable(po_joint))
                cond_entropy_group += joint_prob * H_o_given_s

            for m in group_modalities:
                qo_t[m] = maths.normalize(qo_ms[m])

            qo_joint_group = maths.normalize(qo_joint_group)
            pred_entropy_group = -np.sum(qo_joint_group * maths.log_stable(qo_joint_group))
            timestep_info_gain += pred_entropy_group - cond_entropy_group

        # Approximate button_just_pressed (works with both SA and MA naming)
        if 'button_just_pressed' in observation_state_dependencies:
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
        total_info_gain += timestep_info_gain

        if debug and t_idx < 3:
            print(f"      t={t_idx}: IG={timestep_info_gain:.4f} ({len(groups)} groups)")

    return qo_pi, float(total_info_gain)


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
    return_policy_details=False,
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
        return_policy_details: if True, return list of dicts with utility, info_gain per policy
    
    Returns:
        q_pi: array of policy posterior probabilities
        G: array of expected free energies per policy
        policy_details: if return_policy_details, list of dicts with keys
            policy_idx, policy, utility, info_gain, G, prob; else None
    
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

    # Evaluate each policy (store details for later debug / printing)
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
            qo_pi = get_expected_obs_sequence(
                A_fn,
                qs_pi,
                state_factors,
                state_sizes,
                observation_labels=observation_labels,
                observation_state_dependencies=observation_state_dependencies,
            )
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
    
    # Optionally build compact details for caller (utility, info_gain, G, prob per policy)
    out_details = None
    if return_policy_details:
        out_details = [
            {
                "policy_idx": i,
                "policy": list(pol),
                "utility": float(u),
                "info_gain": float(ig),
                "G": float(G[i]),
                "prob": float(q_pi[i]),
            }
            for (i, pol, _qs_pi, u, ig) in policy_details
        ]
    
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
    
    if return_policy_details:
        return q_pi, G, out_details
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
    qo_pi = get_expected_obs_sequence(
        A_fn,
        qs_pi,
        state_factors,
        state_sizes,
        observation_labels=observation_labels,
        observation_state_dependencies=observation_state_dependencies,
    )
    
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
