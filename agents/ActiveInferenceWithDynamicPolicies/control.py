"""
Policy evaluation and action selection for Active Inference with functional generative model.

This module implements:
- Expected state computation using B_fn
- Expected observation prediction using A_fn
- Expected Free Energy (EFE) calculation
- Policy posterior inference
- Action selection

Notes:
- Policies are sequences of primitive actions.
- The candidate policy set can be regenerated at each timestep.
- This module evaluates whatever policy list it is given; it does not assume
  a fixed global policy library.
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
        B_fn: functional transition model (qs, action) -> next_qs
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
    return B_fn(qs_current, action, **env_params)


def get_expected_states(B_fn, qs_current, policy, env_params):
    """
    Roll out expected states under a sequence of actions (policy).

    Args:
        B_fn: functional transition model
        qs_current: dict of current beliefs
        policy: list/array of primitive actions [a_0, a_1, ..., a_T]
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

def get_expected_obs_from_beliefs(
    A_fn,
    qs_dict,
    state_factors,
    state_sizes,
    observation_labels=None,
    observation_state_dependencies=None,
):
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

    # Convert arrays to numpy to avoid compilation overhead
    qs_dict_np = {f: np.array(qs_dict[f]) for f in state_factors}

    map_indices = {f: int(np.argmax(qs_dict_np[f])) for f in state_factors}

    SKIP_MODALITIES = {"button_just_pressed"}
    # Performance: skip counter modalities to avoid combinatorial blow-up.
    # Overcooked uses ctr_<grid>_obs; RedBlueButton uses counter_*.
    if observation_state_dependencies is not None:
        SKIP_MODALITIES |= {
            m
            for m in observation_state_dependencies.keys()
            if m.startswith("counter_") or m.startswith("ctr_")
        }

    # Prevent epistemic loops where agents toggle held items to generate predictable observations.
    # (We still use held observations for state inference, just not for EFE scoring computations here.)
    SKIP_MODALITIES.add("agent_held_obs")

    # Adaptive entropy threshold based on belief concentration
    max_entropy_observed = max(
        -np.sum(qs_dict_np[f] * np.log(qs_dict_np[f] + 1e-16))
        for f in state_factors
    )
    ENTROPY_THRESHOLD = min(0.1, max(0.01, max_entropy_observed * 0.1))

    dynamic_factors = set()
    for f in state_factors:
        q_f = qs_dict_np[f]
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

    # Enumerate combinations of dynamic factors only
    dep_list = sorted(all_deps)
    dep_ranges = [range(len(qs_dict_np[dep])) for dep in dep_list]

    # Precompute likelihoods for all state combinations
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

        obs_likelihoods = A_fn(state_indices)
        likelihood_cache.append(obs_likelihoods)
        prob_cache.append(joint_prob)

    # Marginalize each modality using cached likelihoods
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
    if "button_just_pressed" in observation_state_dependencies:
        if "on_red_button" in qo_dict:
            p_on_red = qo_dict["on_red_button"][1]
            p_on_blue = qo_dict["on_blue_button"][1]
        elif "my_on_red_button" in qo_dict:
            p_on_red = qo_dict["my_on_red_button"][1]
            p_on_blue = qo_dict["my_on_blue_button"][1]
        else:
            p_on_red = 0.0
            p_on_blue = 0.0
        p_just_pressed = min(1.0, p_on_red + p_on_blue)
        qo_dict["button_just_pressed"] = np.array([1.0 - p_just_pressed, p_just_pressed])

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


def get_expected_obs_and_info_gain_unified(
    A_fn,
    qs_pi,
    state_factors,
    state_sizes,
    observation_labels,
    observation_state_dependencies=None,
    debug=False,
):
    """
    Compute BOTH expected observations AND information gain in one pass.

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

        # Adaptive entropy threshold
        max_entropy_observed = max(
            -np.sum(qs_dict_np[f] * np.log(qs_dict_np[f] + 1e-16))
            for f in state_factors
        )
        ENTROPY_THRESHOLD = min(0.1, max(0.01, max_entropy_observed * 0.1))

        dynamic_factors = set()
        for f in state_factors:
            q_f = qs_dict_np[f]
            entropy = -np.sum(q_f * np.log(q_f + 1e-16))
            if entropy > ENTROPY_THRESHOLD:
                dynamic_factors.add(f)

        # Find dynamic deps
        SKIP_MODALITIES = {"button_just_pressed"}
        if observation_state_dependencies is not None:
            SKIP_MODALITIES |= {
                m
                for m in observation_state_dependencies.keys()
                if m.startswith("counter_") or m.startswith("ctr_")
            }
        SKIP_MODALITIES.add("agent_held_obs")

        all_deps = set()
        for modality, deps in observation_state_dependencies.items():
            if modality not in SKIP_MODALITIES:
                for dep in deps:
                    if dep in dynamic_factors:
                        all_deps.add(dep)

        # Enumerate states once, cache A_fn results
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

            obs_likelihoods = A_fn(state_indices)
            likelihood_cache.append(obs_likelihoods)
            prob_cache.append(joint_prob)

        # --- Expected observations ---
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
        if "button_just_pressed" in observation_state_dependencies:
            if "on_red_button" in qo_t:
                p_on_red = qo_t["on_red_button"][1]
                p_on_blue = qo_t["on_blue_button"][1]
            elif "my_on_red_button" in qo_t:
                p_on_red = qo_t["my_on_red_button"][1]
                p_on_blue = qo_t["my_on_blue_button"][1]
            else:
                p_on_red = 0.0
                p_on_blue = 0.0
            p_just_pressed = min(1.0, p_on_red + p_on_blue)
            qo_t["button_just_pressed"] = np.array([1.0 - p_just_pressed, p_just_pressed])

        qo_pi.append(qo_t)

        # --- Info gain using joint observations ---
        active_modalities = [
            m for m in observation_state_dependencies.keys()
            if m not in SKIP_MODALITIES
        ]

        if len(active_modalities) == 0:
            timestep_info_gain = 0.0
            pred_entropy_joint = 0.0
            cond_entropy_joint = 0.0
        else:
            obs_sizes = [len(observation_labels[m]) for m in active_modalities]
            total_joint_obs = int(np.prod(obs_sizes))

            qo_joint = np.zeros(total_joint_obs)   # P(o_joint) under beliefs
            cond_entropy_joint = 0.0              # E_Q(s)[H[p(o_joint|s)]]

            for obs_lik, joint_prob in zip(likelihood_cache, prob_cache):
                po_joint = np.array([1.0])
                for m in active_modalities:
                    p_o_m = obs_lik[m]
                    po_joint = np.outer(po_joint, p_o_m).ravel()

                qo_joint += joint_prob * po_joint

                H_o_given_s = -np.sum(po_joint * maths.log_stable(po_joint))
                cond_entropy_joint += joint_prob * H_o_given_s

            qo_joint = maths.normalize(qo_joint)
            pred_entropy_joint = -np.sum(qo_joint * maths.log_stable(qo_joint))
            timestep_info_gain = pred_entropy_joint - cond_entropy_joint

        total_info_gain += timestep_info_gain

        if debug and t_idx < 3:
            if len(active_modalities) > 0:
                print(
                    f"      t={t_idx}: pred_H={pred_entropy_joint:.4f}, "
                    f"cond_H={cond_entropy_joint:.4f}, IG={timestep_info_gain:.4f}"
                )
            else:
                print(f"      t={t_idx}: IG={timestep_info_gain:.4f} (no active modalities)")

    return qo_pi, float(total_info_gain)


# =============================================================================
# Expected Free Energy Components
# =============================================================================

def calc_expected_utility(qo_pi, C_fn, observation_labels):
    """
    Calculate expected utility (preference satisfaction) over time.

    U = sum_t sum_m sum_o q(o_m^t) * C_m(o)

    Args:
        qo_pi: list of observation prediction dicts over time
        C_fn: functional preference model (obs_indices) -> preferences
        observation_labels: dict mapping modality names to label lists

    Returns:
        expected_utility: float, sum of expected preferences

    Notes:
        Higher utility = observations align better with preferences
    """
    total_utility = 0.0

    for qo_t in qo_pi:
        for modality, qo_m in qo_t.items():
            num_obs = len(observation_labels[modality])

            for obs_idx in range(num_obs):
                obs_indices = {modality: obs_idx}
                prefs = C_fn(obs_indices)
                pref_value = prefs.get(modality, 0.0)
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

    For each policy pi:
        G(pi) = -E_pi[U] - E_pi[G_states]
              = -(expected utility) - (expected information gain)

    Then: q(pi) proportional to exp(-gamma * G(pi)) * p(pi)

    Args:
        qs: dict of current state beliefs
        A_fn: functional observation model
        B_fn: functional transition model
        C_fn: functional preference model
        policies: list of policies; each policy is a primitive action sequence
        env_params: dict with environment parameters
        state_factors: list of state factor names
        state_sizes: dict mapping factor names to sizes
        observation_labels: dict mapping modality names to observation labels
        observation_state_dependencies: optional modality dependency mapping
        use_utility: whether to include utility term
        use_states_info_gain: whether to include info gain term
        E: prior over policies (if None, uniform over provided policies)
        gamma: precision parameter (inverse temperature)
        return_policy_details: if True, return list of dicts with utility, info_gain per policy

    Returns:
        q_pi: array of policy posterior probabilities
        G: array of expected free energies per policy
        policy_details: if return_policy_details, list of dicts with keys
            policy_idx, policy, utility, info_gain, G, prob; else None

    Notes:
        - Lower G = better policy
        - Gamma controls how deterministic policy selection is
        - The supplied policy set can be different at each timestep
    """
    num_policies = len(policies)
    G = np.zeros(num_policies)

    # Prior over policies (log space)
    if E is None:
        lnE = np.log(np.ones(num_policies) / max(num_policies, 1))
    else:
        lnE = maths.log_stable(E)

    # Evaluate each policy
    policy_details = []

    for policy_idx, policy in enumerate(policies):
        qs_pi = get_expected_states(B_fn, qs, policy, env_params)

        if use_utility and use_states_info_gain:
            qo_pi, info_gain = get_expected_obs_and_info_gain_unified(
                A_fn,
                qs_pi,
                state_factors,
                state_sizes,
                observation_labels,
                observation_state_dependencies=observation_state_dependencies,
                debug=False,
            )
            utility = calc_expected_utility(qo_pi, C_fn, observation_labels)
            G[policy_idx] -= utility
            G[policy_idx] -= info_gain
            policy_details.append((policy_idx, policy, qs_pi, utility, info_gain))

        elif use_utility:
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
            _, info_gain = get_expected_obs_and_info_gain_unified(
                A_fn,
                qs_pi,
                state_factors,
                state_sizes,
                observation_labels,
                observation_state_dependencies=observation_state_dependencies,
            )
            G[policy_idx] -= info_gain
            utility = 0.0
            policy_details.append((policy_idx, policy, qs_pi, utility, info_gain))

        else:
            utility = 0.0
            info_gain = 0.0
            policy_details.append((policy_idx, policy, qs_pi, utility, info_gain))

    log_q_pi = -gamma * G + lnE
    q_pi = maths.softmax(log_q_pi)

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

    if return_policy_details:
        return q_pi, G, out_details
    return q_pi, G


# =============================================================================
# Action Selection
# =============================================================================

def sample_action(q_pi, policies, action_selection="deterministic", alpha=16.0, actions=None):
    """
    Sample an action from the policy posterior by marginalizing over first actions.

    Args:
        q_pi: policy posterior
        policies: list of candidate primitive-action policies
        action_selection: "deterministic" or "stochastic"
        alpha: precision parameter
        actions: list of available actions

    Returns:
        selected action (int)
    """
    return utils.sample_action(q_pi, policies, action_selection, alpha, actions)


def sample_policy(q_pi, policies, action_selection="deterministic", alpha=16.0):
    """
    Sample a policy from the policy posterior and return its first action.

    Args:
        q_pi: policy posterior
        policies: list of candidate primitive-action policies
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
    observation_state_dependencies=None,
):
    """
    Evaluate individual components of a single policy's EFE.

    Useful for debugging and understanding agent behavior.

    Args:
        policy: single policy (list of primitive actions)
        qs: current beliefs
        A_fn, B_fn, C_fn: generative model functions
        env_params: environment parameters
        state_factors: list of factor names
        state_sizes: dict of factor sizes
        observation_labels: dict of observation labels
        observation_state_dependencies: optional modality dependency mapping

    Returns:
        components: dict with 'utility', 'info_gain', 'G_total'
    """
    qs_pi = get_expected_states(B_fn, qs, policy, env_params)

    qo_pi = get_expected_obs_sequence(
        A_fn,
        qs_pi,
        state_factors,
        state_sizes,
        observation_labels=observation_labels,
        observation_state_dependencies=observation_state_dependencies,
    )

    utility = calc_expected_utility(qo_pi, C_fn, observation_labels)
    info_gain = calc_states_info_gain(A_fn, qs_pi, state_factors, state_sizes)

    G_total = -utility - info_gain

    return {
        "utility": float(utility),
        "info_gain": float(info_gain),
        "G_total": float(G_total),
        "predicted_states": qs_pi,
        "predicted_observations": qo_pi,
    }


def get_top_policies(q_pi, policies, top_k=5):
    """
    Get top-k most likely policies.

    Args:
        q_pi: policy posterior
        policies: list of candidate policies
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