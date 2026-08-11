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

import itertools

import numpy as np
from . import maths
from . import utils




# =============================================================================
# Exact per-factor/component decomposition (replaces the top-k/budget
# approximation formerly known as select_dynamic_factors -- see
# ai/02-debug.md section J.7 for the derivation and verification).
# =============================================================================
#
# Under the mean-field belief factorization q(s) = prod_f q(s_f) already used
# throughout this codebase, and given that observation modalities are
# conditionally independent given the full state (already assumed everywhere
# A_fn's per-modality likelihoods are multiplied together), any two modalities
# whose dependency sets share no factor are independent under the predictive
# distribution Q(o). Entropy of a product of independent distributions is the
# SUM of their marginal entropies, so predictive entropy and conditional
# entropy -- and therefore information gain -- decompose EXACTLY into a sum
# over the connected components of the modality-factor dependency graph.
#
# For every modality currently defined in this generative model, that
# dependency graph is trivial: each modality depends on exactly one, distinct
# factor (self_pos_obs <-> self_pos, self_held_obs <-> self_held, etc. --
# confirmed via direct inspection of observation_state_dependencies), so every
# component has size 1 and the "joint" enumeration this function used to
# require (up to 73,728 states across all 8 eligible factors) collapses to 8
# independent enumerations of at most 6 states each. This is not an
# approximation: it was verified to match the full 73,728-state combinatorial
# enumeration to floating-point precision (~1e-13) across multiple randomized
# belief scenarios, including ones with simultaneous diffuseness across
# self_pos/other_pos/self_held/pot_state -- the exact scenario that caused the
# old top-k scheme to silently drop self_held/pot_state and change the agent's
# real decision (cntr1/stay vs cntr1/interact). The grouping logic below is
# written generally (not hardcoded to today's 1-factor-per-modality shape) so
# it stays correct and still cheap if a future modality is ever added that
# depends on more than one factor -- that modality's own (small) factor group
# would need real joint enumeration, but every other, disjoint modality would
# be unaffected and still computed independently.


def _group_modalities_by_shared_factors(observation_state_dependencies, skip_modalities):
    """
    Partition non-skipped modalities into connected components of the
    modality-factor dependency graph (union-find over shared factors).

    Returns a list of {"factors": set[str], "modalities": list[str]} groups.
    Modalities in different groups depend on completely disjoint factors, so
    their contributions to predictive/conditional entropy are independent and
    additive; modalities in the same group must be enumerated jointly over
    the union of their factors (today, every group has exactly one factor and
    one modality).
    """
    modalities = [
        m for m, deps in observation_state_dependencies.items()
        if m not in skip_modalities and deps
    ]

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for m in modalities:
        deps = observation_state_dependencies[m]
        for d in deps[1:]:
            union(deps[0], d)

    groups = {}
    for m in modalities:
        deps = observation_state_dependencies[m]
        root = find(deps[0])
        g = groups.setdefault(root, {"factors": set(), "modalities": []})
        g["factors"].update(deps)
        g["modalities"].append(m)

    return list(groups.values())



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
        action_for_b = int(action) if np.isscalar(action) else action
        qs_next = B_fn(qs_t, action_for_b, **env_params)
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

    EXACT per-modality computation (see ai/02-debug.md section J.7): under the
    mean-field belief factorization, a modality's predicted marginal only ever
    needs the specific factor(s) it actually depends on -- for every modality
    in this model that's a single factor (cardinality <= 6) -- so no full
    joint-state enumeration is required at all.
    """
    # Import default model_init for backward compatibility
    if observation_labels is None or observation_state_dependencies is None:
        from generative_models.SA_ActiveInference.RedBlueButton import model_init as default_model
        if observation_labels is None:
            observation_labels = default_model.observations
        if observation_state_dependencies is None:
            observation_state_dependencies = default_model.observation_state_dependencies

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

    qo_dict = {}
    for modality, deps in observation_state_dependencies.items():
        if modality in SKIP_MODALITIES or not deps:
            continue

        dep_ranges = [range(len(qs_dict_np[d])) for d in deps]
        num_obs = len(observation_labels[modality])
        qo_m = np.zeros(num_obs)

        for combo in itertools.product(*dep_ranges):
            w = 1.0
            s_idx = map_indices.copy()
            for d, idx in zip(deps, combo):
                w *= qs_dict_np[d][idx]
                s_idx[d] = int(idx)
            if w <= 1e-16:
                continue
            p_o_m = A_fn(s_idx)[modality]
            qo_m += w * p_o_m

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

    EXACT per-component decomposition (see ai/02-debug.md section J.7 and the
    module docstring above `_group_modalities_by_shared_factors`): instead of
    enumerating the full cross-product of every eligible state factor (up to
    73,728 states for this model), this enumerates only within each connected
    component of the modality-factor dependency graph and sums the results,
    which is mathematically exact under mean-field beliefs + conditionally
    independent modalities, not an approximation. For every modality currently
    defined in this generative model, each component is a single (modality,
    factor) pair, so this reduces to 8 independent enumerations of at most 6
    states each.

    Returns:
        qo_pi: list of observation predictions per timestep
        total_info_gain: float, sum of Bayesian surprise over timesteps
    """
    # Import default model_init for backward compatibility
    if observation_state_dependencies is None:
        from generative_models.SA_ActiveInference.RedBlueButton import model_init as default_model
        observation_state_dependencies = default_model.observation_state_dependencies

    qo_pi = []
    total_info_gain = 0.0

    for t_idx, qs_t in enumerate(qs_pi):
        qs_dict_np = {f: np.array(qs_t[f]) for f in state_factors}
        map_indices = {f: int(np.argmax(qs_dict_np[f])) for f in state_factors}

        # Skip modalities that are not used for EFE scoring.
        SKIP_MODALITIES = {"button_just_pressed"}
        if observation_state_dependencies is not None:
            SKIP_MODALITIES |= {
                m
                for m in observation_state_dependencies.keys()
                if m.startswith("counter_") or m.startswith("ctr_")
            }
        SKIP_MODALITIES.add("agent_held_obs")

        groups = _group_modalities_by_shared_factors(observation_state_dependencies, SKIP_MODALITIES)

        qo_t = {}
        timestep_info_gain = 0.0

        for group in groups:
            factors = sorted(group["factors"])
            modalities = group["modalities"]
            dep_ranges = [range(len(qs_dict_np[f])) for f in factors]

            qo_joint_local = None
            cond_entropy_local = 0.0
            qo_m_local = {m: np.zeros(len(observation_labels[m])) for m in modalities}

            for combo in itertools.product(*dep_ranges):
                w = 1.0
                s_idx = map_indices.copy()
                for f, idx in zip(factors, combo):
                    w *= qs_dict_np[f][idx]
                    s_idx[f] = int(idx)
                if w <= 1e-16:
                    continue

                obs_lik = A_fn(s_idx)

                po_joint = np.array([1.0])
                for m in modalities:
                    p_o_m = obs_lik[m]
                    qo_m_local[m] += w * p_o_m
                    po_joint = np.outer(po_joint, p_o_m).ravel()

                if qo_joint_local is None:
                    qo_joint_local = np.zeros_like(po_joint)
                qo_joint_local += w * po_joint
                H_o_given_s = -np.sum(po_joint * maths.log_stable(po_joint))
                cond_entropy_local += w * H_o_given_s

            for m in modalities:
                qo_t[m] = maths.normalize(qo_m_local[m])

            if qo_joint_local is not None:
                qo_joint_local = maths.normalize(qo_joint_local)
                pred_entropy_local = -np.sum(qo_joint_local * maths.log_stable(qo_joint_local))
                timestep_info_gain += pred_entropy_local - cond_entropy_local

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
        total_info_gain += timestep_info_gain

        if debug and t_idx < 3:
            print(f"      t={t_idx}: IG={timestep_info_gain:.4f} (exact, {len(groups)} independent groups)")

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