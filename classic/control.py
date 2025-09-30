import numpy as np
from .maths import log_stable, softmax, calc_surprise
from .maths import spm_dot_optimized, spm_dot_vectorized
from scipy.special import logsumexp  # if you can depend on SciPy
from . import utils


def get_expected_state(B, qs_current, action, control_factors):
    """
    Compute the expected (predicted) beliefs over hidden states for each factor after taking a given action.
    For each state factor, if it is controlled (as indicated by control_factors), the function applies the corresponding transition matrix for the chosen action
    to update the belief. If the factor is not controlled, the belief remains unchanged.
    Args:
        B (object array): Transition matrices for each state factor, where B[f] is the transition matrix for factor f.
        qs_current (object array): Current beliefs (probability distributions) over states for each factor.
        action (int): The action index to consider for the transition.
        control_factors (list or array): Binary indicators specifying which factors are controlled (1) or not (0).
    Returns:
        object array: Updated beliefs over states for each factor after applying the action.
    """
    num_factors = len(B)
    qs_u = np.empty(num_factors, dtype=object)
    for f in range(num_factors):
        if control_factors[f]:
            qs_u[f] = B[f][:, :, action].dot(qs_current[f])
        else:
            qs_u[f] = qs_current[f]
    return qs_u


def get_expected_states(B, qs_current, policy, control_factors):
    """
    Compute the expected beliefs over hidden states for each factor after executing a sequence of actions (policy).
    Args:
        B               : obj-array of length F, each B[f].shape = (S_f, S_f, U)
        qs_current      : obj-array of length F, current q(s_f)
        policy          : 1-D array of length T, where policy[t] is the action at step t
        control_factors : list/array of length F (0 or 1), which factors are controllable

    Returns:
        qs_pred : list of length T, each entry an obj-array of length F
                  giving q(s_f) at t+1 under the policy
    """
    # Normalize policy to a 1-D iterable of actions
    if np.isscalar(policy):
        policy = [int(policy)]
    T = len(policy)
    F = len(B)

    # start from the current beliefs (ensure object array container)
    qs_t = np.empty(F, dtype=object)
    for f in range(F):
        qs_t[f] = np.asarray(qs_current[f])

    qs_pred = []
    for t in range(T):
        a = int(policy[t])  # same action for all factors
        qs_next = np.empty(F, dtype=object)  # placeholder for next beliefs

        for f in range(F):
            if control_factors[f]:
                # apply the transition for action a on factor f
                qs_next[f] = B[f][:, :, a].dot(qs_t[f])
            else:
                # non-controllable factors simply carry their belief forward
                qs_next[f] = qs_t[f]

        qs_pred.append(qs_next)
        qs_t = qs_next

    return qs_pred


def get_expected_obs(qs_pi, A):
    "Compute the expected observations under a policy" # DEBUG: WRONG!
    n_steps = len(qs_pi)  # each element of the list is the PPD at a different timestep

    # initialise expected observations
    qo_pi = []

    for t in range(n_steps):
        qo_pi_t = np.empty(len(A), dtype=object)
        qo_pi.append(qo_pi_t)

    # compute expected observations over time
    for t in range(n_steps):
        for modality, A_m in enumerate(A):
            qo_pi[t][modality] = spm_dot_optimized(A_m, qs_pi[t])

    return qo_pi


def get_expected_obs_optimized(qs_pi, A):
    """Compute expected observations using a vectorized path across timesteps.

    Guarantees identical output shape and values to get_expected_obs:
    - Returns a list of length n_steps
    - Each element is an object ndarray of length len(A)
    - Each entry qo_pi[t][m] is a numeric ndarray of the same shape as spm_dot(A[m], qs_pi[t])
    """
    n_steps = len(qs_pi)

    # Preallocate output structure identically to the baseline version
    qo_pi = []
    for _ in range(n_steps):
        qo_pi.append(np.empty(len(A), dtype=object))

    # Fast path for single-timestep: avoid batching overhead and einsum edge cases
    if n_steps == 1:
        for modality, A_m in enumerate(A):
            qo_pi[0][modality] = spm_dot_optimized(A_m, qs_pi[0])
        return qo_pi

    # Vectorize over timesteps per modality, then scatter results back to the list-of-obj-arrays shape
    for modality, A_m in enumerate(A):
        try:
            # spm_dot_vectorized returns a numeric array stacking results over timesteps
            stacked = spm_dot_vectorized(A_m, qs_pi)

            # Ensure we have a 2D stack along the first axis for timesteps
            if stacked.ndim == 1:
                stacked = stacked[np.newaxis, :]

            # Assign each timestep result back into the object arrays
            for t in range(n_steps):
                qo_pi[t][modality] = stacked[t]
        except ValueError:
            # Fallback to streaming per timestep to handle einsum edge cases
            for t in range(n_steps):
                qo_pi[t][modality] = spm_dot_optimized(A_m, qs_pi[t])

    return qo_pi


def log_softmax_np(x, axis=None, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=True)
    y = x - x_max
    logZ = np.log(np.sum(np.exp(y), axis=axis, keepdims=True))
    out = y - logZ
    return out if keepdims else (out if axis is None else np.squeeze(out, axis=axis))


def precompute_lnC(C):
    lnC = np.empty_like(C, dtype=object)
    for m in range(len(C)):
        if C[m].ndim == 1:
            lnC[m] = log_softmax_np(C[m])
        else:
            lnC[m] = log_softmax_np(C[m], axis=0, keepdims=True)
    return lnC


def calc_expected_utility(qo_pi, lnC):
    """
    qo_pi: list over time of object arrays; qo_pi[t][m] is (num_outcomes_m,)
    lnC:   object array; lnC[m] is (num_outcomes_m,) or (num_outcomes_m, num_steps)
    """
    num_steps = len(qo_pi)
    num_modalities = len(lnC)
    expected_util = 0.0

    for modality in range(num_modalities):
        lnC_m = lnC[modality]

        if lnC_m.ndim == 1:
            sum_q = None
            for t in range(num_steps):
                q_t = qo_pi[t][modality]
                sum_q = q_t if sum_q is None else (sum_q + q_t)
            expected_util += float(sum_q @ lnC_m)
        else:
            Q_m = np.stack([qo_pi[t][modality] for t in range(num_steps)], axis=1)
            expected_util += float(np.sum(Q_m * lnC_m))

    return expected_util


def calc_states_info_gain(A, qs_pi):
    policy_temp_depth = len(qs_pi)
    state_surprise = 0
    for t in range(policy_temp_depth):
        state_surprise += calc_surprise(A, qs_pi[t])
    return state_surprise


def vanilla_fpi_update_posterior_policies(
    qs,
    A,
    B,
    C,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    E=None,
    gamma=16.0,
    control_factors=None,
):

    num_policies = len(policies)
    G = np.zeros(num_policies)
    q_pi = np.zeros(num_policies)  # Should be 1D array, not 2D

    if E is None:
        lnE = log_stable(np.ones(num_policies) / num_policies)
    else:
        lnE = log_stable(E)

    for policy_idx, policy in enumerate(policies):
        qs_pi = get_expected_states(
            B, qs, policy, control_factors
        )  # given the policy, get the expected states at the end of the policy
        qo_pi = get_expected_obs(
            qs_pi, A
        )  # given the expected states, get the expected observations under the policy

        if use_utility:
            lnC = precompute_lnC(C)
            G[policy_idx] += calc_expected_utility(qo_pi, lnC)

        if use_states_info_gain:
            G[policy_idx] += calc_states_info_gain(A, qs_pi)

    q_pi = softmax(G * gamma + lnE)

    return q_pi, G


def sample_policy(q_pi, policies, action_selection, alpha):

    if action_selection == "deterministic":
        policy_idx = np.argmax(q_pi)
    elif action_selection == "stochastic":
        log_qpi = log_stable(q_pi)
        p_policies = softmax(log_qpi * alpha)
        policy_idx = utils.sample(p_policies)

    return int(policies[policy_idx][0])


def sample_action(q_pi, policies, action_selection, alpha, actions):
    
    # Determine the number of possible actions based on the maximum first action
    max_action = int(len(actions))
    action_marginals = np.zeros(max_action + 1)
    

    for pol_idx, policy in enumerate(policies):
        first_action_id = int(policy[0])
        # Handle different q_pi shapes - flatten and take first element if needed
        q_pi_element = q_pi[pol_idx]
        if np.isscalar(q_pi_element):
            q_pi_value = q_pi_element
        else:
            # If it's an array, flatten and take the first (and hopefully only) element
            q_pi_value = np.asarray(q_pi_element).flatten()[0]
        action_marginals[first_action_id] += q_pi_value
    
    action_marginals = action_marginals / np.sum(action_marginals)

    selected_action = None

    if action_selection == 'deterministic':
        selected_action = np.argmax(action_marginals)
    elif action_selection == 'stochastic':
        log_marginal_f = log_stable(action_marginals)
        p_actions = softmax(log_marginal_f * alpha)
        selected_action = utils.sample(p_actions)

    return selected_action
