from generative_models.SA_ActiveInference.RedBlueButton.B import apply_B   # your functional B
import numpy as np
import jax.numpy as jnp
import jax

def get_expected_state(qs_current, action, env_params):
    """
    Compute expected next state distribution given current beliefs and an action,
    using functional B.

    Args:
        qs_current : dict of belief distributions over state factors
                     (keys: "agent_pos", "red_door_pos", "blue_door_pos",
                            "red_door_state", "blue_door_state", "goal_context")
        action     : int, action index
        env_params : dict with environment dimensions & params
                     (width, height, open_success, noise, etc.)

    Returns:
        qs_next : dict of updated belief distributions
    """
    width, height = env_params["width"], env_params["height"]
    return apply_B(qs_current, action, width, height)


def get_expected_states(qs_current, policy, env_params):
    """
    Roll out expected states under a sequence of actions (policy).

    Args:
        qs_current : dict of current beliefs
        policy     : list/array of actions
        env_params : dict with environment params

    Returns:
        qs_pred : list of dicts, one per timestep
    """
    if np.isscalar(policy):
        policy = [int(policy)]

    qs_pred = []
    qs_t = qs_current

    for a in policy:
        qs_next = apply_B(qs_t, int(a), env_params["width"], env_params["height"])
        qs_pred.append(qs_next)
        qs_t = qs_next

    return qs_pred


# def get_expected_obs(qs_pi, A_funcs):
#     """
#     Compute expected observation distributions under a policy using functional A.

#     Args:
#         qs_pi   : list of beliefs over time; each item is a dict of factor marginals
#                   e.g. qs_pi[t] = {
#                         "agent_pos": ...,
#                         "red_button_pos": ...,
#                         "blue_button_pos": ...,
#                         "red_button_state": ...,
#                         "blue_button_state": ...,
#                         "goal_context": ...
#                   }
#         A_funcs : dict (or list) of modality -> callable
#                   Each callable takes qs_t (dict of factor marginals) and returns
#                   a probability vector over that modality's outcomes.

#     Returns:
#         qo_pi : list of dicts, one per timestep.
#                 qo_pi[t][modality] is a 1D array: p(o_m | qs_t)
#     """
#     qo_pi = []
#     # If A_funcs is a list, give synthetic names
#     if isinstance(A_funcs, (list, tuple)):
#         modalities = list(range(len(A_funcs)))
#         get_A = lambda k: A_funcs[k]
#     else:
#         modalities = list(A_funcs.keys())
#         get_A = lambda k: A_funcs[k]

#     for qs_t in qs_pi:
#         qo_t = {}
#         for modality in modalities:
#             A_m = get_A(modality)
#             # Functional A: directly map belief over states -> obs distribution
#             qo_t[modality] = A_m(qs_t)
#         qo_pi.append(qo_t)

#     return qo_pi


def get_state_sizes(width, height):
    num_agent = width * height
    num_red = width * height
    num_blue = width * height
    num_red_state = 2
    num_blue_state = 2
    num_goal = 2
    return [num_agent, num_red, num_blue, num_red_state, num_blue_state, num_goal]

def build_decode_table(width, height):
    sizes = get_state_sizes(width, height)
    S = int(jnp.prod(jnp.array(sizes)))

    decode_table = jnp.zeros((S, len(sizes)), dtype=int)

    # unravel index -> factor tuple
    for idx in range(S):
        tmp = idx
        vals = []
        for size in reversed(sizes):
            vals.append(tmp % size)
            tmp //= size
        decode_table = decode_table.at[idx].set(jnp.array(list(reversed(vals))))
    return decode_table  # shape (S, 6)

def get_expected_obs(q_state, A_funcs, decode_table, width, height):
    """
    q_state: (S,) belief over joint states
    A_funcs: dict of modality -> function(state_tuple, width, height) -> obs_dist
    decode_table: (S, num_factors) int array
    """

    def expected_for_modality(A_func):
        def obs_for_idx(idx, prob):
            state_tuple = decode_table[idx]
            return prob * A_func(state_tuple, width, height)

        obs = jax.vmap(obs_for_idx)(jnp.arange(q_state.shape[0]), q_state)
        return jnp.sum(obs, axis=0)

    return {modality: expected_for_modality(A_func) for modality, A_func in A_funcs.items()}






#----
import numpy as np
import jax.numpy as jnp
from jax.nn import log_softmax

# ------------------------------
# Utilities
# ------------------------------
def log_softmax_np(x, axis=None, keepdims=False):
    x = np.asarray(x)
    return np.log(np.exp(x - np.max(x, axis=axis, keepdims=True)) /
                  np.sum(np.exp(x - np.max(x, axis=axis, keepdims=True)), axis=axis, keepdims=True))

def precompute_lnC(C_funcs):
    """
    Turn functional C into log-preferences.
    C_funcs: dict of modality -> callable returning preference vector
    """
    lnC = {}
    for modality, C_func in C_funcs.items():
        C_vec = np.array(C_func())  # call preference fn
        lnC[modality] = log_softmax_np(C_vec)
    return lnC

def calc_expected_utility(qo_pi, lnC):
    """
    qo_pi: list of dicts (time -> modality -> distribution)
    lnC:   dict modality -> log-preference vector
    """
    num_steps = len(qo_pi)
    expected_util = 0.0

    for modality, lnC_m in lnC.items():
        if lnC_m.ndim == 1:
            sum_q = sum(qo_pi[t][modality] for t in range(num_steps))
            expected_util += float(sum_q @ lnC_m)
        else:
            Q_m = np.stack([qo_pi[t][modality] for t in range(num_steps)], axis=1)
            expected_util += float(np.sum(Q_m * lnC_m))

    return expected_util

# ------------------------------
# Surprise / Info gain
# ------------------------------
def calc_surprise(A_funcs, qs_t, decode_table, width, height):
    """
    Negative expected log-likelihood under A.
    """
    surprise = 0.0
    for modality, A_func in A_funcs.items():
        qo = get_expected_obs(qs_t, {modality: A_func}, decode_table, width, height)[modality]
        surprise += -float(np.sum(qo * np.log(qo + 1e-16)))
    return surprise

def calc_states_info_gain(A_funcs, qs_pi, decode_table, width, height):
    return sum(calc_surprise(A_funcs, qs_t, decode_table, width, height) for qs_t in qs_pi)

# ------------------------------
# Posterior over policies
# ------------------------------
def vanilla_fpi_update_posterior_policies(
    qs,
    A_funcs,
    B,
    C_funcs,
    policies,
    decode_table,
    env_params,
    use_utility=True,
    use_states_info_gain=True,
    E=None,
    gamma=16.0,
):
    num_policies = len(policies)
    G = np.zeros(num_policies)

    if E is None:
        lnE = log_softmax_np(np.ones(num_policies) / num_policies)
    else:
        lnE = log_softmax_np(E)

    for policy_idx, policy in enumerate(policies):
        # Predict states
        qs_pi = get_expected_states(qs, policy, env_params)
        # Predict observations
        qo_pi = [get_expected_obs(qs_t, A_funcs, decode_table,
                                  env_params["width"], env_params["height"])
                 for qs_t in qs_pi]

        if use_utility:
            lnC = precompute_lnC(C_funcs)
            G[policy_idx] += calc_expected_utility(qo_pi, lnC)

        if use_states_info_gain:
            G[policy_idx] += calc_states_info_gain(A_funcs, qs_pi,
                                                   decode_table,
                                                   env_params["width"], env_params["height"])

    q_pi = np.exp(G * gamma + lnE)
    q_pi /= np.sum(q_pi)
    return q_pi, G


