import numpy as np
import jax.numpy as jnp

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
    
    # Fill the decode table
    for i in range(S):
        remaining = i
        for j in range(len(sizes) - 1, -1, -1):
            decode_table = decode_table.at[i, j].set(remaining % sizes[j])
            remaining = remaining // sizes[j]
    
    return decode_table


def get_joint_likelihood(A_funcs, obs, num_states, decode_table=None, width=3, height=3):
    """
    Compute joint likelihood p(o|s) for functional A.

    Args:
        A_funcs : dict of callables
            Each A_funcs[m](state_tuple, width, height) -> np.array of shape (O_m,)
            distribution over outcomes for modality m.
        obs : list of one-hot np.arrays
            Observations for each modality.
        num_states : list[int]
            Number of states per factor.
        decode_table : np.ndarray, optional
            Shape (prod(num_states), len(num_states)), mapping flat index -> factor tuple.
        width : int, optional
            Environment width for A functions
        height : int, optional
            Environment height for A functions

    Returns:
        joint_likelihood : np.ndarray
            Shape = (prod(num_states),), likelihood for each joint state.
    """
    if isinstance(num_states, int):
        num_states = [num_states]

    S = int(np.prod(num_states))

    # Precompute decode table if not given
    if decode_table is None:
        ranges = [range(n) for n in num_states]
        decode_table = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(len(num_states), -1).T

    joint_likelihood = np.ones(S)

    # For each modality
    for m, f in A_funcs.items():
        obs_idx = int(obs[m].argmax())  # which observation was seen
        likelihood_m = np.zeros(S)
        for s_idx in range(S):
            state_tuple = tuple(decode_table[s_idx])
            likelihood_m[s_idx] = f(state_tuple, width, height)[obs_idx]
        joint_likelihood *= likelihood_m

    return joint_likelihood


def calc_accuracy(log_likelihood, qs, decode_table):
    """
    Expected log-likelihood E_q[ln p(o|s)].
    log_likelihood: np.ndarray over joint states (already log-probs)
    qs: list of factorized marginals
    decode_table: mapping to joint states
    """
    q_joint = joint_from_marginals(qs, decode_table)
    return float(np.sum(q_joint * log_likelihood))


def calc_surprise(A_funcs, qs, decode_table, obs_sizes):
    """
    Functional Bayesian surprise (expected information gain).

    A_funcs: dict of callables, each f(state_tuple) -> obs distribution
    qs: list of factorized marginals
    decode_table: flat->tuple mapping of all states
    obs_sizes: dict { modality: number of possible outcomes }
    """
    q_joint = joint_from_marginals(qs, decode_table)

    # Predictive observation distribution Q(o)
    qo = {m: np.zeros(o_size) for m, o_size in obs_sizes.items()}
    for s_idx, q in enumerate(q_joint):
        state_tuple = tuple(decode_table[s_idx])
        for m, f in A_funcs.items():
            qo[m] += q * f(state_tuple)

    # Entropy of Q(o) minus expected conditional entropy
    G = 0.0
    for m, f in A_funcs.items():
        qom = qo[m] / np.sum(qo[m])
        H_Qo = -np.sum(qom * np.log(qom + 1e-16))

        H_cond = 0.0
        for s_idx, q in enumerate(q_joint):
            state_tuple = tuple(decode_table[s_idx])
            pom = f(state_tuple)
            H_cond += q * (-np.sum(pom * np.log(pom + 1e-16)))

        G += H_Qo - H_cond

    return float(G)


import numpy as np
import jax.numpy as jnp

def joint_from_marginals(qs, decode_table):
    """
    Combine factorized marginals qs[f] into a flat joint distribution.
    qs: list of arrays, one per factor
    decode_table: np.ndarray of shape (num_states, num_factors)
    """
    q_joint = []
    for state_tuple in decode_table:
        prob = 1.0
        for f, idx in enumerate(state_tuple):
            prob *= qs[f][idx]
        q_joint.append(prob)
    q_joint = np.array(q_joint, dtype=np.float64)
    q_joint /= np.sum(q_joint)
    return q_joint

# ---------- Utility Functions ----------
def log_stable(x, eps=1e-16):
    """
    Log with numerical stability.
    """
    return np.log(np.maximum(x, eps))

def obj_log_stable(x_list, eps=1e-16):
    """
    Apply log stabilization to a list of arrays.
    """
    return [log_stable(x, eps) for x in x_list]

def calc_variational_free_energy(qs, prior, num_factors, likelihood=None):
    """
    Calculate variational free energy.
    """
    # Simplified VFE calculation
    vfe = 0.0
    
    # Add prior terms
    for i, (q, p) in enumerate(zip(qs, prior)):
        vfe += np.sum(q * log_stable(q) - q * log_stable(p))
    
    # Add likelihood term if provided
    if likelihood is not None:
        # This is a simplified version - in practice you'd need to handle joint states
        pass
    
    return vfe

def softmax(x, axis=None):
    """
    Stable softmax function.
    """
    x = np.array(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
