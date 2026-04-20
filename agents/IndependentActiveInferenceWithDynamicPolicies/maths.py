"""
Mathematical utilities for Active Inference with functional generative model.

This module provides core mathematical operations for:
- Likelihood computation with functional A
- Bayesian surprise calculation  
- Variational free energy
- Numerical stability helpers
"""

import numpy as np
import itertools


# =============================================================================
# Numerical Stability
# =============================================================================

EPS_VAL = 1e-16  # Small constant to prevent log(0)


def log_stable(x, eps=EPS_VAL):
    """
    Numerically stable logarithm.
    
    Args:
        x: array-like input
        eps: small constant to prevent log(0)
    
    Returns:
        log(x + eps)
    """
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.maximum(x, eps))


def softmax(x, axis=None):
    """
    Numerically stable softmax function.

    Args:
        x: array-like input  
        axis: axis along which to apply softmax

    Returns:
        softmax(x) - probability distribution
    """
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# Functional Likelihood Computation
# =============================================================================

def compute_likelihood_for_state(A_fn, obs_dict, state_indices):
    """
    Compute p(observations | specific state) using functional A.
    
    Args:
        A_fn: functional observation model - takes state_indices, returns obs likelihoods
        obs_dict: dict mapping modality names to observed indices
        state_indices: dict mapping state factor names to specific indices
    
    Returns:
        likelihood: scalar probability p(o | s)
    
    Examples
    --------
    >>> obs_dict = {'agent_pos': 3, 'button_state': 1}
    >>> state_indices = {'agent_pos': 3, 'red_button_pos': 2, ...}
    >>> p_o_given_s = compute_likelihood_for_state(A_fn, obs_dict, state_indices)
    """
    # Get observation likelihoods for this state configuration
    obs_likelihoods = A_fn(state_indices)
    
    # Modalities that require previous state (exclude from state inference)
    EXCLUDE_MODALITIES = {'button_just_pressed'}
    
    # Multiply likelihoods across modalities (conditional independence)
    likelihood = 1.0
    for modality, obs_idx in obs_dict.items():
        # Skip modalities that can't be computed without previous state
        if modality in EXCLUDE_MODALITIES:
            continue
        if modality in obs_likelihoods:
            p_o_m = obs_likelihoods[modality][obs_idx]
            likelihood *= p_o_m
    
    return likelihood


def compute_log_likelihood_for_state(A_fn, obs_dict, state_indices):
    """
    Compute log p(observations | specific state) using functional A.
    
    More numerically stable than computing likelihood and taking log.
    
    Args:
        A_fn: functional observation model
        obs_dict: dict mapping modality names to observed indices
        state_indices: dict mapping state factor names to specific indices
    
    Returns:
        log_likelihood: scalar log probability log p(o | s)
    """
    # Get observation likelihoods for this state configuration
    obs_likelihoods = A_fn(state_indices)
    
    # Modalities that require previous state (exclude from state inference)
    EXCLUDE_MODALITIES = {'button_just_pressed'}
    
    # Sum log-likelihoods across modalities (log of product = sum of logs)
    log_likelihood = 0.0
    for modality, obs_idx in obs_dict.items():
        # Skip modalities that can't be computed without previous state
        if modality in EXCLUDE_MODALITIES:
            continue
        if modality in obs_likelihoods:
            p_o_m = obs_likelihoods[modality][obs_idx]
            log_likelihood += log_stable(p_o_m)
    
    return log_likelihood


def get_joint_likelihood_functional(A_fn, obs_dict, state_factors, state_sizes):
    """
    Compute joint likelihood p(o|s) for ALL state configurations using functional A.
    
    This iterates over all possible state combinations and evaluates A_fn.

    Args:
        A_fn: functional observation model
        obs_dict: dict of observed indices per modality
        state_factors: list of state factor names in order
        state_sizes: dict mapping factor names to number of states

    Returns:
        likelihood_array: flattened array of likelihoods for all joint states
    
    Notes
    -----
    For factorized inference, we typically don't need to build this full array.
    This is provided for compatibility with algorithms that need joint likelihood.
    """
    # Get state space dimensions
    dims = [state_sizes[factor] for factor in state_factors]
    total_states = int(np.prod(dims))
    
    # Allocate likelihood array
    likelihood = np.zeros(total_states)
    
    # Iterate over all state combinations
    for flat_idx in range(total_states):
        # Convert flat index to state indices
        state_indices = flat_idx_to_state_indices(flat_idx, state_factors, state_sizes)
        
        # Compute likelihood for this state
        likelihood[flat_idx] = compute_likelihood_for_state(A_fn, obs_dict, state_indices)
    
    return likelihood


def flat_idx_to_state_indices(flat_idx, state_factors, state_sizes):
    """
    Convert flat joint state index to dict of factor indices.
    
    Args:
        flat_idx: flat index into joint state space
        state_factors: list of factor names (defines order)
        state_sizes: dict mapping factor names to sizes
    
    Returns:
        state_indices: dict mapping factor names to indices
    
    Examples
    --------
    >>> state_factors = ['agent_pos', 'button_state']
    >>> state_sizes = {'agent_pos': 9, 'button_state': 2}
    >>> flat_idx_to_state_indices(15, state_factors, state_sizes)
    {'agent_pos': 7, 'button_state': 1}  # 15 = 7*2 + 1
    """
    state_indices = {}
    remaining = flat_idx
    
    # Process factors in reverse order (like unraveling multi-dimensional index)
    for factor in reversed(state_factors):
        size = state_sizes[factor]
        state_indices[factor] = remaining % size
        remaining = remaining // size
    
    return state_indices


def state_indices_to_flat_idx(state_indices, state_factors, state_sizes):
    """
    Convert dict of factor indices to flat joint state index.
    
    Args:
        state_indices: dict mapping factor names to indices
        state_factors: list of factor names (defines order)
        state_sizes: dict mapping factor names to sizes
    
    Returns:
        flat_idx: flat index into joint state space
    """
    flat_idx = 0
    multiplier = 1
    
    # Process factors in reverse order
    for factor in reversed(state_factors):
        flat_idx += state_indices[factor] * multiplier
        multiplier *= state_sizes[factor]
    
    return flat_idx


# =============================================================================
# Bayesian Surprise (Information Gain)
# =============================================================================
def calc_surprise_functional(A_fn, qs_dict, state_factors, state_sizes,
                             skip_modalities=('button_just_pressed',),
                             entropy_thresh=0.01, eps=1e-8):
    """
    Functional expected information gain:
      G = H[ Q(o) ] - E_{Q(s)}[ H[p(o|s)] ],
    where Q(o) = sum_s Q(s) p(o|s), and o is the JOINT observation across modalities.
    """
    import numpy as np
    import itertools

    def log_stable(p):
        return np.log(np.clip(p, eps, 1.0))

    def normalize(p):
        z = p.sum()
        return p / (z if z > 0 else 1.0)

    # --- 1) Pick which factors to enumerate (your pruning idea) ---
    qs_np = {f: np.asarray(qs_dict[f], dtype=float) for f in state_factors}
    map_indices = {f: int(np.argmax(qs_np[f])) for f in state_factors}

    dynamic_factors = set()
    for f in state_factors:
        qf = qs_np[f]
        Hf = -np.sum(qf * log_stable(qf))
        if Hf > entropy_thresh:
            dynamic_factors.add(f)

    # Only include factors that the observed modalities depend on AND are dynamic
    from generative_models.SA_ActiveInference.RedBlueButton import model_init
    dep_set = set()
    for modality, deps in model_init.observation_state_dependencies.items():
        if modality in skip_modalities:
            continue
        for d in deps:
            if d in dynamic_factors:
                dep_set.add(d)

    dep_list = sorted(dep_set)
    dep_ranges = [range(len(qs_np[d])) for d in dep_list]

    # --- 2) Enumerate the (reduced) joint state support & cache likelihoods ---
    state_weight = []
    state_like_by_mod = []  # list of dicts: modality -> p(o_m | s)
    for combo in itertools.product(*dep_ranges) if dep_ranges else [()]:
        # joint Q(s) over the enumerated factors
        w = 1.0
        s_idx = map_indices.copy()
        for d, idx in zip(dep_list, combo):
            w *= qs_np[d][idx]
            s_idx[d] = int(idx)
        if w <= eps:
            continue

        # functional A: returns dict {modality: p(o_m | s)}
        like_mod = A_fn(s_idx)
        state_weight.append(w)
        state_like_by_mod.append(like_mod)

    if not state_weight:
        return 0.0

    state_weight = np.asarray(state_weight, dtype=float)
    state_weight /= state_weight.sum()  # ensure proper normalization over enumerated states

    # --- 3) E_{Q(s)}[ H[p(o|s)] ] = sum_m E_{Q(s)}[ H[p(o_m|s)] ] ---
    cond_entropy = 0.0
    modalities = [m for m in model_init.observation_state_dependencies.keys()
                  if m not in skip_modalities]

    # We'll also need to build the JOINT predictive Q(o) across modalities
    # Accumulate qo_joint over the full joint observation space by mixing
    # per-state joint observation distributions.
    # First, collect per-modality sizes to pre-allocate the joint vector.
    O_sizes = []
    for m in modalities:
        O_sizes.append(len(model_init.observations[m]))
    joint_O = int(np.prod(O_sizes))
    qo_joint = np.zeros(joint_O, dtype=float)

    # helper: cross product of modality vectors into a flat joint vector
    def cross_to_joint(vectors):
        out = vectors[0]
        for v in vectors[1:]:
            out = np.multiply.outer(out, v)
        return out.reshape(-1)

    for w, like_mod in zip(state_weight, state_like_by_mod):
        per_state_joint = []
        for m in modalities:
            p_m = np.asarray(like_mod[m], dtype=float)
            p_m = normalize(p_m)  # safety: ensure each modality likelihood is normalized
            # accumulate conditional entropy term
            cond_entropy += w * (-np.sum(p_m * log_stable(p_m)))
            per_state_joint.append(p_m)
        # mix joint observation distribution
        po_joint = cross_to_joint(per_state_joint)
        qo_joint += w * po_joint

    # --- 4) Predictive entropy over the JOINT observations ---
    qo_joint = normalize(qo_joint)
    pred_entropy = -np.sum(qo_joint * log_stable(qo_joint))

    G = float(pred_entropy - cond_entropy)
    return G

# def calc_surprise_functional(A_fn, qs_dict, state_factors, state_sizes):
#     """
#     Calculate Bayesian surprise (expected information gain) using functional A.
    
#     G = H[p(o|qs)] - E_qs[H[p(o|s)]]
#       = Entropy of predicted observations - Expected conditional entropy
    
#     Args:
#         A_fn: functional observation model
#         qs_dict: dict of belief distributions over state factors
#         state_factors: list of state factor names
#         state_sizes: dict mapping factor names to sizes
    
#     Returns:
#         G: float, expected information gain
    
#     Notes
#     -----
#     This measures how much we expect to learn from the next observation.
#     Higher surprise → observation will be more informative about hidden state.
#     """
#     # Optimized: call A_fn once per unique state, reuse across modalities
#     from generative_models.SA_ActiveInference.RedBlueButton import model_init
#     import itertools
    
#     # Convert JAX to numpy to avoid compilation overhead
#     qs_dict_np = {f: np.array(qs_dict[f]) for f in state_factors}
    
#     map_indices = {f: int(np.argmax(qs_dict_np[f])) for f in state_factors}
#     SKIP_MODALITIES = {'button_just_pressed'}
    
#     # Only enumerate factors with uncertainty
#     ENTROPY_THRESHOLD = 0.01
#     dynamic_factors = set()
#     for f in state_factors:
#         q_f = qs_dict_np[f]
#         entropy = -np.sum(q_f * log_stable(q_f))
#         if entropy > ENTROPY_THRESHOLD:
#             dynamic_factors.add(f)
    
#     # Find deps that are dynamic
#     all_deps = set()
#     for modality, deps in model_init.observation_state_dependencies.items():
#         if modality not in SKIP_MODALITIES:
#             for dep in deps:
#                 if dep in dynamic_factors:
#                     all_deps.add(dep)
    
#     # Enumerate DYNAMIC factors only
#     dep_list = sorted(all_deps)
#     dep_ranges = [range(len(qs_dict_np[dep])) for dep in dep_list]
    
#     likelihood_cache = []
#     prob_cache = []
    
#     for combo in itertools.product(*dep_ranges):
#         joint_prob = 1.0
#         state_indices = map_indices.copy()
#         for dep, idx in zip(dep_list, combo):
#             joint_prob *= qs_dict_np[dep][idx]
#             state_indices[dep] = int(idx)
        
#         if joint_prob <= 1e-16:
#             continue
        
#         obs_likelihoods = A_fn(state_indices)
#         likelihood_cache.append(obs_likelihoods)
#         prob_cache.append(joint_prob)
    
#     # Marginalize using cached likelihoods
#     qo_per_modality = {}
#     cond_entropy_per_modality = {}
    
#     for modality, deps in model_init.observation_state_dependencies.items():
#         if modality in SKIP_MODALITIES:
#             continue
        
#         num_obs = len(model_init.observations[modality])
#         qo_m = np.zeros(num_obs)
#         cond_entropy_m = 0.0
        
#         for obs_lik, joint_prob in zip(likelihood_cache, prob_cache):
#             p_o = obs_lik[modality]
#             qo_m += joint_prob * p_o
#             H_o_given_s = -np.sum(p_o * log_stable(p_o))
#             cond_entropy_m += joint_prob * H_o_given_s
        
#         qo_per_modality[modality] = normalize(qo_m)
#         cond_entropy_per_modality[modality] = cond_entropy_m
    
#     # Approximate button_just_pressed
#     if 'button_just_pressed' in model_init.observation_state_dependencies:
#         qo_per_modality['button_just_pressed'] = np.array([0.9, 0.1])
#         cond_entropy_per_modality['button_just_pressed'] = 0.01
    
#     # Compute total entropies
#     pred_entropy = 0.0
#     cond_entropy = 0.0
    
#     for modality in qo_per_modality.keys():
#         qo = qo_per_modality[modality]
#         H_qo = -np.sum(qo * log_stable(qo))
#         pred_entropy += H_qo
#         cond_entropy += cond_entropy_per_modality[modality]
    
#     G = pred_entropy - cond_entropy
    
#     return float(G)


def get_plausible_states(qs_dict, state_factors, threshold=1e-10):
    """
    Get list of plausible state combinations with their joint probabilities.
    
    Only returns states where joint probability > threshold.
    
    Args:
        qs_dict: dict of belief distributions
        state_factors: list of factor names
        threshold: minimum probability to include
    
    Returns:
        list of (state_indices_dict, joint_prob) tuples
    
    Examples
    --------
    >>> qs_dict = {'agent_pos': np.array([0.5, 0.5, 0, ...]), 
    ...            'button_state': np.array([0.3, 0.7])}
    >>> plausible = get_plausible_states(qs_dict, ['agent_pos', 'button_state'])
    """
    # Get indices with non-zero probability for each factor
    plausible_indices = {}
    for factor in state_factors:
        qs_f = qs_dict[factor]
        plausible_indices[factor] = np.where(qs_f > threshold)[0]
    
    # Generate all combinations of plausible indices
    plausible_states = []
    for indices in itertools.product(*[plausible_indices[f] for f in state_factors]):
        state_indices = {factor: idx for factor, idx in zip(state_factors, indices)}
        
        # Compute joint probability (assumes independence)
        joint_prob = 1.0
        for factor, idx in state_indices.items():
            joint_prob *= qs_dict[factor][idx]
        
        if joint_prob > threshold:
            plausible_states.append((state_indices, joint_prob))
    
    return plausible_states


# =============================================================================
# Variational Free Energy
# =============================================================================

def calc_variational_free_energy(qs_dict, prior_dict, obs_dict=None, A_fn=None, 
                                 state_factors=None, state_sizes=None):
    """
    Calculate variational free energy for factorized beliefs.
    
    F = sum_f KL[q(s_f) || p(s_f)] - E_q[log p(o|s)]
    
    Args:
        qs_dict: dict of posterior beliefs
        prior_dict: dict of prior beliefs
        obs_dict: optional dict of observed indices (for accuracy term)
        A_fn: optional functional observation model (for accuracy term)
        state_factors: optional list of state factor names
        state_sizes: optional dict of factor sizes
    
    Returns:
        vfe: float, variational free energy
    """
    vfe = 0.0
    
    # KL divergence term (complexity) for each factor
    for factor in qs_dict.keys():
        q = qs_dict[factor]
        p = prior_dict[factor]
        
        # KL[q||p] = sum q * log(q/p) = sum q * (log q - log p)
        kl = np.sum(q * (log_stable(q) - log_stable(p)))
        vfe += kl
    
    # Accuracy term: -E_q[log p(o|s)]
    if obs_dict is not None and A_fn is not None and state_factors is not None:
        # Enumerate plausible states and compute expected log-likelihood
        plausible_states = get_plausible_states(qs_dict, state_factors, threshold=1e-10)
        
        expected_log_likelihood = 0.0
        for state_indices, joint_prob in plausible_states:
            # Compute log p(o|s) for this state
            log_lik = compute_log_likelihood_for_state(A_fn, obs_dict, state_indices)
            expected_log_likelihood += joint_prob * log_lik
        
        # Subtract accuracy from VFE (we want to maximize likelihood)
        vfe -= expected_log_likelihood
    
    return float(vfe)



# def calc_variational_free_energy(qs_dict, prior_dict, likelihood_dict=None):
#     """
#     Calculate variational free energy for factorized beliefs.
    
#     F = sum_f KL[q(s_f) || p(s_f)] - E_q[log p(o|s)]
    
#     Args:
#         qs_dict: dict of posterior beliefs
#         prior_dict: dict of prior beliefs
#         likelihood_dict: optional dict for likelihood term
    
#     Returns:
#         vfe: float, variational free energy
#     """
#     vfe = 0.0
    
#     # KL divergence term for each factor
#     for factor in qs_dict.keys():
#         q = qs_dict[factor]
#         p = prior_dict[factor]
        
#         # KL[q||p] = sum q * log(q/p) = sum q * (log q - log p)
#         kl = np.sum(q * (log_stable(q) - log_stable(p)))
#         vfe += kl
    
#     # Likelihood term (if provided)
#     if likelihood_dict is not None:
#         # This would require joint state enumeration
#         # For now, simplified version
#         pass
    
#     return float(vfe)
# =============================================================================
# Cross Product for Mean-Field Beliefs
# =============================================================================

def spm_cross(*arrays):
    """
    Compute outer product of arrays (for combining factorized beliefs).
    
    This is the functional equivalent of the tensor outer product used
    in matrix-based Active Inference.
    
    Args:
        *arrays: variable number of arrays to combine
    
    Returns:
        outer product array
    
    Examples
    --------
    >>> a = np.array([0.3, 0.7])
    >>> b = np.array([0.5, 0.5])
    >>> spm_cross(a, b)
    array([[0.15, 0.15],
           [0.35, 0.35]])
    """
    if len(arrays) == 0:
        return np.array(1.0)
    if len(arrays) == 1:
        return arrays[0]
    
    # Start with first array
    result = arrays[0]
    
    # Sequentially combine with remaining arrays
    for arr in arrays[1:]:
        # Reshape for broadcasting
        result_shape = list(result.shape) + [1] * arr.ndim
        arr_shape = [1] * result.ndim + list(arr.shape)
        
        result_broadcast = result.reshape(result_shape)
        arr_broadcast = arr.reshape(arr_shape)
        
        result = result_broadcast * arr_broadcast
    
    return result


# =============================================================================
# Helper: Normalize Distribution
# =============================================================================

def normalize(dist, axis=None):
    """
    Normalize distribution to sum to 1.
    
    Args:
        dist: array to normalize
        axis: axis along which to normalize
    
    Returns:
        normalized array
    """
    dist = np.asarray(dist, dtype=np.float64)
    return dist / np.maximum(np.sum(dist, axis=axis, keepdims=True), EPS_VAL)
