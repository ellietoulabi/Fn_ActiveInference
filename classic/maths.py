import numpy as np
from pymdp import utils
from pymdp.maths import spm_log_single
    
    
EPS_VAL = 1e-16  # global constant for use in spm_log() function


def get_joint_likelihood(
    A: np.ndarray, obs: np.ndarray, num_states
) -> np.ndarray:
    """
    Joint likelihood under conditional‐independence of modalities,
    assuming each obs[m] is a one-hot vector.

    Parameters
    ----------
    A : obj-array of length M
        Each A[m].shape = (O_m, S1, S2, …) is the likelihood tensor for modality m.
    obs : obj-array of length M
        Each obs[m] is a one-hot vector of length O_m.
    num_states : list of ints or int
        The shape of the joint state‐space.

    Returns
    -------
    joint_likelihood : ndarray of shape tuple(num_states)
        The joint likelihood p(o¹,…,oᴹ | s¹,…,sᶠ).
    """
    # Ensure consistent types
    A = np.array(A, dtype=object)
    obs = np.array(obs, dtype=object)

    # Handle singleton case
    if isinstance(num_states, int):
        num_states = [num_states]

    # Initialise the joint likelihood over the entire state‐space
    joint_likelihood = np.ones(tuple(num_states),dtype=object)

    # Incorporate each modality by slicing the observed outcome
    for m, Am in enumerate(A):
        k = int(obs[m].argmax())  # observed index for modality m
        joint_likelihood *= Am[k, ...]

    return joint_likelihood


def log_stable(arr: np.ndarray) -> np.ndarray:
    # return np.log(arr + EPS_VAL)
    arr = np.asarray(arr, dtype=np.float64)
    return np.log(arr + EPS_VAL)

def obj_log_stable(obj_arr):
    """
    Apply a numerically-stable log to each element of an object array.
    """
    logged = np.empty(len(obj_arr), dtype=object)
    for idx, arr in enumerate(obj_arr):
        if arr is None:
            raise ValueError(f"Cannot take log of None at index {idx}")
        # Ensure the array is properly converted to float64 to match pymdp behavior
        arr_float = np.asarray(arr, dtype=np.float64)
        logged[idx] = log_stable(arr_float)
    return logged

def calc_accuracy(log_likelihood: np.ndarray, qs: np.ndarray) -> float:
    """
    Optimized expected log-likelihood E_q[ln p(o|s)] computation.
    
    Uses fast paths for common cases and optimized einsum for general cases,
    maintaining identical results to pymdp's compute_accuracy.
    """
    # Pre-convert qs to float64 arrays for efficiency
    qs_float = [np.asarray(q, dtype=np.float64) for q in qs]
    
    ndims_ll, n_factors = log_likelihood.ndim, len(qs_float)
    
    # Fast path for single factor cases (very common in practice)
    if n_factors == 1:
        if ndims_ll == 1:
            # Vector dot product - fastest possible case
            return float(np.dot(log_likelihood, qs_float[0]))
        else:
            # Multi-dimensional single factor - use tensordot on last axis
            axis = ndims_ll - 1
            result = np.tensordot(log_likelihood, qs_float[0], axes=([axis], [0]))
            return float(np.sum(result))
    
    # For multi-factor cases, use the proven einsum approach with optimizations
    else:
        from itertools import chain
        
        # Calculate dimensions (same as pymdp logic)
        dims = list(range(ndims_ll - n_factors, ndims_ll))
        
        # Build argument list efficiently (avoid repeated list operations)
        arg_list = [log_likelihood, list(range(ndims_ll))]
        for i, q in enumerate(qs_float):
            arg_list.extend([q, [dims[i]]])
        
        # Use optimized einsum
        result = np.einsum(*arg_list, optimize=True)
        
        return float(np.sum(result))


def calc_variational_free_energy(
    qs: np.ndarray, prior: np.ndarray, n_factors, likelihood: np.ndarray = None
) -> float:
    """
    Calculate variational free energy matching pymdp's calc_free_energy implementation.
    
    F = ∑_f [ q_f · log(q_f) - q_f · log(p_f) ] - accuracy(likelihood, qs)
    
    Parameters:
    -----------
    qs : list of arrays
        Posterior beliefs over factors
    prior : list of arrays
        Prior beliefs over factors  
    n_factors : int
        Number of factors (matches pymdp interface)
    likelihood : array, optional
        Likelihood tensor for accuracy calculation
    """
    # Convert n_factors parameter to handle both int and array inputs
    if hasattr(n_factors, '__len__'):
        num_factors = len(n_factors)
    else:
        num_factors = n_factors
    
    free_energy = 0.0
    
    # Process each factor (matches pymdp's loop structure exactly)
    for factor in range(num_factors):
        q = np.asarray(qs[factor], dtype=np.float64)
        p = np.asarray(prior[factor], dtype=np.float64)
        
        # Negative entropy of posterior marginal H(q[f])
        # Note: pymdp uses q.dot(np.log(q[:, np.newaxis] + 1e-16))
        # The [:, np.newaxis] is for broadcasting but doesn't change the math for 1D arrays
        negH_qs = q.dot(np.log(q + 1e-16))
        
        # Cross entropy of posterior marginal with prior marginal H(q[f],p[f])
        # Note: pymdp uses -q.dot(prior[:, np.newaxis]) which is just -q.dot(p)
        # This is NOT the standard cross-entropy formula - it's pymdp's specific implementation
        xH_qp = -q.dot(p)
        
        free_energy += negH_qs + xH_qp
    
    # Accuracy term (subtract expected log-likelihood)
    # Note: pymdp's compute_accuracy works with likelihood values directly
    # We need to match this behavior using our optimized calc_accuracy
    if likelihood is not None:
        # For pymdp compatibility, we compute accuracy as q.dot(likelihood) when likelihood has
        # the same number of dimensions as factors, otherwise use our tensor contraction
        likelihood_array = np.asarray(likelihood, dtype=np.float64)
        if likelihood_array.ndim == num_factors:
            # Simple case: direct tensor contraction with likelihood values
            accuracy = calc_accuracy(likelihood_array, qs)  
        else:
            # Complex case: use pymdp's approach for compatibility
            from pymdp.maths import compute_accuracy
            accuracy = compute_accuracy(likelihood, qs)
        free_energy -= accuracy
    
    return float(free_energy)




def spm_dot_optimized(X, x, dims_to_omit=None):
    """
    Optimized dot product focusing on the most common performance bottlenecks.
    
    Key optimizations:
    - Remove optimize=True from einsum for small arrays (adds overhead)
    - Use the original chain approach which is actually efficient
    - Minimize function calls and type conversions
    - Fast path for most common single-factor case
    
    Parameters
    ----------
    X : numpy.ndarray
        Multidimensional array to perform dot product with
    x : numpy.ndarray or obj_array
        Either vector or array of arrays to dot with X
    dims_to_omit : list of int, optional
        Which dimensions to omit from summation
    
    Returns 
    -------
    Y : numpy.ndarray
        Result of the dot product
    """
    # Use original approach but without itertools import overhead
    # Construct dims to perform dot product on (exact original logic)
    if hasattr(x, 'dtype') and x.dtype == object:
        dims = list(range(X.ndim - len(x), len(x) + X.ndim - len(x)))
    else:
        dims = [1]  # Original explicitly sets this for non-object arrays
        # Convert to object array (minimal overhead)
        x_obj = np.empty(1, dtype=object)
        x_obj[0] = np.asarray(x).squeeze()
        x = x_obj
    
    if dims_to_omit is not None:
        # Build arg list directly without chain
        arg_list = [X, list(range(X.ndim))]
        for xdim_i in range(len(x)):
            if xdim_i not in dims_to_omit:
                arg_list.extend([x[xdim_i], [dims[xdim_i]]])
        arg_list.append(dims_to_omit)
    else:
        # Build arg list directly without chain  
        arg_list = [X, list(range(X.ndim))]
        for xdim_i in range(len(x)):
            arg_list.extend([x[xdim_i], [dims[xdim_i]]])
        arg_list.append([0])

    # Use einsum WITHOUT optimize=True for better performance on small arrays
    try:
        Y = np.einsum(*arg_list)
    except ValueError as e:
        if "too many subscripts" in str(e):
            # Handle edge case where dimensions don't work out (e.g., single element arrays)
            # Fall back to a simple approach
            if len(x) == 1 and X.size == 1:
                # Single element case: just return the single element as array
                return np.array([X.item() * np.sum(x[0])], dtype=np.float64)
            else:
                raise e
        else:
            raise e

    # Fast scalar handling
    if Y.size <= 1:
        return np.array([Y.item()], dtype=np.float64)

    return Y


def spm_dot_vectorized(X, x, dims_to_omit=None):
    """
    Vectorized version of spm_dot that can handle batch operations efficiently.
    
    Parameters
    ----------
    X : numpy.ndarray
        Multidimensional array(s) to perform dot product with
    x : numpy.ndarray or obj_array or list of obj_arrays
        Either vector(s) or array(s) of arrays to dot with X
    dims_to_omit : list of int, optional
        Which dimensions to omit from summation
    
    Returns 
    -------
    Y : numpy.ndarray
        Result of the dot product(s)
    """
    # Check if we have batch inputs
    if isinstance(x, list) and len(x) > 1 and all(isinstance(xi, np.ndarray) and xi.dtype == object for xi in x):
        # Batch processing
        results = []
        for xi in x:
            results.append(spm_dot_optimized(X, xi, dims_to_omit))
        return np.array(results)
    else:
        # Single input - delegate to optimized version
        return spm_dot_optimized(X, x, dims_to_omit)


def spm_dot(X, x, dims_to_omit=None):
    """
    Original spm_dot implementation for compatibility.
    
    Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product
    
    Parameters
    ----------
    X : numpy.ndarray
        Multidimensional array
    x : numpy.ndarray or obj_array
        Either vector or array of arrays
    dims_to_omit : list of int, optional
        Which dimensions to omit
    
    Returns 
    -------
    Y : numpy.ndarray
        Result of the dot product
    """
    from itertools import chain
    
    # Construct dims to perform dot product on
    if isinstance(x, np.ndarray) and x.dtype == object:
        dims = list(range(X.ndim - len(x), len(x) + X.ndim - len(x)))
    else:
        dims = [1]
        # Convert to object array
        x_obj = np.empty(1, dtype=object)
        x_obj[0] = np.asarray(x).squeeze()
        x = x_obj

    if dims_to_omit is not None:
        arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x)) if xdim_i not in dims_to_omit))) + [dims_to_omit]
    else:
        arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x))))) + [[0]]

    Y = np.einsum(*arg_list)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y


def softmax(dist):
    """ 
    Original softmax function - computes the softmax function on a set of values
    """
    # Handle edge cases
    if isinstance(dist, (int, float)):
        dist = np.array([dist])
    elif hasattr(dist, 'shape') and dist.shape == ():
        dist = np.array([dist.item()])
    
    # Ensure it's a numpy array
    dist = np.asarray(dist)
    
    # Handle 1D case
    if dist.ndim == 1:
        max_val = np.max(dist)
        # Ensure max_val is a numpy scalar
        max_val = np.asarray(max_val, dtype=dist.dtype)
        output = dist - max_val
        # Ensure output is a proper numeric array
        output = np.asarray(output, dtype=np.float64)
        output = np.exp(output)
        output = output / np.sum(output)
    else:
        # Multi-dimensional case
        max_vals = np.max(dist, axis=0)
        max_vals = np.asarray(max_vals, dtype=dist.dtype)
        output = dist - max_vals
        output = np.exp(output)
        output = output / np.sum(output, axis=0)
    
    return output


def softmax_optimized(dist):
    """
    Optimized softmax with several performance improvements:
    - Single array allocation and in-place operations
    - Efficient axis handling
    - Reduced function call overhead
    - Better numerical stability
    
    Parameters
    ----------
    dist : numpy.ndarray
        Input array to apply softmax to
        
    Returns
    -------
    numpy.ndarray
        Softmax-normalized array
    """
    # Convert to float64 for numerical stability and efficiency
    dist = np.asarray(dist, dtype=np.float64)
    
    # Handle different input shapes efficiently
    if dist.ndim == 1:
        # 1D case - most common and fastest path
        max_val = np.max(dist)
        shifted = dist - max_val
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)
    else:
        # Multi-dimensional case
        # Use keepdims to avoid shape broadcasting issues
        max_vals = np.max(dist, axis=0, keepdims=True)
        shifted = dist - max_vals
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)


def softmax_inplace(dist):
    """
    In-place softmax for maximum memory efficiency.
    Modifies the input array directly to save memory.
    
    WARNING: This modifies the input array!
    
    Parameters
    ----------
    dist : numpy.ndarray
        Input array to apply softmax to (will be modified in-place)
        
    Returns
    -------
    numpy.ndarray
        The same array (modified in-place)
    """
    # Ensure we have a float array for in-place operations
    if dist.dtype != np.float64:
        dist = dist.astype(np.float64)
    
    if dist.ndim == 1:
        # 1D in-place operations
        max_val = np.max(dist)
        dist -= max_val  # in-place subtraction
        np.exp(dist, out=dist)  # in-place exponential
        sum_val = np.sum(dist)
        dist /= sum_val  # in-place division
    else:
        # Multi-dimensional in-place operations
        max_vals = np.max(dist, axis=0, keepdims=True)
        dist -= max_vals  # in-place subtraction
        np.exp(dist, out=dist)  # in-place exponential
        sum_vals = np.sum(dist, axis=0, keepdims=True)
        dist /= sum_vals  # in-place division
    
    return dist


def softmax_stable(dist, temperature=1.0):
    """
    Numerically stable softmax with temperature scaling.
    
    Features:
    - Enhanced numerical stability with configurable epsilon
    - Temperature parameter for controlling sharpness
    - Robust handling of edge cases (inf, nan, very large/small values)
    - Optimized for both speed and stability
    
    Parameters
    ----------
    dist : numpy.ndarray
        Input array to apply softmax to
    temperature : float, default=1.0
        Temperature parameter. Higher values make distribution more uniform,
        lower values make it more peaked.
        
    Returns
    -------
    numpy.ndarray
        Temperature-scaled softmax-normalized array
    """
    # Convert and validate input
    dist = np.asarray(dist, dtype=np.float64)
    
    # Handle temperature scaling
    if temperature != 1.0:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        dist = dist / temperature
    
    # Handle edge cases
    if np.any(np.isinf(dist)):
        # If we have inf values, handle them specially
        is_inf = np.isinf(dist)
        if np.all(is_inf):
            # All values are inf - return uniform distribution
            return np.ones_like(dist) / dist.shape[0]
        else:
            # Some values are inf - they dominate
            result = np.zeros_like(dist)
            result[is_inf] = 1.0 / np.sum(is_inf)
            return result
    
    # Standard stable softmax computation
    if dist.ndim == 1:
        # 1D case
        max_val = np.max(dist)
        # Additional stability: clamp the shifted values
        shifted = np.clip(dist - max_val, -500, 500)  # Prevent overflow/underflow
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        
        # Handle numerical edge case where sum is too small
        if sum_exp < 1e-300:
            return np.ones_like(dist) / len(dist)
        
        return exp_vals / sum_exp
    else:
        # Multi-dimensional case
        max_vals = np.max(dist, axis=0, keepdims=True)
        shifted = np.clip(dist - max_vals, -500, 500)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals, axis=0, keepdims=True)
        
        # Handle numerical edge cases
        sum_exp = np.maximum(sum_exp, 1e-300)
        
        return exp_vals / sum_exp


def softmax_vectorized(dists):
    """
    Vectorized softmax for batch processing multiple distributions.
    Optimized for processing many distributions at once.
    
    Parameters
    ----------
    dists : numpy.ndarray
        Array where each column (or row) is a separate distribution
        Shape: (n_features, n_distributions) or (n_distributions, n_features)
        
    Returns
    -------
    numpy.ndarray
        Batch softmax-normalized arrays
    """
    dists = np.asarray(dists, dtype=np.float64)
    
    if dists.ndim == 1:
        # Single distribution
        return softmax_optimized(dists)
    elif dists.ndim == 2:
        # Batch of distributions - assume each column is a distribution
        max_vals = np.max(dists, axis=0, keepdims=True)
        shifted = dists - max_vals
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
    else:
        # Higher dimensional - apply along last axis
        max_vals = np.max(dists, axis=-1, keepdims=True)
        shifted = dists - max_vals
        exp_vals = np.exp(dists - max_vals)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)



def spm_cross_manual_broadcast(x, y=None, *args):
    """Manual broadcasting approach - fastest in benchmarks"""
    
    # Handle single array case
    if y is None and len(args) == 0:
        return x
    
    # Collect all arrays
    arrays = [x]
    if y is not None:
        arrays.append(y)
    arrays.extend(args)
    
    # Start with the first array
    result = arrays[0]
    
    # Process each subsequent array
    for arr in arrays[1:]:
        # Create new shapes for broadcasting
        result_shape = list(result.shape) + [1] * arr.ndim
        arr_shape = [1] * result.ndim + list(arr.shape)
        
        # Reshape for broadcasting
        result_broadcast = result.reshape(result_shape)
        arr_broadcast = arr.reshape(arr_shape)
        
        # Multiply - NumPy's broadcasting handles the rest
        result = result_broadcast * arr_broadcast
    
    # Handle edge case: if all input arrays have size 1, return scalar to match pymdp
    if all(arr.size == 1 for arr in arrays):
        return np.asarray(result.item())
    
    return result


def calc_surprise(A, x):
    """
    Calculate Bayesian surprise (expected information gain) with optimized cross products.
    
    Think of it like this:
        - From your current belief about states, you predict what you might see next (Q(o)).
        - Ask: "How unpredictable are those possible observations overall?" (entropy of Q(o)).
        - Then ask: "If I actually knew the true state, how unpredictable would observations be on average?" (average conditional entropy).
        - The difference is the expected info you'll gain from the next observation—i.e., how much it should help you figure out the hidden state.
        
    
    Parameters
    ----------
    A : numpy ndarray or array-object
        Array assigning likelihoods of observations/outcomes under the various 
        hidden state configurations
        or
        for each hidden state, what observations are probable (p(o | s))
    x : numpy ndarray or array-object
        Categorical distribution presenting probabilities of hidden states 
        or 
        current belief over hidden states (q(s))
        
    Returns
    -------
    G : float
        The (expected or not) Bayesian surprise under the density specified by x
    """
    # Extract the number of modalities
    num_modalities = len(A)

    # Combine the factorized beliefs in x into a single joint belief qx over full state configurations.
    if len(x) == 1:
        qx = x[0]
    else:
        qx = spm_cross_manual_broadcast(*x) # P(s) = P(s1) * P(s2) * ... * P(sN) NOTE: assumes independence of the factors
    
    G = 0 # surprise
    qo = 0 #  overall forecast of observations; “What I expect to see on average before observing.
    idx = np.array(np.where(qx > np.exp(-16))).T # indices of the states with non-zero probability

    if utils.is_obj_array(A):
        for i in idx: #For each plausible state configuration i
            # Compute po: the observation distribution if state i were true, p(o | s=i). If there are multiple modalities, combine them.
            po = np.ones(1) # initialize po to 1; po: “What I’d see if this specific state were true.” for all modalities
            for modality_idx, A_m in enumerate(A): # for each modality
                index_vector = [slice(0, A_m.shape[0])] + list(i) # = A_m[:, i1, i2, i3]; slice(0, A_m.shape[0]) is the first dimension of A_m (all possible outcomes in this modality); list(i) is the indices of the states;
                # OPTIMIZATION: Use faster cross product
                po = spm_cross_manual_broadcast(po, A_m[tuple(index_vector)]) # po = po * A_m[:, i1, i2, i3]; po is the observation distribution if state i were true.; A_m[:, i1, i2, i3]: likelihood of the observations in modality m given the state;

            po = po.ravel() # let's not forget that we want to calc the surprise eventually not po and qo so it's fine to ravel it
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
    else:
        for i in idx:
            po = np.ones(1)
            index_vector = [slice(0, A.shape[0])] + list(i)
            # OPTIMIZATION: Use faster cross product
            po = spm_cross_manual_broadcast(po, A[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
   
    G = G - qo.dot(spm_log_single(qo))
    return G

    
    
    
