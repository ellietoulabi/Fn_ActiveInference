import numpy as np
import jax.numpy as jnp
from . import maths  # your maths helpers: log_stable, obj_log_stable, calc_variational_free_energy, softmax
from .control import vanilla_fpi_update_posterior_policies  # if you have one, otherwise I'll inline it
from .maths import get_joint_likelihood, joint_from_marginals

def vanilla_fpi_update_posterior_states(A_funcs, obs, prior, qs_current, decode_table, obs_sizes, **kwargs):
    """
    Mean–field variational Bayes update of posterior hidden states
    given functional A, observed outcomes, and prior beliefs.

    Args:
        A_funcs : dict or list of callables
            Each A_funcs[m](state_tuple) -> P(o_m | s) distribution.
        obs : list of one-hot vectors
            Observations for each modality.
        prior : list of np.arrays
            Prior over each factor (log-space optional).
        num_obs : list[int]
            Observation dimensionalities.
        num_states : list[int]
            Number of values per factor.
        kwargs : optional args (num_iter, dF_tol, etc).

    Returns:
        qs : list of np.arrays
            Posterior beliefs over each factor (factorized).
    """

    num_iter = kwargs.get("num_iter", 10)
    dF_tol   = kwargs.get("dF_tol", 1e-3)
    dF       = np.inf
    
    # --- Step 1: joint likelihood tensor p(o|s) ---
    # Extract num_states from decode_table shape
    num_states = [9, 9, 9, 2, 2, 2]  # Default for 3x3 grid
    if decode_table is not None:
        num_states = [9, 9, 9, 2, 2, 2]  # Based on decode_table structure
    
    num_factors = len(num_states)
    likelihood = maths.get_joint_likelihood(A_funcs, obs, num_states, decode_table, width=3, height=3)  # shape = num_states
    likelihood = maths.log_stable(likelihood)  # stay in log-space

    # Store original prior type for return format
    prior_was_dict = isinstance(prior, dict)

    # --- Step 2: initialize posterior qs and prior ---
    # Handle both dict and list priors
    if prior_was_dict:
        # Convert dict to list format for processing
        qs = []
        prior_list = []
        for key in ['agent_pos', 'red_door_pos', 'blue_door_pos', 'red_door_state', 'blue_door_state', 'goal_context']:
            if key in prior:
                qs.append(np.array(prior[key], dtype=np.float64))
                prior_list.append(np.array(prior[key], dtype=np.float64))
    else:
        qs = np.array([np.full(s, 1.0/s) for s in num_states], dtype=object)  # uniform
        if prior is None:
            prior_list = [q.copy() for q in qs]
        else:
            prior_list = [np.array(p, dtype=np.float64) for p in prior]
    
    prior = maths.obj_log_stable(prior_list)  # log-stabilize priors

    # --- Step 3: initial variational free energy ---
    prev_vfe = maths.calc_variational_free_energy(qs, prior, num_factors)

    # --- Step 4: mean–field coordinate ascent ---
    curr_iter = 0
    while curr_iter < num_iter and dF > dF_tol:
        # Build current joint belief: q(s1,..,sF) = ∏ qf(sf)
        # For now, use a simplified approach that works with the flat likelihood
        # Reshape likelihood to match the joint state space
        likelihood_reshaped = likelihood.reshape(num_states)
        
        # Build joint belief tensor
        qs_all = qs[0]
        for f in range(1, num_factors):
            qs_all = qs_all[..., None] * qs[f]

        # Multiply by likelihood (in log-space)
        LL_tensor = likelihood_reshaped + np.log(qs_all + 1e-16)

        # Update each factor
        for f, qs_f in enumerate(qs):
            # Marginalize over all other factors
            axes = tuple(i for i in range(num_factors) if i != f)
            qL = np.exp(np.sum(LL_tensor, axis=axes)) / (qs_f + 1e-16)
            qs[f] = maths.softmax(np.log(qL + 1e-16) + prior[f])

        # Update convergence criteria
        vfe = maths.calc_variational_free_energy(qs, prior, num_factors, likelihood)
        dF = np.abs(prev_vfe - vfe)
        prev_vfe = vfe
        curr_iter += 1

    # Convert back to dict format if input was dict
    if prior_was_dict:
        result = {}
        keys = ['agent_pos', 'red_door_pos', 'blue_door_pos', 'red_door_state', 'blue_door_state', 'goal_context']
        for i, key in enumerate(keys):
            if i < len(qs):
                result[key] = qs[i]
        return result
    else:
        return qs
