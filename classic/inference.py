import numpy as np
from . import maths


def vanilla_fpi_update_posterior_states(A, obs, prior, num_obs, num_states, **kwargs):
    # Extract parameters from kwargs with defaults
    num_iter = kwargs.get("num_iter", 10)
    dF = kwargs.get("dF", 1.0)
    dF_tol = kwargs.get("dF_tol", 0.001)

    num_factors = len(num_states)

    """
    Step 1:
    Given this observation, what's the likelihood of being in each state?
    This block of code is taking your per-modality observation likelihoods (the A-matrices) and your actual observations,
    and combining them into one joint likelihood over all hidden‐state factors
    If the modalities are conditionally independent given the hidden state s, then the chance of seeing all your observations
    at once is just the product of the chances of seeing each one individually.
    """
    likelihood = maths.get_joint_likelihood(A, obs, num_states)
    likelihood = maths.log_stable(likelihood)

    """
    Step 2: 
    Create a flat posterior (and prior if needed) to start with.
    """
    # 1) Posterior
    qs = np.array([np.full(s, 1.0 / s) for s in num_states], dtype=object)

    # 2) If no prior was given, copy qs
    if prior is None:
        prior = qs.copy()
    prior = maths.obj_log_stable(prior)

    """
    Step 3: 
    Initialize initial variational free energy
    """
    prev_vfe = maths.calc_variational_free_energy(qs, prior, num_factors)

    """
    Step 4:
    
    """
    if num_factors == 1:
        qL = maths.spm_dot(likelihood, qs, [0])
        return np.array([maths.softmax(qL + prior[0])], dtype=object)

    else:
        curr_iter = 0
        while curr_iter < num_iter and dF >= dF_tol:

            vfe = 0

            # Build joint mean-field belief: q(s1, s2, ...) = q1(s1) * q2(s2) * ...
            qs_all = qs[0]
            for factor in range(num_factors - 1):
                qs_all = qs_all[..., None] * qs[factor + 1]

            # Compute unnormalized joint posterior: likelihood * joint_belief
            LL_tensor = likelihood * qs_all

            # Coordinate ascent updates for each factor
            for factor, qs_i in enumerate(qs):
                # Marginalize out other factors and divide by current factor belief
                qL = np.einsum(LL_tensor, list(range(num_factors)), [factor]) / qs_i

                qs[factor] = maths.softmax(qL + prior[factor])

            # Calculate new variational free energy
            vfe = maths.calc_variational_free_energy(qs, prior, num_factors, likelihood)

            # Update convergence criteria
            dF = np.abs(prev_vfe - vfe)
            prev_vfe = vfe
            curr_iter += 1

        return qs




    
    
    
    