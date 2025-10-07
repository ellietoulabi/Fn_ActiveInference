"""
Utility functions for Active Inference agent with functional generative model.

This module provides helper functions for:
- Policy construction
- Action sampling
- Observation formatting
- State belief conversion
"""

import numpy as np
import itertools


# =============================================================================
# Policy Construction
# =============================================================================

def construct_policies(actions, policy_len):
    """
    Construct all possible policies of given length.
    
    Args:
        actions: list of available actions
        policy_len: length of each policy
    
    Returns:
        list of policies, where each policy is a list of actions
    
    Examples
    --------
    >>> actions = [0, 1, 2]
    >>> policies = construct_policies(actions, policy_len=2)
    >>> len(policies)
    9
    >>> policies[0]
    [0, 0]
    """
    if policy_len == 1:
        return [[action] for action in actions]
    else:
        # Generate all combinations with replacement
        policies = list(itertools.product(actions, repeat=policy_len))
        return [list(policy) for policy in policies]


# =============================================================================
# Action/Policy Sampling
# =============================================================================

def sample_action(q_pi, policies, action_selection="deterministic", alpha=16.0, actions=None):
    """
    Sample action from policy posterior (marginal sampling).
    
    Marginalizes over policies to get action distribution, then samples.
    
    Args:
        q_pi: policy posterior probabilities (1D array)
        policies: list of policies, each policy is a list of actions
        action_selection: "deterministic" or "stochastic"
        alpha: precision parameter for action selection (inverse temperature)
        actions: list of available actions
    
    Returns:
        selected action (int)
    
    Notes
    -----
    Alpha (precision) controls determinism:
    - High alpha → more deterministic (confident) selection
    - Low alpha → more stochastic (exploratory) selection
    """
    # Determine number of actions
    if actions is not None:
        num_actions = len(actions)
    else:
        num_actions = max(max(policy) for policy in policies) + 1
    
    # Marginalize policy posterior to get action distribution
    action_marginals = np.zeros(num_actions)
    for pol_idx, policy in enumerate(policies):
        first_action = int(policy[0])
        action_marginals[first_action] += q_pi[pol_idx]
    
    # Normalize
    action_marginals = action_marginals / np.sum(action_marginals)
    
    # Sample action
    if action_selection == "deterministic":
        selected_action = np.argmax(action_marginals)
    elif action_selection == "stochastic":
        # Apply precision scaling and sample
        log_marginals = log_stable(action_marginals)
        p_actions = softmax(log_marginals * alpha)
        selected_action = np.random.choice(num_actions, p=p_actions)
    else:
        raise ValueError(f"Unknown action selection mode: {action_selection}")
    
    return int(selected_action)


def sample_policy(q_pi, policies, action_selection="deterministic", alpha=16.0):
    """
    Sample policy from policy posterior (full policy sampling).
    
    Samples complete policy and returns its first action.
    
    Args:
        q_pi: policy posterior probabilities (1D array)
        policies: list of policies
        action_selection: "deterministic" or "stochastic"
        alpha: precision parameter for policy selection
    
    Returns:
        selected action (int) - first action of selected policy
    """
    if action_selection == "deterministic":
        # Select most likely policy
        policy_idx = np.argmax(q_pi)
    elif action_selection == "stochastic":
        # Sample from policy posterior with precision scaling
        log_q_pi = log_stable(q_pi)
        p_policies = softmax(log_q_pi * alpha)
        policy_idx = np.random.choice(len(policies), p=p_policies)
    else:
        raise ValueError(f"Unknown action selection mode: {action_selection}")
    
    # Return first action of selected policy
    return int(policies[policy_idx][0])


# =============================================================================
# Numerical Stability Functions
# =============================================================================

def log_stable(x, eps=1e-16):
    """
    Numerically stable logarithm.
    
    Args:
        x: array-like input
        eps: small constant to prevent log(0)
    
    Returns:
        log(x + eps)
    """
    return np.log(np.maximum(x, eps))


def softmax(x, axis=None):
    """
    Numerically stable softmax function.
    
    Args:
        x: array-like input
        axis: axis along which to apply softmax
    
    Returns:
        softmax(x)
    """
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# State Belief Format Conversion
# =============================================================================

def qs_dict_to_list(qs_dict, factor_order=None):
    """
    Convert qs from dict format to list format.
    
    Args:
        qs_dict: dict mapping factor names to belief arrays
        factor_order: list of factor names defining order (optional)
    
    Returns:
        list of belief arrays in specified order
    
    Examples
    --------
    >>> qs_dict = {'agent_pos': np.array([0.5, 0.5]), 
    ...            'red_button_state': np.array([0.3, 0.7])}
    >>> qs_list = qs_dict_to_list(qs_dict, ['agent_pos', 'red_button_state'])
    """
    if factor_order is None:
        factor_order = sorted(qs_dict.keys())
    
    return [qs_dict[factor] for factor in factor_order]


def qs_list_to_dict(qs_list, factor_order):
    """
    Convert qs from list format to dict format.
    
    Args:
        qs_list: list of belief arrays
        factor_order: list of factor names defining order
    
    Returns:
        dict mapping factor names to belief arrays
    
    Examples
    --------
    >>> qs_list = [np.array([0.5, 0.5]), np.array([0.3, 0.7])]
    >>> factor_order = ['agent_pos', 'red_button_state']
    >>> qs_dict = qs_list_to_dict(qs_list, factor_order)
    """
    return {factor: qs_list[i] for i, factor in enumerate(factor_order)}


# =============================================================================
# Observation Formatting
# =============================================================================

def format_observation(obs, obs_labels):
    """
    Format observation from environment to model-compatible dict.
    
    This is a simple wrapper - for full conversion use env_utils.
    
    Args:
        obs: observation dict from environment
        obs_labels: dict mapping modality names to lists of observation labels
    
    Returns:
        formatted observation dict with indices
    """
    # This is a placeholder - actual conversion should use env_utils
    # from generative_models.SA_ActiveInference.RedBlueButton.env_utils
    return obs


def observation_to_one_hot(obs_idx, num_obs):
    """
    Convert observation index to one-hot vector.
    
    Args:
        obs_idx: observation index (int)
        num_obs: total number of possible observations
    
    Returns:
        one-hot array of length num_obs
    
    Examples
    --------
    >>> observation_to_one_hot(2, 5)
    array([0., 0., 1., 0., 0.])
    """
    one_hot = np.zeros(num_obs)
    one_hot[obs_idx] = 1.0
    return one_hot


def observations_to_one_hot(obs_dict, observation_labels):
    """
    Convert observation dict to dict of one-hot vectors.
    
    Args:
        obs_dict: dict mapping modality names to observation indices
        observation_labels: dict mapping modality names to lists of labels
    
    Returns:
        dict mapping modality names to one-hot arrays
    
    Examples
    --------
    >>> obs_dict = {'agent_pos': 3, 'button_state': 1}
    >>> obs_labels = {'agent_pos': list(range(9)), 'button_state': [0, 1]}
    >>> one_hot_obs = observations_to_one_hot(obs_dict, obs_labels)
    """
    one_hot_dict = {}
    for modality, obs_idx in obs_dict.items():
        num_obs = len(observation_labels[modality])
        one_hot_dict[modality] = observation_to_one_hot(obs_idx, num_obs)
    return one_hot_dict


# =============================================================================
# Helper: Sample from Categorical Distribution
# =============================================================================

def sample(probs):
    """
    Sample from categorical distribution.
    
    Args:
        probs: probability distribution (1D array summing to 1)
    
    Returns:
        sampled index (int)
    """
    return np.random.choice(len(probs), p=probs)
