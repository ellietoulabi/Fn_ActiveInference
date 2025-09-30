import numpy as np
import itertools

def construct_policies(actions, policy_len):
    """
    Construct all possible policies of given length.
    
    Args:
        actions: list of available actions
        policy_len: length of each policy
    
    Returns:
        list of policies, where each policy is a list of actions
    """
    if policy_len == 1:
        return [[action] for action in actions]
    else:
        # Generate all combinations with replacement
        policies = list(itertools.product(actions, repeat=policy_len))
        return [list(policy) for policy in policies]

def sample_action(q_pi, policies, action_selection="deterministic", alpha=16.0, actions=None):
    """
    Sample action from policy posterior (marginal sampling).
    
    Args:
        q_pi: policy posterior probabilities
        policies: list of policies
        action_selection: "deterministic" or "stochastic"
        alpha: precision parameter for action selection
        actions: list of available actions
    
    Returns:
        selected action (int)
    """
    if action_selection == "deterministic":
        # Find most likely policy and return its first action
        best_policy_idx = np.argmax(q_pi)
        return policies[best_policy_idx][0]
    elif action_selection == "stochastic":
        # Sample from policy posterior and return first action
        policy_idx = np.random.choice(len(policies), p=q_pi)
        return policies[policy_idx][0]
    else:
        raise ValueError(f"Unknown action selection mode: {action_selection}")

def sample_policy(q_pi, policies, action_selection="deterministic", alpha=16.0):
    """
    Sample policy from policy posterior (full policy sampling).
    
    Args:
        q_pi: policy posterior probabilities
        policies: list of policies
        action_selection: "deterministic" or "stochastic"
        alpha: precision parameter for policy selection
    
    Returns:
        selected action (int) - first action of selected policy
    """
    if action_selection == "deterministic":
        # Find most likely policy and return its first action
        best_policy_idx = np.argmax(q_pi)
        return policies[best_policy_idx][0]
    elif action_selection == "stochastic":
        # Sample from policy posterior and return first action
        policy_idx = np.random.choice(len(policies), p=q_pi)
        return policies[policy_idx][0]
    else:
        raise ValueError(f"Unknown action selection mode: {action_selection}")

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

