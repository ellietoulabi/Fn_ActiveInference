"""
Observation model (A) for FullyCollective paradigm (JOINT state).
"""

import numpy as np
from . import model_init

# Keep small observation noise for positions (SA-like)
A_NOISE_LEVEL = 0.01


def _noisy_pos_obs(pos_idx, num_obs, noise_level=A_NOISE_LEVEL):
    p = np.full(num_obs, noise_level / max(1, (num_obs - 1)), dtype=float)
    p[pos_idx] = 1.0 - noise_level
    return p


def A_fn(state_indices):
    """
    Compute observation likelihoods for the full joint observation.
    """
    a1 = int(state_indices["agent1_pos"])
    a2 = int(state_indices["agent2_pos"])
    red_pos = int(state_indices["red_button_pos"])
    blue_pos = int(state_indices["blue_button_pos"])
    red_state = int(state_indices["red_button_state"])
    blue_state = int(state_indices["blue_button_state"])

    S = model_init.S
    obs = {}

    obs["agent1_pos"] = _noisy_pos_obs(a1, S)
    obs["agent2_pos"] = _noisy_pos_obs(a2, S)

    obs["agent1_on_red_button"] = np.array([0.0, 1.0]) if a1 == red_pos else np.array([1.0, 0.0])
    obs["agent1_on_blue_button"] = np.array([0.0, 1.0]) if a1 == blue_pos else np.array([1.0, 0.0])
    obs["agent2_on_red_button"] = np.array([0.0, 1.0]) if a2 == red_pos else np.array([1.0, 0.0])
    obs["agent2_on_blue_button"] = np.array([0.0, 1.0]) if a2 == blue_pos else np.array([1.0, 0.0])

    obs["red_button_state"] = np.zeros(2)
    obs["red_button_state"][red_state] = 1.0
    obs["blue_button_state"] = np.zeros(2)
    obs["blue_button_state"][blue_state] = 1.0

    obs["game_result"] = np.zeros(3)
    if blue_state == 1:
        obs["game_result"][1 if red_state == 1 else 2] = 1.0
    else:
        obs["game_result"][0] = 1.0

    # Dynamic observation; approximate using "on any button" heuristic (SA-like)
    on_any_button = (a1 == red_pos) or (a1 == blue_pos) or (a2 == red_pos) or (a2 == blue_pos)
    obs["button_just_pressed"] = np.array([0.5, 0.5]) if on_any_button else np.array([1.0, 0.0])

    return obs





# =============================================================================
# Inference utility: Apply A to belief distributions
# =============================================================================

def get_observation_likelihood(modality, state_indices):
    """
    Get observation likelihood for a specific state configuration.
    
    Parameters
    ----------
    modality : str
        Observation modality name (e.g., "on_red_button")
    state_indices : dict
        Dictionary mapping state factor names to specific indices
        Example: {"agent_pos": 4, "red_button_pos": 2}
    
    Returns
    -------
    p_obs : array
        p(observation | states) for this modality
    """
    A_func = A_FUNCTIONS[modality]
    deps = model_init.observation_state_dependencies[modality]
    
    # Extract relevant state indices based on dependencies
    args = [state_indices[dep] for dep in deps]
    
    # Special cases for modalities with extra parameters
    if modality == "agent_pos":
        num_obs = len(model_init.states["agent_pos"])
        return A_func(args[0], num_obs)
    else:
        return A_func(*args)


def predict_obs_from_beliefs(modality, state_beliefs, prev_state_beliefs=None):
    """
    Compute predicted observation distribution from uncertain state beliefs.
    
    This implements: q(o) = Σ_s q(s) p(o|s)
    
    Parameters
    ----------
    modality : str
        Observation modality name
    state_beliefs : dict
        Dictionary mapping state factor names to belief distributions
        Example: {"agent_pos": array([0.1, 0.2, ...]), ...}
    prev_state_beliefs : dict, optional
        Previous state beliefs (for button_just_pressed modality)
    
    Returns
    -------
    q_obs : array
        Predicted observation distribution q(o)
    """
    deps = model_init.observation_state_dependencies[modality]
    A_func = A_FUNCTIONS[modality]
    
    # Get shapes of dependent factors
    factor_sizes = {dep: len(state_beliefs[dep]) for dep in deps}
    
    # Determine output size from observations dict
    num_obs = len(model_init.observations[modality])
    q_obs = np.zeros(num_obs)
    
    # Special handling for agent_pos (has noise parameter)
    if modality == "agent_pos":
        S = factor_sizes["agent_pos"]
        for pos in range(S):
            if state_beliefs["agent_pos"][pos] > 1e-10:
                p_obs = A_func(pos, num_obs)
                q_obs += state_beliefs["agent_pos"][pos] * p_obs
        return q_obs / np.sum(q_obs)  # Normalize
    
    # For other modalities, iterate over all combinations of dependent factors
    import itertools
    
    # Create ranges for each dependent factor
    factor_ranges = [range(factor_sizes[dep]) for dep in deps]
    
    # Iterate over all combinations
    for state_combo in itertools.product(*factor_ranges):
        # Create state_indices dict
        state_indices = {dep: idx for dep, idx in zip(deps, state_combo)}
        
        # Compute joint probability under factorized beliefs
        joint_prob = 1.0
        for dep, idx in state_indices.items():
            joint_prob *= state_beliefs[dep][idx]
        
        # Skip negligible probabilities
        if joint_prob < 1e-10:
            continue
        
        # Get observation likelihood for this state combination
        p_obs = get_observation_likelihood(modality, state_indices)
        
        # Add weighted contribution
        q_obs += joint_prob * p_obs
    
    return q_obs / np.sum(q_obs)  # Normalize


def predict_all_obs_from_beliefs(state_beliefs, prev_state_beliefs=None):
    """
    Predict observation distributions for ALL modalities from state beliefs.
    
    Parameters
    ----------
    state_beliefs : dict
        Belief distributions over all state factors
    prev_state_beliefs : dict, optional
        Previous state beliefs
    
    Returns
    -------
    obs_predictions : dict
        Dictionary mapping modality names to predicted observation distributions
    """
    obs_predictions = {}
    
    for modality in model_init.observation_state_dependencies.keys():
        obs_predictions[modality] = predict_obs_from_beliefs(
            modality, state_beliefs, prev_state_beliefs
        )
    
    return obs_predictions

