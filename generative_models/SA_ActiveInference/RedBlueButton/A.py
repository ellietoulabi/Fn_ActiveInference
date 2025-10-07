"""
Functional A (observation model) for RedBlueButton environment.

Each observation modality is implemented as a pure function that takes SPECIFIC state
indices (not belief distributions) and returns p(observation | states).

This represents the environment's observation dynamics, independent of agent beliefs.
Belief propagation is handled separately in inference code.

IMPORTANT CONVENTIONS:
- Binary observations consistently use [FALSE, TRUE] ordering (index 0=FALSE, 1=TRUE)
- This applies to: on_red_button, on_blue_button, button_just_pressed
- Button states use [not_pressed, pressed] ordering
- Game results use [neutral, win, lose] ordering

DESIGN PRINCIPLE:
- A functions encode p(o | s) - the generative model / environment dynamics
- They take specific state indices, not belief distributions
- Belief propagation q(o) = Σ_s q(s)p(o|s) is done in inference utilities
"""

import jax.numpy as jnp
import numpy as np
from . import model_init

# Noise parameter for self-position observations
A_NOISE_LEVEL = 0.01


# =============================================================================
# Core A functions: p(observation | specific states)
# =============================================================================

def A_agent_pos(agent_pos_idx, num_obs, noise_level=A_NOISE_LEVEL):
    """
    Observation likelihood for agent position.
    
    Dependencies: agent_pos
    
    Parameters
    ----------
    agent_pos_idx : int
        Specific agent position (0 to S-1)
    num_obs : int
        Number of possible observations (same as S)
    noise_level : float
        Observation noise level
    
    Returns
    -------
    p_obs : array, shape (num_obs,)
        p(observe position o | agent at agent_pos_idx)
        Mostly observes correct position with small uniform noise
    """
    pos_obs = np.full(num_obs, noise_level / (num_obs - 1))
    pos_obs[agent_pos_idx] = 1.0 - noise_level
    return pos_obs


def A_on_red_button(agent_pos_idx, red_button_pos_idx):
    """
    Observation likelihood for being on red button.
    
    Dependencies: agent_pos, red_button_pos
    
    Parameters
    ----------
    agent_pos_idx : int
        Specific agent position
    red_button_pos_idx : int
        Specific red button position
    
    Returns
    -------
    p_obs : array, shape (2,)
        [p(FALSE), p(TRUE)] for whether agent is on red button
    """
    if agent_pos_idx == red_button_pos_idx:
        return np.array([0.0, 1.0])  # TRUE - on button
    else:
        return np.array([1.0, 0.0])  # FALSE - not on button


def A_on_blue_button(agent_pos_idx, blue_button_pos_idx):
    """
    Observation likelihood for being on blue button.
    
    Dependencies: agent_pos, blue_button_pos
    
    Parameters
    ----------
    agent_pos_idx : int
        Specific agent position
    blue_button_pos_idx : int
        Specific blue button position
    
    Returns
    -------
    p_obs : array, shape (2,)
        [p(FALSE), p(TRUE)] for whether agent is on blue button
    """
    if agent_pos_idx == blue_button_pos_idx:
        return np.array([0.0, 1.0])  # TRUE - on button
    else:
        return np.array([1.0, 0.0])  # FALSE - not on button


def A_red_button_state(red_button_state_idx):
    """
    Observation likelihood for red button state.
    
    Dependencies: red_button_state
    
    Parameters
    ----------
    red_button_state_idx : int
        Specific red button state (0=not_pressed, 1=pressed)
    
    Returns
    -------
    p_obs : array, shape (2,)
        [p(not_pressed), p(pressed)]
        Deterministic observation of button state
    """
    obs = np.zeros(2)
    obs[red_button_state_idx] = 1.0
    return obs


def A_blue_button_state(blue_button_state_idx):
    """
    Observation likelihood for blue button state.
    
    Dependencies: blue_button_state
    
    Parameters
    ----------
    blue_button_state_idx : int
        Specific blue button state (0=not_pressed, 1=pressed)
    
    Returns
    -------
    p_obs : array, shape (2,)
        [p(not_pressed), p(pressed)]
        Deterministic observation of button state
    """
    obs = np.zeros(2)
    obs[blue_button_state_idx] = 1.0
    return obs


def A_game_result(red_button_state_idx, blue_button_state_idx):
    """
    Observation likelihood for game result.
    
    Dependencies: red_button_state, blue_button_state
    
    Parameters
    ----------
    red_button_state_idx : int
        Specific red button state (0=not_pressed, 1=pressed)
    blue_button_state_idx : int
        Specific blue button state (0=not_pressed, 1=pressed)
    
    Returns
    -------
    p_obs : array, shape (3,)
        [p(neutral), p(win), p(lose)]
        - neutral: not both pressed
        - win: both pressed (correct order)
        - lose: only blue pressed (wrong order)
    """
    both_pressed = (red_button_state_idx == 1 and blue_button_state_idx == 1)
    only_blue = (red_button_state_idx == 0 and blue_button_state_idx == 1)
    
    if both_pressed:
        return np.array([0.0, 1.0, 0.0])  # win
    elif only_blue:
        return np.array([0.0, 0.0, 1.0])  # lose
    else:
        return np.array([1.0, 0.0, 0.0])  # neutral


def A_button_just_pressed(agent_pos_idx, red_button_pos_idx, blue_button_pos_idx,
                          red_button_state_idx, blue_button_state_idx,
                          prev_red_button_state_idx=None, prev_blue_button_state_idx=None):
    """
    Observation likelihood for button just pressed.
    
    Dependencies: agent_pos, red_button_pos, blue_button_pos, 
                  red_button_state, blue_button_state
                  (and optionally previous button states for transition detection)
    
    Parameters
    ----------
    agent_pos_idx : int
        Current agent positiongit 
    red_button_pos_idx : int
        Red button position
    blue_button_pos_idx : int
        Blue button position
    red_button_state_idx : int
        Current red button state
    blue_button_state_idx : int
        Current blue button state
    prev_red_button_state_idx : int, optional
        Previous red button state (for detecting transitions)
    prev_blue_button_state_idx : int, optional
        Previous blue button state (for detecting transitions)
    
    Returns
    -------
    p_obs : array, shape (2,)
        [p(FALSE), p(TRUE)] for whether a button was just pressed
    """
    if prev_red_button_state_idx is None or prev_blue_button_state_idx is None:
        # Without previous state, use current position as proxy
        at_red = (agent_pos_idx == red_button_pos_idx)
        at_blue = (agent_pos_idx == blue_button_pos_idx)
        at_button = at_red or at_blue
        
        if at_button:
            return np.array([0.0, 1.0])  # TRUE - might have just pressed
        else:
            return np.array([1.0, 0.0])  # FALSE - not at button
    
    # With previous state, detect transitions (0 → 1)
    red_just_pressed = (prev_red_button_state_idx == 0 and red_button_state_idx == 1)
    blue_just_pressed = (prev_blue_button_state_idx == 0 and blue_button_state_idx == 1)
    
    if red_just_pressed or blue_just_pressed:
        return np.array([0.0, 1.0])  # TRUE - button just pressed
    else:
        return np.array([1.0, 0.0])  # FALSE - no button just pressed


# =============================================================================
# A function registry - maps modality names to functions
# =============================================================================

A_FUNCTIONS = {
    "agent_pos": A_agent_pos,
    "on_red_button": A_on_red_button,
    "on_blue_button": A_on_blue_button,
    "red_button_state": A_red_button_state,
    "blue_button_state": A_blue_button_state,
    "game_result": A_game_result,
    "button_just_pressed": A_button_just_pressed,
}


# =============================================================================
# A_fn: Main interface (analogous to B_fn)
# =============================================================================

def A_fn(state_indices, prev_state_indices=None, modalities=None):
    """
    Get observation likelihoods for specified modalities given specific state configuration.
    
    This is the main A function analogous to B_fn for transitions.
    Takes specific state indices and returns observation likelihoods.
    
    Parameters
    ----------
    state_indices : dict
        Dictionary mapping state factor names to specific indices.
        Example: {
            "agent_pos": 4,
            "red_button_pos": 2,
            "blue_button_pos": 6,
            "red_button_state": 0,
            "blue_button_state": 0,
        }
    prev_state_indices : dict, optional
        Previous state indices (needed for button_just_pressed modality)
    modalities : list of str, optional
        List of modality names to compute. If None, compute all modalities.
        This allows selective computation for efficiency.
    
    Returns
    -------
    obs_likelihoods : dict
        Dictionary mapping modality names to observation likelihood distributions.
        Each value is p(o | state configuration) for that modality.
    """
    obs_likelihoods = {}
    
    # Determine which modalities to compute
    if modalities is None:
        modalities_to_compute = model_init.observation_state_dependencies.keys()
    else:
        modalities_to_compute = modalities
    
    for modality in modalities_to_compute:
        deps = model_init.observation_state_dependencies[modality]
        A_func = A_FUNCTIONS[modality]
        
        # Extract relevant state indices
        args = [state_indices[dep] for dep in deps]
        
        # Handle special cases
        if modality == "agent_pos":
            # Need to pass num_obs parameter
            num_obs = len(model_init.observations["agent_pos"])
            obs_likelihoods[modality] = A_func(args[0], num_obs)
        
        elif modality == "button_just_pressed" and prev_state_indices is not None:
            # Need previous state for transition detection
            prev_args = [prev_state_indices[dep] for dep in deps]
            # Current states
            curr_agent_pos = state_indices["agent_pos"]
            curr_red_pos = state_indices["red_button_pos"]
            curr_blue_pos = state_indices["blue_button_pos"]
            curr_red_state = state_indices["red_button_state"]
            curr_blue_state = state_indices["blue_button_state"]
            # Previous button states
            prev_red_state = prev_state_indices["red_button_state"]
            prev_blue_state = prev_state_indices["blue_button_state"]
            
            obs_likelihoods[modality] = A_func(
                curr_agent_pos, curr_red_pos, curr_blue_pos,
                curr_red_state, curr_blue_state,
                prev_red_state, prev_blue_state
            )
        
        else:
            # Standard case: just pass the state indices
            obs_likelihoods[modality] = A_func(*args)
    
    return obs_likelihoods





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

