"""
Utility functions for converting between environment observations/actions
and Active Inference model format.

The environment uses (x, y) coordinates and specific naming conventions,
while the Active Inference model uses flat indices and different observation keys.
"""

import numpy as np
from . import model_init


# =============================================================================
# Coordinate Conversion
# =============================================================================

def xy_to_index(x, y, width=3):
    """
    Convert (x, y) coordinates to flat grid index.
    
    Parameters
    ----------
    x : int
        Column position (0 to width-1)
    y : int
        Row position (0 to height-1)
    width : int, optional
        Grid width (default: 3)
    
    Returns
    -------
    index : int
        Flat index (0 to width*height - 1)
    
    Examples
    --------
    >>> xy_to_index(0, 0, width=3)  # Top-left
    0
    >>> xy_to_index(2, 0, width=3)  # Top-right
    2
    >>> xy_to_index(0, 2, width=3)  # Bottom-left
    6
    """
    return y * width + x


def index_to_xy(index, width=3):
    """
    Convert flat grid index to (x, y) coordinates.
    
    Parameters
    ----------
    index : int
        Flat index (0 to width*height - 1)
    width : int, optional
        Grid width (default: 3)
    
    Returns
    -------
    x : int
        Column position
    y : int
        Row position
    
    Examples
    --------
    >>> index_to_xy(0, width=3)  # Top-left
    (0, 0)
    >>> index_to_xy(2, width=3)  # Top-right
    (2, 0)
    >>> index_to_xy(6, width=3)  # Bottom-left
    (0, 2)
    """
    y = index // width
    x = index % width
    return x, y


# =============================================================================
# Observation Conversion: Environment → Model
# =============================================================================

def env_obs_to_model_obs(env_obs, width=3):
    """
    Convert environment observation to Active Inference model format.
    
    Parameters
    ----------
    env_obs : dict
        Environment observation with keys:
        - 'position': (x, y) array
        - 'on_red_button': int (0 or 1)
        - 'on_blue_button': int (0 or 1)
        - 'red_button_pressed': int (0 or 1)
        - 'blue_button_pressed': int (0 or 1)
        - 'win_lose_neutral': int (0=neutral, 1=win, 2=lose)
        - 'button_just_pressed': None, "red", or "blue"
    width : int, optional
        Grid width (default: 3)
    
    Returns
    -------
    model_obs : dict
        Model observation indices with keys:
        - 'agent_pos': int (0 to 8)
        - 'on_red_button': int (0=FALSE, 1=TRUE)
        - 'on_blue_button': int (0=FALSE, 1=TRUE)
        - 'red_button_state': int (0=not_pressed, 1=pressed)
        - 'blue_button_state': int (0=not_pressed, 1=pressed)
        - 'game_result': int (0=neutral, 1=win, 2=lose)
        - 'button_just_pressed': int (0=FALSE, 1=TRUE)
    
    Examples
    --------
    >>> env_obs = {
    ...     'position': np.array([0, 0]),
    ...     'on_red_button': 0,
    ...     'on_blue_button': 0,
    ...     'red_button_pressed': 0,
    ...     'blue_button_pressed': 0,
    ...     'win_lose_neutral': 0,
    ...     'button_just_pressed': None,
    ... }
    >>> model_obs = env_obs_to_model_obs(env_obs)
    >>> model_obs['agent_pos']
    0
    >>> model_obs['button_just_pressed']
    0
    """
    # Convert position from (x, y) to flat index
    x, y = env_obs['position']
    agent_pos_idx = xy_to_index(int(x), int(y), width)
    
    # Convert button_just_pressed from None/"red"/"blue" to 0/1
    if env_obs['button_just_pressed'] is None:
        button_just_pressed_idx = 0  # FALSE - no button just pressed
    else:
        button_just_pressed_idx = 1  # TRUE - a button was just pressed
    
    # Build model observation dict
    model_obs = {
        'agent_pos': agent_pos_idx,
        'on_red_button': int(env_obs['on_red_button']),
        'on_blue_button': int(env_obs['on_blue_button']),
        'red_button_state': int(env_obs['red_button_pressed']),
        'blue_button_state': int(env_obs['blue_button_pressed']),
        'game_result': int(env_obs['win_lose_neutral']),
        'button_just_pressed': button_just_pressed_idx,
    }
    
    return model_obs


def env_obs_to_model_obs_verbose(env_obs, width=3):
    """
    Convert environment observation to model format with verbose output.
    
    Same as env_obs_to_model_obs but prints conversion details.
    Useful for debugging.
    
    Parameters
    ----------
    env_obs : dict
        Environment observation
    width : int, optional
        Grid width (default: 3)
    
    Returns
    -------
    model_obs : dict
        Model observation indices
    """
    print("Converting Environment → Model Observation:")
    print("-" * 60)
    
    # Position conversion
    x, y = env_obs['position']
    agent_pos_idx = xy_to_index(int(x), int(y), width)
    print(f"  position: ({x}, {y}) → agent_pos: {agent_pos_idx}")
    
    # Button just pressed conversion
    if env_obs['button_just_pressed'] is None:
        button_just_pressed_idx = 0
        print(f"  button_just_pressed: None → 0 (FALSE)")
    else:
        button_just_pressed_idx = 1
        print(f"  button_just_pressed: '{env_obs['button_just_pressed']}' → 1 (TRUE)")
    
    # Direct mappings
    print(f"  on_red_button: {env_obs['on_red_button']}")
    print(f"  on_blue_button: {env_obs['on_blue_button']}")
    print(f"  red_button_pressed: {env_obs['red_button_pressed']} → red_button_state")
    print(f"  blue_button_pressed: {env_obs['blue_button_pressed']} → blue_button_state")
    print(f"  win_lose_neutral: {env_obs['win_lose_neutral']} → game_result")
    
    model_obs = env_obs_to_model_obs(env_obs, width)
    return model_obs


# =============================================================================
# Action Conversion: Model → Environment
# =============================================================================

def model_action_to_env_action(model_action):
    """
    Convert model action to environment action.
    
    Since the action spaces are identical, this is an identity mapping.
    Included for completeness and clarity.
    
    Parameters
    ----------
    model_action : int
        Model action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=OPEN, 5=NOOP)
    
    Returns
    -------
    env_action : int
        Environment action (0=up, 1=down, 2=left, 3=right, 4=press, 5=noop)
    
    Notes
    -----
    - Model action 4 (OPEN) = Environment action 4 (press)
    - All other actions map directly
    """
    return model_action  # Identity mapping


def env_action_to_model_action(env_action):
    """
    Convert environment action to model action.
    
    Identity mapping - included for completeness.
    
    Parameters
    ----------
    env_action : int
        Environment action
    
    Returns
    -------
    model_action : int
        Model action
    """
    return env_action  # Identity mapping


# =============================================================================
# State Conversion: Environment → Model
# =============================================================================

def env_state_to_model_state(env_state, width=3):
    """
    Convert environment full state to model state indices.
    
    Parameters
    ----------
    env_state : dict
        Environment state from env.get_state() with keys:
        - 'agent_position': (x, y)
        - 'red_button': (x, y)
        - 'blue_button': (x, y)
        - 'red_button_pressed': bool
        - 'blue_button_pressed': bool
        - 'step_count': int
        - 'max_steps': int
    width : int, optional
        Grid width (default: 3)
    
    Returns
    -------
    model_state : dict
        Model state indices with keys:
        - 'agent_pos': int
        - 'red_button_pos': int
        - 'blue_button_pos': int
        - 'red_button_state': int (0 or 1)
        - 'blue_button_state': int (0 or 1)
    """
    # Convert positions
    agent_x, agent_y = env_state['agent_position']
    red_x, red_y = env_state['red_button']
    blue_x, blue_y = env_state['blue_button']
    
    model_state = {
        'agent_pos': xy_to_index(agent_x, agent_y, width),
        'red_button_pos': xy_to_index(red_x, red_y, width),
        'blue_button_pos': xy_to_index(blue_x, blue_y, width),
        'red_button_state': 1 if env_state['red_button_pressed'] else 0,
        'blue_button_state': 1 if env_state['blue_button_pressed'] else 0,
    }
    
    return model_state


# =============================================================================
# Helper: Get environment-compatible D config
# =============================================================================

def get_D_config_from_env(env):
    """
    Extract D_fn configuration from environment to ensure alignment.
    
    Parameters
    ----------
    env : SingleAgentRedBlueButtonEnv
        Environment instance
    
    Returns
    -------
    config : dict
        Configuration for D_fn with keys:
        - agent_start_pos: int (flat index)
        - red_button_pos: int (flat index)
        - blue_button_pos: int (flat index)
        - red_button_pressed: bool
        - blue_button_pressed: bool
    
    Examples
    --------
    >>> from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
    >>> env = SingleAgentRedBlueButtonEnv()
    >>> config = get_D_config_from_env(env)
    >>> from generative_models.SA_ActiveInference.RedBlueButton.D import D_fn
    >>> D = D_fn(config)  # Now D matches env initial state
    """
    # Convert environment positions to model indices
    agent_x, agent_y = env.agent_start_pos
    red_x, red_y = env.red_button
    blue_x, blue_y = env.blue_button
    
    config = {
        'agent_start_pos': xy_to_index(agent_x, agent_y, env.width),
        'red_button_pos': xy_to_index(red_x, red_y, env.width),
        'blue_button_pos': xy_to_index(blue_x, blue_y, env.width),
        'red_button_pressed': env.red_button_pressed,
        'blue_button_pressed': env.blue_button_pressed,
    }
    
    return config
