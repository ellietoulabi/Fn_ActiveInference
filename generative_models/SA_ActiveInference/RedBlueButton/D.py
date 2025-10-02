"""
Functional D (prior beliefs) for RedBlueButton environment.

D encodes the agent's initial beliefs about the state of the world
at the start of an episode.

DESIGN PRINCIPLE:
- D represents p(s_0) - the prior belief over initial states
- Each state factor has a distribution (can be certain or uncertain)
- This is the agent's belief, not necessarily the true initial state
"""

import numpy as np
import jax.numpy as jnp
from . import model_init


# =============================================================================
# D: Prior belief distributions for each state factor
# =============================================================================

def build_D(agent_start_pos=0, red_button_pos=2, blue_button_pos=6,
            red_button_pressed=False, blue_button_pressed=False):
    """
    Build prior belief distributions D for the initial state.
    
    Parameters
    ----------
    agent_start_pos : int, optional
        Initial agent position (default: 0 = top-left corner)
    red_button_pos : int, optional
        Red button position (default: 2)
    blue_button_pos : int, optional
        Blue button position (default: 6)
    red_button_pressed : bool, optional
        Whether red button starts pressed (default: False)
    blue_button_pressed : bool, optional
        Whether blue button starts pressed (default: False)
    
    Returns
    -------
    D : dict
        Dictionary mapping state factor names to prior belief distributions.
    """
    S = model_init.S  # Grid size (9 for 3x3)
    
    D = {}
    
    # Agent position: certain at start position
    D["agent_pos"] = np.zeros(S)
    D["agent_pos"][agent_start_pos] = 1.0
    
    # Red button position: certain
    D["red_button_pos"] = np.zeros(S)
    D["red_button_pos"][red_button_pos] = 1.0
    
    # Blue button position: certain
    D["blue_button_pos"] = np.zeros(S)
    D["blue_button_pos"][blue_button_pos] = 1.0
    
    # Red button state: certain (0=not pressed, 1=pressed)
    D["red_button_state"] = np.zeros(2)
    D["red_button_state"][1 if red_button_pressed else 0] = 1.0
    
    # Blue button state: certain (0=not pressed, 1=pressed)
    D["blue_button_state"] = np.zeros(2)
    D["blue_button_state"][1 if blue_button_pressed else 0] = 1.0
    
    return D


def build_D_uncertain(agent_start_pos=0, 
                      button_pos_uncertainty=False,
                      button_state_uncertainty=False):
    """
    Build prior with optional uncertainty.
    
    Parameters
    ----------
    agent_start_pos : int
        Agent's certain start position
    button_pos_uncertainty : bool
        If True, agent is uncertain about button positions (uniform)
    button_state_uncertainty : bool
        If True, agent is uncertain about initial button states (uniform)
    
    Returns
    -------
    D : dict
        Prior belief distributions
    """
    S = model_init.S
    
    D = {}
    
    # Agent position: always certain at start
    D["agent_pos"] = np.zeros(S)
    D["agent_pos"][agent_start_pos] = 1.0
    
    # Button positions
    if button_pos_uncertainty:
        # Uncertain - uniform over all positions
        D["red_button_pos"] = np.ones(S) / S
        D["blue_button_pos"] = np.ones(S) / S
    else:
        # Certain - default positions
        D["red_button_pos"] = np.zeros(S)
        D["red_button_pos"][2] = 1.0
        D["blue_button_pos"] = np.zeros(S)
        D["blue_button_pos"][6] = 1.0
    
    # Button states
    if button_state_uncertainty:
        # Uncertain - uniform over pressed/not pressed
        D["red_button_state"] = np.ones(2) / 2
        D["blue_button_state"] = np.ones(2) / 2
    else:
        # Certain - both not pressed
        D["red_button_state"] = np.array([1.0, 0.0])
        D["blue_button_state"] = np.array([1.0, 0.0])
    
    return D


# =============================================================================
# D_fn: Main interface (analogous to A_fn, B_fn, C_fn)
# =============================================================================

def D_fn(config=None):
    """
    Get prior belief distributions D.
    
    This is the main D function analogous to A_fn, B_fn, and C_fn.
    Returns initial beliefs over all state factors.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dict with keys:
        - agent_start_pos: int (default: 0)
        - red_button_pos: int (default: 2)
        - blue_button_pos: int (default: 6)
        - red_button_pressed: bool (default: False)
        - blue_button_pressed: bool (default: False)
        If None, uses default configuration:
          Agent at 0, red at 2, blue at 6, both not pressed
    
    Returns
    -------
    D : dict
        Prior belief distributions for each state factor.
        Each value is a probability distribution over that factor's states.
    
    Examples
    --------
    >>> # Default configuration
    >>> D = D_fn()
    >>> 
    >>> # Custom configuration
    >>> D = D_fn({'agent_start_pos': 4, 'red_button_pos': 0})
    """
    if config is None:
        # Default configuration: 3x3 grid, agent at 0, red at 2, blue at 6, both not pressed
        return build_D(
            agent_start_pos=0,
            red_button_pos=2,
            blue_button_pos=6,
            red_button_pressed=False,
            blue_button_pressed=False
        )
    else:
        # Build custom D from config
        return build_D(
            agent_start_pos=config.get('agent_start_pos', 0),
            red_button_pos=config.get('red_button_pos', 2),
            blue_button_pos=config.get('blue_button_pos', 6),
            red_button_pressed=config.get('red_button_pressed', False),
            blue_button_pressed=config.get('blue_button_pressed', False)
        )
