
import numpy as np
from . import model_init



# def build_D(agent_start_pos=0, red_button_pos=2, blue_button_pos=6,
#             red_button_pressed=False, blue_button_pressed=False):
#     """
#     Build prior belief distributions D for the initial state.
    
#     Parameters
#     ----------
#     agent_start_pos : int, optional
#         Initial agent position (default: 0 = top-left corner)
#     red_button_pos : int, optional
#         Red button position (default: 2)
#     blue_button_pos : int, optional
#         Blue button position (default: 6)
#     red_button_pressed : bool, optional
#         Whether red button starts pressed (default: False)
#     blue_button_pressed : bool, optional
#         Whether blue button starts pressed (default: False)
    
#     Returns
#     -------
#     D : dict
#         Dictionary mapping state factor names to prior belief distributions.
#     """
#     S = model_init.S  # Grid size (9 for 3x3)
    
#     D = {}
    
#     # Agent position: certain at start position
#     D["agent_pos"] = np.zeros(S)
#     D["agent_pos"][agent_start_pos] = 1.0
    
#     # Red button position: certain
#     D["red_button_pos"] = np.zeros(S)
#     D["red_button_pos"][red_button_pos] = 1.0
    
#     # Blue button position: certain
#     D["blue_button_pos"] = np.zeros(S)
#     D["blue_button_pos"][blue_button_pos] = 1.0
    
#     # Red button state: certain (0=not pressed, 1=pressed)
#     D["red_button_state"] = np.zeros(2)
#     D["red_button_state"][1 if red_button_pressed else 0] = 1.0
    
#     # Blue button state: certain (0=not pressed, 1=pressed)
#     D["blue_button_state"] = np.zeros(2)
#     D["blue_button_state"][1 if blue_button_pressed else 0] = 1.0
    
#     return D


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
   
    if config is None:
        # Default configuration: agent at 0, button positions uncertain (uniform), 
        # button states certain (not pressed)
        return build_D_uncertain(
            agent_start_pos=0,
            button_pos_uncertainty=True,
            button_state_uncertainty=False
        )
    else:
        # Build custom D from config
        return build_D_uncertain(
            agent_start_pos=config.get('agent_start_pos', 0),
            button_pos_uncertainty=config.get('button_pos_uncertainty', True),
            button_state_uncertainty=config.get('button_state_uncertainty', False)
        )
