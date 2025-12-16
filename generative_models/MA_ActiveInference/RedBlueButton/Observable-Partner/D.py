"""
Prior Beliefs (D) for Multi-Agent RedBlueButton.

D_fn returns prior belief distributions over hidden states.
"""

import numpy as np
from . import model_init


def build_D(my_start_pos=0, other_start_pos=8, 
            button_pos_uncertainty=True,
            button_state_uncertainty=False):
    """
    Build prior belief distributions.
    
    Parameters
    ----------
    my_start_pos : int
        My certain start position
    other_start_pos : int
        Other agent's start position (observed)
    button_pos_uncertainty : bool
        If True, uncertain about button positions
    button_state_uncertainty : bool
        If True, uncertain about initial button states
    
    Returns
    -------
    D : dict
        Prior belief distributions
    """
    S = model_init.S
    D = {}
    
    # My position: certain at start
    D['my_pos'] = np.zeros(S)
    D['my_pos'][my_start_pos] = 1.0
    
    # Other agent's position: observed, so certain
    D['other_pos'] = np.zeros(S)
    D['other_pos'][other_start_pos] = 1.0
    
    # Button positions
    if button_pos_uncertainty:
        # Uncertain - factorized uniform, but avoid trivially impossible starts
        # (buttons cannot be under agents at t=0 in our env configs).
        base = np.ones(S, dtype=float)
        for idx in (my_start_pos, other_start_pos):
            if 0 <= idx < S:
                base[idx] = 0.0
        if float(base.sum()) <= 0:
            base = np.ones(S, dtype=float)
        base = base / base.sum()
        D['red_button_pos'] = base.copy()
        D['blue_button_pos'] = base.copy()
    else:
        # Default positions
        D['red_button_pos'] = np.zeros(S)
        D['red_button_pos'][6] = 1.0  # Default red at (0,2)
        D['blue_button_pos'] = np.zeros(S)
        D['blue_button_pos'][2] = 1.0  # Default blue at (2,0)
    
    # Button states
    if button_state_uncertainty:
        D['red_button_state'] = np.ones(2) / 2
        D['blue_button_state'] = np.ones(2) / 2
    else:
        # Certain: not pressed
        D['red_button_state'] = np.array([1.0, 0.0])
        D['blue_button_state'] = np.array([1.0, 0.0])
    
    return D


def D_fn(config=None):
    """
    Main interface for getting prior beliefs.
    
    Parameters
    ----------
    config : dict, optional
        Configuration with keys:
        - 'my_start_pos': int
        - 'other_start_pos': int
        - 'button_pos_uncertainty': bool
        - 'button_state_uncertainty': bool
    
    Returns
    -------
    D : dict
        Prior belief distributions
    """
    if config is None:
        return build_D(
            my_start_pos=0,
            other_start_pos=8,
            button_pos_uncertainty=True,
            button_state_uncertainty=False
        )
    else:
        return build_D(
            my_start_pos=config.get('my_start_pos', 0),
            other_start_pos=config.get('other_start_pos', 8),
            button_pos_uncertainty=config.get('button_pos_uncertainty', True),
            button_state_uncertainty=config.get('button_state_uncertainty', False)
        )
