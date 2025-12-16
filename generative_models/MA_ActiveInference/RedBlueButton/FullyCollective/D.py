"""
Prior beliefs (D) for FullyCollective paradigm (JOINT state).
"""

import numpy as np
from . import model_init


def build_D(
    agent1_start_pos=0,
    agent2_start_pos=8,
    button_pos_uncertainty=True,
    button_state_uncertainty=False,
):
    S = model_init.S
    D = {}

    D["agent1_pos"] = np.zeros(S)
    D["agent1_pos"][int(agent1_start_pos)] = 1.0

    D["agent2_pos"] = np.zeros(S)
    D["agent2_pos"][int(agent2_start_pos)] = 1.0

    if button_pos_uncertainty:
        base = np.ones(S, dtype=float)
        for idx in (int(agent1_start_pos), int(agent2_start_pos)):
            if 0 <= idx < S:
                base[idx] = 0.0
        if float(base.sum()) <= 0:
            base = np.ones(S, dtype=float)
        base = base / base.sum()
        D["red_button_pos"] = base.copy()
        D["blue_button_pos"] = base.copy()
    else:
        D["red_button_pos"] = np.zeros(S)
        D["red_button_pos"][6] = 1.0
        D["blue_button_pos"] = np.zeros(S)
        D["blue_button_pos"][2] = 1.0

    if button_state_uncertainty:
        D["red_button_state"] = np.ones(2) / 2
        D["blue_button_state"] = np.ones(2) / 2
    else:
        D["red_button_state"] = np.array([1.0, 0.0])
        D["blue_button_state"] = np.array([1.0, 0.0])

    return D


def D_fn(config=None):
    if config is None:
        return build_D()
    return build_D(
        agent1_start_pos=config.get("agent1_start_pos", 0),
        agent2_start_pos=config.get("agent2_start_pos", 8),
        button_pos_uncertainty=config.get("button_pos_uncertainty", True),
        button_state_uncertainty=config.get("button_state_uncertainty", False),
    )


