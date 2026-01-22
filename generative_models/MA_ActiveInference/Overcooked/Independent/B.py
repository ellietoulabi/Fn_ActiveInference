"""
Transition model (B) for Independent paradigm (single-agent actions) - Cramped Room.
"""

import numpy as np
from . import model_init


def normalize(p):
    """Normalize probability distribution."""
    return p / np.maximum(np.sum(p), 1e-8)


def _compute_new_pos(pos_idx, action, width=model_init.GRID_WIDTH, height=model_init.GRID_HEIGHT):
    """Compute new position index after action."""
    if action == model_init.STAY or action == model_init.INTERACT:
        return pos_idx
    
    x, y = model_init.index_to_xy(pos_idx, width)
    
    if action == model_init.NORTH:
        y = max(0, y - 1)
    elif action == model_init.SOUTH:
        y = min(height - 1, y + 1)
    elif action == model_init.EAST:
        x = min(width - 1, x + 1)
    elif action == model_init.WEST:
        x = max(0, x - 1)
    
    return model_init.xy_to_index(x, y, width)


def _update_orientation(ori_idx, action):
    """Update orientation based on action."""
    if action in (model_init.NORTH, model_init.SOUTH, model_init.EAST, model_init.WEST):
        return action
    return ori_idx


def B_agent_pos(qs, action):
    """Update agent position given action."""
    q_pos = np.array(qs["agent_pos"], dtype=float)
    q_other = np.array(qs["other_agent_pos"], dtype=float)
    S = model_init.GRID_SIZE

    next_q = np.zeros(S, dtype=float)

    for pos in range(S):
        if q_pos[pos] <= 1e-16:
            continue
        for other_pos in range(S):
            if q_other[other_pos] <= 1e-16:
                continue
            if pos == other_pos:
                continue
            
            w = q_pos[pos] * q_other[other_pos]
            if w <= 1e-16:
                continue
            
            new_pos = _compute_new_pos(pos, action)
            if new_pos == other_pos:
                new_pos = pos
            
            if new_pos < S and new_pos != other_pos:
                next_q[new_pos] += w

    return normalize(next_q)


def B_agent_orientation(qs, action):
    """Update orientation based on action."""
    q_ori = np.array(qs["agent_orientation"], dtype=float)
    next_q = np.zeros(model_init.N_DIRECTIONS, dtype=float)
    
    for ori_idx in range(model_init.N_DIRECTIONS):
        if q_ori[ori_idx] <= 1e-16:
            continue
        new_ori = _update_orientation(ori_idx, action)
        if 0 <= new_ori < model_init.N_DIRECTIONS:
            next_q[new_ori] += q_ori[ori_idx]
    
    return normalize(next_q)


def B_agent_held(qs, action):
    """Update held objects based on interactions."""
    q_held = np.array(qs["agent_held"], dtype=float)
    q_pos = np.array(qs["agent_pos"], dtype=float)
    q_pot_state = np.array(qs["pot_state"], dtype=float)
    
    next_q = np.zeros(model_init.N_HELD_TYPES, dtype=float)
    
    for pos_idx in range(model_init.GRID_SIZE):
        if q_pos[pos_idx] <= 1e-16:
            continue
        for held_idx in range(model_init.N_HELD_TYPES):
            if q_held[held_idx] <= 1e-16:
                continue
            for pot_state_idx in range(model_init.N_POT_STATES):
                if q_pot_state[pot_state_idx] <= 1e-16:
                    continue
                
                w = q_pos[pos_idx] * q_held[held_idx] * q_pot_state[pot_state_idx]
                if w <= 1e-16:
                    continue
                
                new_held = held_idx
                
                if action == model_init.INTERACT:
                    if model_init.is_at_onion_dispenser(pos_idx) and held_idx == model_init.HELD_NONE:
                        new_held = model_init.HELD_ONION
                    elif model_init.is_at_pot(pos_idx) and held_idx == model_init.HELD_ONION and pot_state_idx == model_init.POT_IDLE:
                        new_held = model_init.HELD_NONE
                    elif model_init.is_at_pot(pos_idx) and held_idx == model_init.HELD_NONE and pot_state_idx == model_init.POT_READY:
                        new_held = model_init.HELD_SOUP
                    elif model_init.is_at_serving(pos_idx) and held_idx == model_init.HELD_SOUP:
                        new_held = model_init.HELD_NONE
                
                if 0 <= new_held < model_init.N_HELD_TYPES:
                    next_q[new_held] += w
    
    return normalize(next_q)


def B_other_agent_pos(qs):
    """Other agent position (treated as part of environment, simplified)."""
    q_other = np.array(qs["other_agent_pos"], dtype=float)
    # Simplified: other agent moves independently (we don't model their actions)
    noise = 0.1
    next_q = q_other * (1.0 - noise)
    S = len(next_q)
    if S > 1:
        next_q += noise / S
    return normalize(next_q)


def B_pot_state(qs, action):
    """Update pot state based on agent interactions."""
    q_pot_state = np.array(qs["pot_state"], dtype=float)
    q_pos = np.array(qs["agent_pos"], dtype=float)
    q_held = np.array(qs["agent_held"], dtype=float)
    
    next_q = np.zeros(model_init.N_POT_STATES, dtype=float)
    
    p_at_pot = 0.0
    p_has_onion = 0.0
    p_has_nothing = 0.0
    
    for pos_idx in range(model_init.GRID_SIZE):
        if model_init.is_at_pot(pos_idx):
            p_at_pot += q_pos[pos_idx]
    
    p_has_onion = q_held[model_init.HELD_ONION]
    p_has_nothing = q_held[model_init.HELD_NONE]
    
    for pot_state_idx in range(model_init.N_POT_STATES):
        if q_pot_state[pot_state_idx] <= 1e-16:
            continue
        
        w = q_pot_state[pot_state_idx]
        
        if pot_state_idx == model_init.POT_IDLE:
            if action == model_init.INTERACT and p_at_pot * p_has_onion > 0.5:
                next_q[model_init.POT_COOKING] += w * 0.8
                next_q[model_init.POT_IDLE] += w * 0.2
            else:
                next_q[model_init.POT_IDLE] += w
        
        elif pot_state_idx == model_init.POT_COOKING:
            next_q[model_init.POT_READY] += w * 0.2
            next_q[model_init.POT_COOKING] += w * 0.8
        
        elif pot_state_idx == model_init.POT_READY:
            if action == model_init.INTERACT and p_at_pot * p_has_nothing > 0.5:
                next_q[model_init.POT_IDLE] += w * 0.9
                next_q[model_init.POT_READY] += w * 0.1
            else:
                next_q[model_init.POT_READY] += w
    
    return normalize(next_q)


def B_soup_delivered(qs, action):
    """Detect soup delivery."""
    q_pos = np.array(qs["agent_pos"], dtype=float)
    q_held = np.array(qs["agent_held"], dtype=float)
    
    next_q = np.zeros(2, dtype=float)
    
    p_deliver = 0.0
    for pos_idx in range(model_init.GRID_SIZE):
        if model_init.is_at_serving(pos_idx):
            if action == model_init.INTERACT:
                p_deliver += q_pos[pos_idx] * q_held[model_init.HELD_SOUP]
    
    if p_deliver > 0.5:
        next_q[1] = min(p_deliver, 0.5)
        next_q[0] = 1.0 - next_q[1]
    else:
        next_q[0] = 1.0
    
    return normalize(next_q)


def B_fn(qs, action, width=model_init.GRID_WIDTH, height=model_init.GRID_HEIGHT, B_NOISE_LEVEL=0.0):
    """
    Main transition model function for single agent.
    
    Args:
        qs: Dictionary of current state belief distributions
        action: Single agent action index
        width: Grid width (default: 5 for cramped_room)
        height: Grid height (default: 4 for cramped_room)
        B_NOISE_LEVEL: Movement noise level (not used currently)
    
    Returns:
        Dictionary mapping state factor names to transition probability arrays
    """
    action = int(action)
    next_state = {}
    
    next_state["agent_pos"] = B_agent_pos(qs, action)
    next_state["agent_orientation"] = B_agent_orientation(qs, action)
    next_state["agent_held"] = B_agent_held(qs, action)
    next_state["other_agent_pos"] = B_other_agent_pos(qs)
    next_state["pot_state"] = B_pot_state(qs, action)
    next_state["soup_delivered"] = B_soup_delivered(qs, action)
    
    # Final normalization safety
    for f in next_state:
        next_state[f] = normalize(np.array(next_state[f], dtype=float))
    
    return next_state
