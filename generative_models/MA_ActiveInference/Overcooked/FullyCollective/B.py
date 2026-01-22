"""
Transition model (B) for FullyCollective paradigm (JOINT actions) - Cramped Room.

Handles:
- Agent movement with collision detection
- Orientation changes
- Object pickup/interaction at specific locations (pot, dispensers, serving)
- Pot state transitions (idle -> cooking -> ready)
- Soup delivery at serving location
"""

import numpy as np
from . import model_init

state_state_dependencies = model_init.state_state_dependencies


def normalize(p):
    """Normalize probability distribution."""
    return p / np.maximum(np.sum(p), 1e-8)


def decode_joint_action(joint_action):
    """Decode joint action index to (a1, a2)."""
    a = int(joint_action)
    return a // model_init.N_ACTIONS, a % model_init.N_ACTIONS


def _compute_new_pos(pos_idx, action, width=model_init.GRID_WIDTH, height=model_init.GRID_HEIGHT):
    """
    Compute new position index after action.
    
    Args:
        pos_idx: Current position index (flattened)
        action: Action index (NORTH, SOUTH, EAST, WEST, STAY, INTERACT)
        width: Grid width
        height: Grid height
    
    Returns:
        New position index
    """
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


def B_agent_positions(qs, joint_action):
    """
    Update marginals for (agent1_pos, agent2_pos) given joint action.
    Handles collision detection (agents can't occupy same cell).
    """
    q1 = np.array(qs["agent1_pos"], dtype=float)
    q2 = np.array(qs["agent2_pos"], dtype=float)
    S = model_init.GRID_SIZE

    a1, a2 = decode_joint_action(joint_action)

    next_q1 = np.zeros(S, dtype=float)
    next_q2 = np.zeros(S, dtype=float)

    # Iterate over all position combinations
    for p1 in range(S):
        if q1[p1] <= 1e-16:
            continue
        for p2 in range(S):
            if q2[p2] <= 1e-16:
                continue
            if p1 == p2:  # Collision: can't be at same position
                continue
            
            w = q1[p1] * q2[p2]
            if w <= 1e-16:
                continue
            
            # Agent 1 moves first
            np1 = _compute_new_pos(p1, a1)
            # Block if moving into agent 2's current position
            if np1 == p2:
                np1 = p1
            
            # Agent 2 moves next
            np2 = _compute_new_pos(p2, a2)
            # Block if moving into agent 1's new position
            if np2 == np1:
                np2 = p2
            
            # Ensure no collision in final state
            if np1 != np2:
                if np1 < S:
                    next_q1[np1] += w
                if np2 < S:
                    next_q2[np2] += w

    next_q1 = normalize(next_q1)
    next_q2 = normalize(next_q2)
    return next_q1, next_q2


def _update_orientation(ori_idx, action):
    """
    Update orientation based on action.
    Orientation matches movement direction.
    """
    if action in (model_init.NORTH, model_init.SOUTH, model_init.EAST, model_init.WEST):
        return action  # Action indices match orientation indices
    return ori_idx  # Stay or interact: keep orientation


def B_agent_orientations(qs, joint_action):
    """
    Update orientations based on actions.
    """
    q1_ori = np.array(qs["agent1_orientation"], dtype=float)
    q2_ori = np.array(qs["agent2_orientation"], dtype=float)
    
    a1, a2 = decode_joint_action(joint_action)
    
    next_q1_ori = np.zeros(model_init.N_DIRECTIONS, dtype=float)
    next_q2_ori = np.zeros(model_init.N_DIRECTIONS, dtype=float)
    
    # Agent 1 orientation
    for ori_idx in range(model_init.N_DIRECTIONS):
        if q1_ori[ori_idx] <= 1e-16:
            continue
        new_ori = _update_orientation(ori_idx, a1)
        if 0 <= new_ori < model_init.N_DIRECTIONS:
            next_q1_ori[new_ori] += q1_ori[ori_idx]
    
    # Agent 2 orientation
    for ori_idx in range(model_init.N_DIRECTIONS):
        if q2_ori[ori_idx] <= 1e-16:
            continue
        new_ori = _update_orientation(ori_idx, a2)
        if 0 <= new_ori < model_init.N_DIRECTIONS:
            next_q2_ori[new_ori] += q2_ori[ori_idx]
    
    next_q1_ori = normalize(next_q1_ori)
    next_q2_ori = normalize(next_q2_ori)
    return next_q1_ori, next_q2_ori


def B_agent_held(qs, joint_action):
    """
    Update held objects based on interactions at specific locations.
    
    Rules for cramped_room:
    - At onion dispenser + INTERACT + holding nothing -> pick up onion
    - At pot + INTERACT + holding onion + pot idle -> add onion to pot (drop onion)
    - At pot + INTERACT + holding nothing + pot ready -> pick up soup
    - At serving + INTERACT + holding soup -> deliver soup (drop soup, get reward)
    """
    q1_held = np.array(qs["agent1_held"], dtype=float)
    q2_held = np.array(qs["agent2_held"], dtype=float)
    q1_pos = np.array(qs["agent1_pos"], dtype=float)
    q2_pos = np.array(qs["agent2_pos"], dtype=float)
    q_pot_state = np.array(qs["pot_state"], dtype=float)
    
    a1, a2 = decode_joint_action(joint_action)
    
    next_q1_held = np.zeros(model_init.N_HELD_TYPES, dtype=float)
    next_q2_held = np.zeros(model_init.N_HELD_TYPES, dtype=float)
    
    # Agent 1 interactions
    for pos_idx in range(model_init.GRID_SIZE):
        if q1_pos[pos_idx] <= 1e-16:
            continue
        for held_idx in range(model_init.N_HELD_TYPES):
            if q1_held[held_idx] <= 1e-16:
                continue
            for pot_state_idx in range(model_init.N_POT_STATES):
                if q_pot_state[pot_state_idx] <= 1e-16:
                    continue
                
                w = q1_pos[pos_idx] * q1_held[held_idx] * q_pot_state[pot_state_idx]
                if w <= 1e-16:
                    continue
                
                new_held = held_idx  # Default: keep current
                
                if a1 == model_init.INTERACT:
                    # At onion dispenser: pick up onion if holding nothing
                    if model_init.is_at_onion_dispenser(pos_idx) and held_idx == model_init.HELD_NONE:
                        new_held = model_init.HELD_ONION
                    # At pot: add onion if holding onion and pot is idle
                    elif model_init.is_at_pot(pos_idx) and held_idx == model_init.HELD_ONION and pot_state_idx == model_init.POT_IDLE:
                        new_held = model_init.HELD_NONE  # Drop onion into pot
                    # At pot: pick up soup if holding nothing and pot is ready
                    elif model_init.is_at_pot(pos_idx) and held_idx == model_init.HELD_NONE and pot_state_idx == model_init.POT_READY:
                        new_held = model_init.HELD_SOUP
                    # At serving: deliver soup (drop soup, get reward)
                    elif model_init.is_at_serving(pos_idx) and held_idx == model_init.HELD_SOUP:
                        new_held = model_init.HELD_NONE  # Deliver soup
                
                if 0 <= new_held < model_init.N_HELD_TYPES:
                    next_q1_held[new_held] += w
    
    # Agent 2 interactions (same logic)
    for pos_idx in range(model_init.GRID_SIZE):
        if q2_pos[pos_idx] <= 1e-16:
            continue
        for held_idx in range(model_init.N_HELD_TYPES):
            if q2_held[held_idx] <= 1e-16:
                continue
            for pot_state_idx in range(model_init.N_POT_STATES):
                if q_pot_state[pot_state_idx] <= 1e-16:
                    continue
                
                w = q2_pos[pos_idx] * q2_held[held_idx] * q_pot_state[pot_state_idx]
                if w <= 1e-16:
                    continue
                
                new_held = held_idx
                
                if a2 == model_init.INTERACT:
                    if model_init.is_at_onion_dispenser(pos_idx) and held_idx == model_init.HELD_NONE:
                        new_held = model_init.HELD_ONION
                    elif model_init.is_at_pot(pos_idx) and held_idx == model_init.HELD_ONION and pot_state_idx == model_init.POT_IDLE:
                        new_held = model_init.HELD_NONE
                    elif model_init.is_at_pot(pos_idx) and held_idx == model_init.HELD_NONE and pot_state_idx == model_init.POT_READY:
                        new_held = model_init.HELD_SOUP
                    elif model_init.is_at_serving(pos_idx) and held_idx == model_init.HELD_SOUP:
                        new_held = model_init.HELD_NONE
                
                if 0 <= new_held < model_init.N_HELD_TYPES:
                    next_q2_held[new_held] += w
    
    next_q1_held = normalize(next_q1_held)
    next_q2_held = normalize(next_q2_held)
    return next_q1_held, next_q2_held


def B_pot_state(qs, joint_action):
    """
    Update pot state based on agent interactions.
    
    Rules:
    - idle -> cooking: when agent adds onion to pot (INTERACT at pot with onion)
    - cooking -> ready: after some time (simplified: small probability per step)
    - ready -> idle: when agent picks up soup (INTERACT at pot with nothing)
    """
    q_pot_state = np.array(qs["pot_state"], dtype=float)
    q1_pos = np.array(qs["agent1_pos"], dtype=float)
    q2_pos = np.array(qs["agent2_pos"], dtype=float)
    q1_held = np.array(qs["agent1_held"], dtype=float)
    q2_held = np.array(qs["agent2_held"], dtype=float)
    
    a1, a2 = decode_joint_action(joint_action)
    
    next_q = np.zeros(model_init.N_POT_STATES, dtype=float)
    
    # Check if any agent interacts with pot
    p1_at_pot = 0.0
    p2_at_pot = 0.0
    p1_has_onion = 0.0
    p2_has_onion = 0.0
    p1_has_nothing = 0.0
    p2_has_nothing = 0.0
    
    for pos_idx in range(model_init.GRID_SIZE):
        if model_init.is_at_pot(pos_idx):
            p1_at_pot += q1_pos[pos_idx]
            p2_at_pot += q2_pos[pos_idx]
    
    p1_has_onion = q1_held[model_init.HELD_ONION]
    p2_has_onion = q2_held[model_init.HELD_ONION]
    p1_has_nothing = q1_held[model_init.HELD_NONE]
    p2_has_nothing = q2_held[model_init.HELD_NONE]
    
    # Iterate over current pot states
    for pot_state_idx in range(model_init.N_POT_STATES):
        if q_pot_state[pot_state_idx] <= 1e-16:
            continue
        
        w = q_pot_state[pot_state_idx]
        
        if pot_state_idx == model_init.POT_IDLE:
            # idle -> cooking: if agent adds onion
            p_add_onion = 0.0
            if a1 == model_init.INTERACT:
                p_add_onion += p1_at_pot * p1_has_onion
            if a2 == model_init.INTERACT:
                p_add_onion += p2_at_pot * p2_has_onion
            
            if p_add_onion > 0.5:  # Threshold
                next_q[model_init.POT_COOKING] += w * 0.8  # 80% chance to start cooking
                next_q[model_init.POT_IDLE] += w * 0.2
            else:
                next_q[model_init.POT_IDLE] += w
        
        elif pot_state_idx == model_init.POT_COOKING:
            # cooking -> ready: after time (simplified: 20% chance per step)
            next_q[model_init.POT_READY] += w * 0.2
            next_q[model_init.POT_COOKING] += w * 0.8
        
        elif pot_state_idx == model_init.POT_READY:
            # ready -> idle: if agent picks up soup
            p_pickup = 0.0
            if a1 == model_init.INTERACT:
                p_pickup += p1_at_pot * p1_has_nothing
            if a2 == model_init.INTERACT:
                p_pickup += p2_at_pot * p2_has_nothing
            
            if p_pickup > 0.5:
                next_q[model_init.POT_IDLE] += w * 0.9  # 90% chance to pick up
                next_q[model_init.POT_READY] += w * 0.1
            else:
                next_q[model_init.POT_READY] += w
    
    return normalize(next_q)


def B_soup_delivered(qs, joint_action):
    """
    Detect soup delivery at serving location.
    """
    q1_pos = np.array(qs["agent1_pos"], dtype=float)
    q2_pos = np.array(qs["agent2_pos"], dtype=float)
    q1_held = np.array(qs["agent1_held"], dtype=float)
    q2_held = np.array(qs["agent2_held"], dtype=float)
    
    a1, a2 = decode_joint_action(joint_action)
    
    next_q = np.zeros(2, dtype=float)
    
    # Check if agent delivers soup
    p_deliver = 0.0
    
    for pos_idx in range(model_init.GRID_SIZE):
        if model_init.is_at_serving(pos_idx):
            if a1 == model_init.INTERACT:
                p_deliver += q1_pos[pos_idx] * q1_held[model_init.HELD_SOUP]
            if a2 == model_init.INTERACT:
                p_deliver += q2_pos[pos_idx] * q2_held[model_init.HELD_SOUP]
    
    if p_deliver > 0.5:
        next_q[1] = min(p_deliver, 0.8)  # Max 80% chance per step
        next_q[0] = 1.0 - next_q[1]
    else:
        next_q[0] = 1.0
    
    return normalize(next_q)


def B_fn(qs, action, width=model_init.GRID_WIDTH, height=model_init.GRID_HEIGHT, B_NOISE_LEVEL=0.0):
    """
    Main transition model function.
    
    Args:
        qs: Dictionary of current state belief distributions
        action: Joint action index
        width: Grid width (default: 5 for cramped_room)
        height: Grid height (default: 4 for cramped_room)
        B_NOISE_LEVEL: Movement noise level (not used currently)
    
    Returns:
        Dictionary mapping state factor names to transition probability arrays
    """
    joint_action = int(action)
    next_state = {}
    
    # Update positions
    next_q1_pos, next_q2_pos = B_agent_positions(qs, joint_action)
    next_state["agent1_pos"] = next_q1_pos
    next_state["agent2_pos"] = next_q2_pos
    
    # Update orientations
    next_q1_ori, next_q2_ori = B_agent_orientations(qs, joint_action)
    next_state["agent1_orientation"] = next_q1_ori
    next_state["agent2_orientation"] = next_q2_ori
    
    # Update held objects
    next_q1_held, next_q2_held = B_agent_held(qs, joint_action)
    next_state["agent1_held"] = next_q1_held
    next_state["agent2_held"] = next_q2_held
    
    # Update pot state
    next_q_pot = B_pot_state(qs, joint_action)
    next_state["pot_state"] = next_q_pot
    
    # Update soup delivery
    next_q_soup = B_soup_delivered(qs, joint_action)
    next_state["soup_delivered"] = next_q_soup
    
    # Final normalization safety
    for f in next_state:
        next_state[f] = normalize(np.array(next_state[f], dtype=float))
    
    return next_state
