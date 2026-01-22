"""
Env <-> model utilities for FullyCollective paradigm - Cramped Room.

This model sees the FULL joint observation and selects a JOINT action (a1, a2),
encoded as a single integer in [0, 35].
"""

from . import model_init


def encode_joint_action(a1, a2):
    """Encode joint action (a1, a2) to single integer."""
    return int(a1) * model_init.N_ACTIONS + int(a2)


def decode_joint_action(joint_action):
    """Decode joint action index to (a1, a2)."""
    a = int(joint_action)
    return a // model_init.N_ACTIONS, a % model_init.N_ACTIONS


def env_obs_to_model_obs(env_state, reward_info=None):
    """
    Convert OvercookedState to joint-model observation indices.
    
    Args:
        env_state: OvercookedState object
        reward_info: Optional dict with reward information (for soup_delivered)
    
    Returns:
        Dictionary mapping observation modality names to indices
    """
    # Get player positions and orientations
    pos1, pos2 = env_state.player_positions
    ori1, ori2 = env_state.player_orientations
    
    # Convert positions to indices
    a1_pos_idx = model_init.xy_to_index(pos1[0], pos1[1])
    a2_pos_idx = model_init.xy_to_index(pos2[0], pos2[1])
    
    # Convert orientations to indices
    a1_ori_idx = model_init.direction_to_index(ori1)
    a2_ori_idx = model_init.direction_to_index(ori2)
    
    # Get held objects
    player1 = env_state.players[0]
    player2 = env_state.players[1]
    
    a1_held = model_init.object_name_to_held_type(
        player1.held_object.name if player1.has_object() else None
    )
    a2_held = model_init.object_name_to_held_type(
        player2.held_object.name if player2.has_object() else None
    )
    
    # Get pot state
    pot_state = model_init.POT_IDLE  # Default
    pot_locations = model_init.POT_INDICES
    for pos, obj in env_state.objects.items():
        if model_init.xy_to_index(pos[0], pos[1]) in pot_locations:
            if obj.name == "soup":
                if obj.is_ready:
                    pot_state = model_init.POT_READY
                elif obj.is_cooking:
                    pot_state = model_init.POT_COOKING
                else:
                    pot_state = model_init.POT_IDLE
            break
    
    # Check for soup delivery (from reward info or sparse reward)
    soup_delivered = 0
    if reward_info is not None:
        sparse_reward = reward_info.get("sparse_reward_by_agent", [0, 0])
        if sum(sparse_reward) > 0:
            soup_delivered = 1
    
    return {
        "agent1_pos": a1_pos_idx,
        "agent2_pos": a2_pos_idx,
        "agent1_orientation": a1_ori_idx,
        "agent2_orientation": a2_ori_idx,
        "agent1_held": a1_held,
        "agent2_held": a2_held,
        "pot_state": pot_state,
        "soup_delivered": soup_delivered,
    }


def get_D_config_from_mdp(mdp, state):
    """
    Extract D_fn config from an OvercookedGridworld MDP and state.
    
    Args:
        mdp: OvercookedGridworld instance
        state: OvercookedState instance
    
    Returns:
        Dictionary with configuration for D_fn
    """
    pos1, pos2 = state.player_positions
    ori1, ori2 = state.player_orientations
    
    return {
        "agent1_start_pos": model_init.xy_to_index(pos1[0], pos1[1]),
        "agent2_start_pos": model_init.xy_to_index(pos2[0], pos2[1]),
        "agent1_start_ori": model_init.direction_to_index(ori1),
        "agent2_start_ori": model_init.direction_to_index(ori2),
    }


def model_action_to_env_action(model_action):
    """
    Convert joint action index to (action1, action2) tuple for env.step(...).
    
    Args:
        model_action: Joint action index [0, 35]
    
    Returns:
        Tuple of (action1, action2) where each is an Action object
    """
    from overcooked_ai_py.mdp.actions import Action
    
    a1_idx, a2_idx = decode_joint_action(model_action)
    a1 = Action.INDEX_TO_ACTION[a1_idx]
    a2 = Action.INDEX_TO_ACTION[a2_idx]
    return (a1, a2)


def env_action_to_model_action(env_action):
    """
    Convert (action1, action2) tuple to joint action index.
    
    Args:
        env_action: Tuple of (action1, action2) Action objects
    
    Returns:
        Joint action index [0, 35]
    """
    from overcooked_ai_py.mdp.actions import Action
    
    a1, a2 = env_action
    a1_idx = Action.ACTION_TO_INDEX[a1]
    a2_idx = Action.ACTION_TO_INDEX[a2]
    return encode_joint_action(a1_idx, a2_idx)
