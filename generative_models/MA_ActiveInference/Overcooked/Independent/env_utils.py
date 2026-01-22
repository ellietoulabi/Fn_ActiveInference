"""
Env <-> model utilities for Independent paradigm - Cramped Room.

This adapter converts Overcooked observations into single-agent model
observations from a specific agent's perspective.
"""

from . import model_init


def env_obs_to_model_obs(env_state, agent_idx, reward_info=None):
    """
    Convert OvercookedState to single-agent model observation indices.
    
    Args:
        env_state: OvercookedState object
        agent_idx: Index of this agent (0 or 1)
        reward_info: Optional dict with reward information
    
    Returns:
        Dictionary mapping observation modality names to indices
    """
    # Get this agent's info
    this_agent = env_state.players[agent_idx]
    other_agent = env_state.players[1 - agent_idx]
    
    # Positions
    this_pos = this_agent.position
    other_pos = other_agent.position
    
    this_pos_idx = model_init.xy_to_index(this_pos[0], this_pos[1])
    other_pos_idx = model_init.xy_to_index(other_pos[0], other_pos[1])
    
    # Orientation
    this_ori_idx = model_init.direction_to_index(this_agent.orientation)
    
    # Held object
    this_held = model_init.object_name_to_held_type(
        this_agent.held_object.name if this_agent.has_object() else None
    )
    
    # Get pot state
    pot_state = model_init.POT_IDLE
    for pos, obj in env_state.objects.items():
        if model_init.xy_to_index(pos[0], pos[1]) in model_init.POT_INDICES:
            if obj.name == "soup":
                if obj.is_ready:
                    pot_state = model_init.POT_READY
                elif obj.is_cooking:
                    pot_state = model_init.POT_COOKING
            break
    
    # Soup delivery
    soup_delivered = 0
    if reward_info is not None:
        sparse_reward = reward_info.get("sparse_reward_by_agent", [0, 0])
        if sparse_reward[agent_idx] > 0:
            soup_delivered = 1
    
    return {
        "agent_pos": this_pos_idx,
        "agent_orientation": this_ori_idx,
        "agent_held": this_held,
        "other_agent_pos": other_pos_idx,
        "pot_state": pot_state,
        "soup_delivered": soup_delivered,
    }


def get_D_config_from_mdp(mdp, state, agent_idx):
    """
    Extract D_fn config for this agent.
    
    Args:
        mdp: OvercookedGridworld instance
        state: OvercookedState instance
        agent_idx: Index of this agent (0 or 1)
    
    Returns:
        Dictionary with configuration for D_fn
    """
    this_agent = state.players[agent_idx]
    other_agent = state.players[1 - agent_idx]
    
    this_pos = this_agent.position
    other_pos = other_agent.position
    
    return {
        "agent_start_pos": model_init.xy_to_index(this_pos[0], this_pos[1]),
        "agent_start_ori": model_init.direction_to_index(this_agent.orientation),
        "other_agent_start_pos": model_init.xy_to_index(other_pos[0], other_pos[1]),
    }


def model_action_to_env_action(model_action):
    """
    Convert model action index to Action object.
    
    Args:
        model_action: Action index [0, 5]
    
    Returns:
        Action object
    """
    from overcooked_ai_py.mdp.actions import Action
    return Action.INDEX_TO_ACTION[int(model_action)]


def env_action_to_model_action(env_action):
    """
    Convert Action object to model action index.
    
    Args:
        env_action: Action object
    
    Returns:
        Action index [0, 5]
    """
    from overcooked_ai_py.mdp.actions import Action
    return Action.ACTION_TO_INDEX[env_action]
