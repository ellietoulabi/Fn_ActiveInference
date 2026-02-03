"""
Env <-> model utilities for Independent paradigm - Cramped Room (Stage 1).

This adapter converts Overcooked observations into single-agent model
observations from a specific agent's perspective.

Stage 1 model details reflected here:
- front_tile_type is computed inside A from (agent_pos, agent_orientation, other_agent_pos),
  so we do NOT need to output it here.
- soup_delivered is OBSERVATION-ONLY (event), not a hidden state factor.
- pot_state supports 3-onion recipe:
    POT_0, POT_1, POT_2, POT_COOKING, POT_READY
"""

from . import model_init


def _extract_pot_state_from_env(env_state):
    """
    Map Overcooked env pot contents to our abstract pot_state.

    We assume:
      - env has a soup object at the pot location when anything is in the pot.
      - soup.ingredients encodes how many onions are in the pot.
      - soup.is_cooking / soup.is_ready indicate stage.
    """
    # Default: empty pot
    pot_state = model_init.POT_0

    # Find the object at pot location (there is typically a soup object when pot has ingredients)
    for (pos, obj) in env_state.objects.items():
        idx = model_init.xy_to_index(pos[0], pos[1])
        if idx not in model_init.POT_INDICES:
            continue

        # Pot cell found
        if obj is None:
            return model_init.POT_0

        # In Overcooked, pot usually contains a 'soup' object with ingredients list
        if getattr(obj, "name", None) == "soup":
            # Count onions in soup object
            ingredients = getattr(obj, "ingredients", None)

            # ingredients might be a list of objects, names, or strings depending on version
            onion_count = 0
            if ingredients is None:
                onion_count = 0
            else:
                for ing in ingredients:
                    # robust: ing may be "onion" or an object with .name
                    ing_name = ing if isinstance(ing, str) else getattr(ing, "name", None)
                    if ing_name == "onion":
                        onion_count += 1

            # Cooking/ready flags
            is_ready = bool(getattr(obj, "is_ready", False))
            is_cooking = bool(getattr(obj, "is_cooking", False))

            # Map to our pot abstraction
            if is_ready:
                return model_init.POT_READY
            if is_cooking:
                return model_init.POT_COOKING

            # Not cooking yet: represent by onion count (clipped to 0..2; 3 implies cooking usually)
            if onion_count <= 0:
                return model_init.POT_0
            if onion_count == 1:
                return model_init.POT_1
            if onion_count == 2:
                return model_init.POT_2

            # onion_count >= 3 but not flagged cooking/ready: treat as COOKING (closest)
            return model_init.POT_COOKING

        # If something else is at pot (rare), keep empty prior
        return model_init.POT_0

    # If we didn't find pot object in env_state.objects, assume empty
    return pot_state


def env_obs_to_model_obs(env_state, agent_idx, reward_info=None):
    """
    Convert OvercookedState to single-agent model observation indices.

    Args:
        env_state: OvercookedState object
        agent_idx: Index of this agent (0 or 1)
        reward_info: Optional dict with reward information (to detect delivery event)

    Returns:
        dict mapping observation modality names to indices
        (these are OBSERVATIONS, not necessarily hidden states)
    """
    # Get this agent's info
    this_agent = env_state.players[agent_idx]
    other_agent = env_state.players[1 - agent_idx]

    # Positions
    this_pos = this_agent.position
    other_pos = other_agent.position

    this_pos_idx = model_init.xy_to_index(this_pos[0], this_pos[1])
    other_pos_idx = model_init.xy_to_index(other_pos[0], other_pos[1])

    # Orientation: IMPORTANT
    # Your model_init uses orientation indices [N,S,E,W] = [0,1,2,3]
    # Ensure direction_to_index uses the same convention.
    this_ori_idx = model_init.direction_to_index(this_agent.orientation)

    # Held object
    this_held = model_init.object_name_to_held_type(
        this_agent.held_object.name if this_agent.has_object() else None
    )

    # Pot state (expanded)
    pot_state = _extract_pot_state_from_env(env_state)

    # soup_delivered is an OBSERVATION-ONLY event (1 only on delivery timestep)
    soup_delivered = 0
    if reward_info is not None:
        # Typical pattern: sparse_reward_by_agent gives +20 on delivery step
        sparse_reward = reward_info.get("sparse_reward_by_agent", None)
        if sparse_reward is not None and len(sparse_reward) > agent_idx:
            soup_delivered = 1 if sparse_reward[agent_idx] > 0 else 0
        else:
            # fallback: check scalar sparse_reward if provided
            r = reward_info.get("sparse_reward", 0)
            soup_delivered = 1 if r > 0 else 0

    return {
        "agent_pos": this_pos_idx,
        "agent_orientation": this_ori_idx,
        "agent_held": this_held,
        "other_agent_pos": other_pos_idx,
        "pot_state": pot_state,
        "soup_delivered": soup_delivered,  # obs-only event
        # NOTE: front_tile_type is computed by A from (pos,ori,other_pos), so not provided here.
    }


def get_D_config_from_state(state, agent_idx):
    """
    Extract initial positions/orientations from the *actual env state*.

    Returns a config dict suitable for D_fn(config) so that prior D matches
    the true initial state (positions and orientation).
    """
    this_agent = state.players[agent_idx]
    other_agent = state.players[1 - agent_idx]

    this_pos = this_agent.position
    other_pos = other_agent.position

    this_pos_idx = model_init.xy_to_index(this_pos[0], this_pos[1])
    other_pos_idx = model_init.xy_to_index(other_pos[0], other_pos[1])

    this_ori_idx = model_init.direction_to_index(this_agent.orientation)

    return {
        "agent_start_pos": this_pos_idx,
        "agent_start_ori": this_ori_idx,
        "other_agent_start_pos": other_pos_idx,
    }


def model_action_to_env_action(model_action):
    """
    Convert model action index to Action object.
    """
    from overcooked_ai_py.mdp.actions import Action
    return Action.INDEX_TO_ACTION[int(model_action)]


def env_action_to_model_action(env_action):
    """
    Convert Action object to model action index.
    """
    from overcooked_ai_py.mdp.actions import Action
    return Action.ACTION_TO_INDEX[env_action]
