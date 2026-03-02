"""
Env <-> model utilities for Independent paradigm - Cramped Room (Monotonic checkbox model).

Converts Overcooked env state to single-agent model observations.
- Agent position uses walkable indices 0..5 (not grid 0..19).
- Pot state uses 4 values: POT_0, POT_1, POT_2, POT_3 (ready); cook_time=0.
- Observation keys match model_init.observations: agent_pos_obs, agent_orientation_obs,
  agent_held_obs, pot_state_obs, soup_delivered_obs, counter_0_obs..counter_4_obs.
"""

from . import model_init


def _extract_pot_state_from_env(env_state):
    """
    Map Overcooked env pot contents to our 4-state pot abstraction.

    POT_0=idle, POT_1=1 onion, POT_2=2 onions, POT_3=ready (3 onions; cook_time=0).
    """
    pot_state = model_init.POT_0

    for (pos, obj) in env_state.objects.items():
        idx = model_init.xy_to_index(pos[0], pos[1])
        if idx not in model_init.POT_INDICES:
            continue

        if obj is None:
            return model_init.POT_0

        if getattr(obj, "name", None) == "soup":
            ingredients = getattr(obj, "ingredients", None)
            onion_count = 0
            if ingredients is not None:
                for ing in ingredients:
                    ing_name = ing if isinstance(ing, str) else getattr(ing, "name", None)
                    if ing_name == "onion":
                        onion_count += 1

            is_ready = bool(getattr(obj, "is_ready", False))
            is_cooking = bool(getattr(obj, "is_cooking", False))

            # 4 states: 0, 1, 2 onions or ready (POT_3)
            if is_ready or is_cooking or onion_count >= 3:
                return model_init.POT_3
            if onion_count <= 0:
                return model_init.POT_0
            if onion_count == 1:
                return model_init.POT_1
            if onion_count == 2:
                return model_init.POT_2
            return model_init.POT_3

        return model_init.POT_0

    return pot_state


def _extract_counter_slots_from_env(env_state):
    """
    Map Overcooked env counter objects to our 5 modeled counter slots.

    Each slot value is in HELD_* space: {NONE, ONION, DISH, SOUP}.
    """
    slots = []
    for grid_idx in getattr(model_init, "COUNTER_SLOT_GRID_IDXS", []):
        x, y = model_init.index_to_xy(int(grid_idx))
        obj = env_state.objects.get((x, y), None)
        obj_name = getattr(obj, "name", None) if obj is not None else None
        slots.append(model_init.object_name_to_held_type(obj_name))
    # Backward-compatible fallback if COUNTER_SLOT_GRID_IDXS missing
    while len(slots) < 5:
        slots.append(model_init.HELD_NONE)
    return slots[:5]


def env_obs_to_model_obs(env_state, agent_idx, reward_info=None):
    """
    Convert OvercookedState to single-agent model observation indices.

    Returns dict with keys matching model_init.observations:
    agent_pos_obs, agent_orientation_obs, agent_held_obs, pot_state_obs, soup_delivered_obs,
    counter_0_obs..counter_4_obs.
    Agent position is in walkable index space (0..5).
    """
    this_agent = env_state.players[agent_idx]

    # Overcooked position is (x, y) with x=column, y=row; same for both agents.
    this_pos = this_agent.position
    x, y = this_pos[0], this_pos[1]
    this_pos_grid = model_init.xy_to_index(x, y)
    this_pos_walkable = model_init.grid_idx_to_walkable_idx(this_pos_grid)
    if this_pos_walkable is None:
        this_pos_walkable = 0  # fallback if env position not on floor (should not happen)

    this_ori_idx = model_init.direction_to_index(this_agent.orientation)

    this_held = model_init.object_name_to_held_type(
        this_agent.held_object.name if this_agent.has_object() else None
    )

    pot_state = _extract_pot_state_from_env(env_state)

    soup_delivered = 0
    if reward_info is not None:
        sparse_reward = reward_info.get("sparse_reward_by_agent", None)
        if sparse_reward is not None and len(sparse_reward) > agent_idx:
            soup_delivered = 1 if sparse_reward[agent_idx] > 0 else 0
        else:
            r = reward_info.get("sparse_reward", 0)
            soup_delivered = 1 if r > 0 else 0

    cslots = _extract_counter_slots_from_env(env_state)

    return {
        "agent_pos_obs": this_pos_walkable,
        "agent_orientation_obs": this_ori_idx,
        "agent_held_obs": this_held,
        "pot_state_obs": pot_state,
        "soup_delivered_obs": soup_delivered,
        "counter_0_obs": int(cslots[0]),
        "counter_1_obs": int(cslots[1]),
        "counter_2_obs": int(cslots[2]),
        "counter_3_obs": int(cslots[3]),
        "counter_4_obs": int(cslots[4]),
    }


def get_D_config_from_state(state, agent_idx):
    """
    Extract initial position (walkable index) and orientation from env state.

    Returns a config dict for D_fn(config). agent_start_pos is in walkable space (0..5).
    """
    this_agent = state.players[agent_idx]

    this_pos = this_agent.position
    this_pos_grid = model_init.xy_to_index(this_pos[0], this_pos[1])
    this_pos_walkable = model_init.grid_idx_to_walkable_idx(this_pos_grid)
    if this_pos_walkable is None:
        this_pos_walkable = 0

    this_ori_idx = model_init.direction_to_index(this_agent.orientation)

    return {
        "agent_start_pos": this_pos_walkable,
        "agent_start_ori": this_ori_idx,
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
