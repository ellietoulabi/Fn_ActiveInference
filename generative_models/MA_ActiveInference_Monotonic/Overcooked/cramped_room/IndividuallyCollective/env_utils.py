# env_utils.py
"""
Env <-> model utilities for IndividuallyCollective paradigm - Cramped Room (Monotonic checkbox model).

Converts Overcooked env state to single-agent model observations.
- Agent position uses walkable indices 0..5 (not grid 0..19).
- Pot state uses 4 values: POT_0, POT_1, POT_2, POT_3 (ready); cook_time=0.
- Adds binary counter occupancy observations for MODELED_COUNTERS.
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


def _counter_occupancy_from_env(env_state, grid_idx: int) -> int:
    """
    Binary counter occupancy:
      CTR_EMPTY if no object at (x,y)
      CTR_FULL  if any object at (x,y)
    """
    x, y = model_init.index_to_xy(grid_idx)
    obj = env_state.objects.get((x, y), None)
    return model_init.CTR_FULL if obj is not None else model_init.CTR_EMPTY


def env_obs_to_model_obs(env_state, agent_idx, reward_info=None):
    """
    Convert OvercookedState to single-agent model observation indices.

    Returns dict with keys matching model_init.observations:
    self_pos_obs, self_orientation_obs, self_held_obs, pot_state_obs, soup_delivered_obs,
    plus ctr_<grid>_obs for each modeled counter.
    """
    this_agent = env_state.players[agent_idx]

    # Overcooked position is (x, y) with x=column, y=row.
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

    obs = {
        "self_pos_obs": this_pos_walkable,
        "self_orientation_obs": this_ori_idx,
        "self_held_obs": this_held,
        "pot_state_obs": pot_state,
        "soup_delivered_obs": soup_delivered,
    }

    # Add binary counter occupancy observations
    for grid_idx in model_init.MODELED_COUNTERS:
        obs[f"ctr_{grid_idx}_obs"] = _counter_occupancy_from_env(env_state, grid_idx)

    return obs


def get_D_config_from_state(state, agent_idx):
    """
    Extract initial position (walkable index) and orientation from env state.

    Returns a config dict for D_fn(config). self_start_pos is in walkable space (0..5).
    """
    this_agent = state.players[agent_idx]

    this_pos = this_agent.position
    this_pos_grid = model_init.xy_to_index(this_pos[0], this_pos[1])
    this_pos_walkable = model_init.grid_idx_to_walkable_idx(this_pos_grid)
    if this_pos_walkable is None:
        this_pos_walkable = 0

    this_ori_idx = model_init.direction_to_index(this_agent.orientation)

    return {
        "self_start_pos": this_pos_walkable,
        "self_start_ori": this_ori_idx,
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