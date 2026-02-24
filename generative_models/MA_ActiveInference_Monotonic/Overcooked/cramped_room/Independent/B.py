"""
Transition model (B) for Independent paradigm — Cramped Room (Monotonic checkbox model).

Full p(s' | s, a) for the single-agent, 6-walkable-position model. The agent knows the
layout (onion/dish dispensers, pot, serving, counters); front tile is computed from
(pos, orientation) via model_init.compute_front_tile_type.

- agent_pos: movement only to walkable cells; INTERACT and STAY do not change position.
  The agent never steps onto the pot, serving, or dispensers — it interacts with the
  adjacent cell in front (see "front tile" below).
- agent_orientation: deterministic move and turn.
- agent_held: INTERACT applies to the tile in front of the agent (not the tile under the
  agent). At front ONION/DISH -> pickup; at front POT -> put onion or take soup; at front
  SERVE with soup -> deliver; at front COUNTER -> drop. INTERACT_SUCCESS_PROB applied.
- pot_state: POT_0->POT_1->POT_2->POT_3 on onion deposits; POT_3->POT_0 when taking soup.
- Checkboxes (ck_put1, ck_put2, ck_put3, ck_plated, ck_delivered): monotonic progress,
  reset to 0 on delivery.
"""

import numpy as np
from . import model_init

state_state_dependencies = model_init.state_state_dependencies


# -------------------------------------------------
# Utility
# -------------------------------------------------
def normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    return p / max(s, 1e-8)


def _update_orientation(ori_idx: int, action: int) -> int:
    """
    Deterministic orientation update: directional actions set orientation, others keep it.
    """
    if action in (model_init.NORTH, model_init.SOUTH, model_init.EAST, model_init.WEST):
        return int(action)
    return int(ori_idx)


def _move_walkable(walkable_idx: int, action: int) -> int:
    """
    Move agent within walkable index space 0..N_WALKABLE-1.
    Only walkable cells are valid targets; pot, serving, dispensers are not steppable.
    STAY and INTERACT do not change position (INTERACT affects the cell in front, not under the agent).
    """
    if action in (model_init.STAY, model_init.INTERACT):
        return int(walkable_idx)

    grid_idx = model_init.walkable_idx_to_grid_idx(int(walkable_idx))
    x, y = model_init.index_to_xy(grid_idx)

    if action == model_init.NORTH:
        y -= 1
    elif action == model_init.SOUTH:
        y += 1
    elif action == model_init.EAST:
        x += 1
    elif action == model_init.WEST:
        x -= 1

    if x < 0 or x >= model_init.GRID_WIDTH or y < 0 or y >= model_init.GRID_HEIGHT:
        return int(walkable_idx)

    new_grid = model_init.xy_to_index(x, y)
    new_walkable = model_init.grid_idx_to_walkable_idx(new_grid)
    if new_walkable is None:
        return int(walkable_idx)
    return int(new_walkable)


def B_agent_pos(parents: dict, action: int) -> np.ndarray:
    """
    B for agent_pos (walkable 0..5). Depends only on previous agent_pos.
    Movement is deterministic within walkable cells; no other-agent collisions in Independent model.
    """
    q_pos = np.array(parents["agent_pos"], dtype=float)
    S = q_pos.shape[0]
    next_q = np.zeros(S, dtype=float)

    for w in range(S):
        p = q_pos[w]
        if p <= 1e-16:
            continue
        new_w = _move_walkable(w, action)
        next_q[new_w] += p

    return normalize(next_q)


def B_agent_orientation(parents: dict, action: int) -> np.ndarray:
    """
    B for agent_orientation. Depends only on previous orientation, updated by action.
    """
    q_ori = np.array(parents["agent_orientation"], dtype=float)
    next_q = np.zeros(model_init.N_DIRECTIONS, dtype=float)

    for ori in range(model_init.N_DIRECTIONS):
        p = q_ori[ori]
        if p <= 1e-16:
            continue
        new_ori = _update_orientation(ori, action)
        next_q[new_ori] += p

    return normalize(next_q)


def _get_front(pos_w: int, ori: int) -> int:
    """Front tile type for (walkable pos, orientation)."""
    return model_init.compute_front_tile_type(pos_w, ori)


INTERACT_SUCCESS_PROB = getattr(model_init, "INTERACT_SUCCESS_PROB", 0.9)


def B_agent_held(parents: dict, action: int) -> np.ndarray:
    """
    B for agent_held. Depends on: agent_pos, agent_orientation, agent_held, pot_state.

    INTERACT rules (agent knows layout — front = compute_front_tile_type(pos, ori)):
    - Front ONION + held NONE -> pick onion.
    - Front DISH  + held NONE -> pick dish.
    - Front POT   + held ONION + pot in (0,1,2) -> put onion -> held NONE.
    - Front POT   + held DISH  + pot POT_3 -> take soup -> held SOUP.
    - Front SERVE + held SOUP -> deliver -> held NONE.
    - Front COUNTER + held in (ONION, DISH, SOUP) -> drop -> held NONE.
      (Counters are not explicitly modelled as state; they just clear held.)
    """
    q_pos = np.array(parents["agent_pos"], dtype=float)
    q_ori = np.array(parents["agent_orientation"], dtype=float)
    q_held = np.array(parents["agent_held"], dtype=float)
    q_pot = np.array(parents["pot_state"], dtype=float)

    next_q = np.zeros(model_init.N_HELD_TYPES, dtype=float)

    for pos_w in range(model_init.N_WALKABLE):
        if q_pos[pos_w] <= 1e-16:
            continue
        for ori in range(model_init.N_DIRECTIONS):
            if q_ori[ori] <= 1e-16:
                continue
            front = _get_front(pos_w, ori)
            for held in range(model_init.N_HELD_TYPES):
                if q_held[held] <= 1e-16:
                    continue
                for pot in range(model_init.N_POT_STATES):
                    if q_pot[pot] <= 1e-16:
                        continue
                    w = q_pos[pos_w] * q_ori[ori] * q_held[held] * q_pot[pot]
                    if w <= 1e-16:
                        continue

                    new_held = held
                    p_success = 1.0
                    if action == model_init.INTERACT:
                        if front == model_init.FRONT_ONION and held == model_init.HELD_NONE:
                            new_held = model_init.HELD_ONION
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_DISH and held == model_init.HELD_NONE:
                            new_held = model_init.HELD_DISH
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_POT and held == model_init.HELD_ONION and pot in (model_init.POT_0, model_init.POT_1, model_init.POT_2):
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_POT and held == model_init.HELD_DISH and pot == model_init.POT_3:
                            new_held = model_init.HELD_SOUP
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_SERVE and held == model_init.HELD_SOUP:
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_COUNTER and held in (model_init.HELD_ONION, model_init.HELD_DISH, model_init.HELD_SOUP):
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                    next_q[new_held] += w * p_success
                    if p_success < 1.0:
                        next_q[held] += w * (1.0 - p_success)

    return normalize(next_q)


def B_pot_state(parents: dict, action: int) -> np.ndarray:
    """
    B for pot_state. Depends on: agent_pos, agent_orientation, agent_held, pot_state.

    - POT_0 -> POT_1 when INTERACT at pot with onion.
    - POT_1 -> POT_2, POT_2 -> POT_3 same.
    - POT_3 -> POT_0 when INTERACT at pot with dish (take soup).
    - cook_time=0 so no separate cooking step; POT_3 is ready.
    """
    q_pos = np.array(parents["agent_pos"], dtype=float)
    q_ori = np.array(parents["agent_orientation"], dtype=float)
    q_held = np.array(parents["agent_held"], dtype=float)
    q_pot = np.array(parents["pot_state"], dtype=float)

    next_q = np.zeros(model_init.N_POT_STATES, dtype=float)

    for pot in range(model_init.N_POT_STATES):
        if q_pot[pot] <= 1e-16:
            continue
        for pos_w in range(model_init.N_WALKABLE):
            if q_pos[pos_w] <= 1e-16:
                continue
            for ori in range(model_init.N_DIRECTIONS):
                if q_ori[ori] <= 1e-16:
                    continue
                front = _get_front(pos_w, ori)
                for held in range(model_init.N_HELD_TYPES):
                    if q_held[held] <= 1e-16:
                        continue
                    w = q_pot[pot] * q_pos[pos_w] * q_ori[ori] * q_held[held]
                    if w <= 1e-16:
                        continue

                    new_pot = pot
                    p_success = 1.0
                    if action == model_init.INTERACT and front == model_init.FRONT_POT:
                        if pot == model_init.POT_0 and held == model_init.HELD_ONION:
                            new_pot = model_init.POT_1
                            p_success = INTERACT_SUCCESS_PROB
                        elif pot == model_init.POT_1 and held == model_init.HELD_ONION:
                            new_pot = model_init.POT_2
                            p_success = INTERACT_SUCCESS_PROB
                        elif pot == model_init.POT_2 and held == model_init.HELD_ONION:
                            new_pot = model_init.POT_3
                            p_success = INTERACT_SUCCESS_PROB
                        elif pot == model_init.POT_3 and held == model_init.HELD_DISH:
                            new_pot = model_init.POT_0
                            p_success = INTERACT_SUCCESS_PROB

                    next_q[new_pot] += w * p_success
                    if p_success < 1.0:
                        next_q[pot] += w * (1.0 - p_success)
    return normalize(next_q)


def _milestone_this_step(pos_w: int, ori: int, held: int, pot: int, action: int):
    """
    Given (pos, ori, held, pot) and action, return which milestones occur this step (with success prob).
    Used by checkbox B. Returns (did_put1, did_put2, did_put3, did_plated, did_deliver) as booleans
    for the deterministic outcome; belief-level will marginalize over (pos, ori, held, pot).
    """
    front = _get_front(pos_w, ori)
    did_put1 = action == model_init.INTERACT and front == model_init.FRONT_POT and held == model_init.HELD_ONION and pot == model_init.POT_0
    did_put2 = action == model_init.INTERACT and front == model_init.FRONT_POT and held == model_init.HELD_ONION and pot == model_init.POT_1
    did_put3 = action == model_init.INTERACT and front == model_init.FRONT_POT and held == model_init.HELD_ONION and pot == model_init.POT_2
    did_plated = action == model_init.INTERACT and front == model_init.FRONT_POT and held == model_init.HELD_DISH and pot == model_init.POT_3
    did_deliver = action == model_init.INTERACT and front == model_init.FRONT_SERVE and held == model_init.HELD_SOUP
    return did_put1, did_put2, did_put3, did_plated, did_deliver


def B_checkboxes(parents: dict, action: int) -> dict[str, np.ndarray]:
    """
    B for all five checkboxes. Monotonic progress within a cycle; reset on delivery.
    Chain implicit (no intermediate rewards): put1 then put2 then put3.

    - ck_put1=1 only when first onion put (pot POT_0->POT_1).
    - ck_put2=1 only when second onion put (pot POT_1->POT_2); requires put1 done.
    - ck_put3=1 only when third onion put (pot POT_2->POT_3); requires put1, put2 done.
    - ck_plated=1 when picked up soup (dish at pot POT_3 -> held SOUP).
    - ck_delivered=1 when delivered (soup at serve -> held NONE).
    - On delivery, all checkboxes reset to 0 for the next cycle.
    """
    q_pos = np.array(parents["agent_pos"], dtype=float)
    q_ori = np.array(parents["agent_orientation"], dtype=float)
    q_held = np.array(parents["agent_held"], dtype=float)
    q_pot = np.array(parents["pot_state"], dtype=float)
    q_ck1 = np.array(parents["ck_put1"], dtype=float)
    q_ck2 = np.array(parents["ck_put2"], dtype=float)
    q_ck3 = np.array(parents["ck_put3"], dtype=float)
    q_plat = np.array(parents["ck_plated"], dtype=float)
    q_del = np.array(parents["ck_delivered"], dtype=float)

    next_ck1 = np.zeros(2, dtype=float)
    next_ck2 = np.zeros(2, dtype=float)
    next_ck3 = np.zeros(2, dtype=float)
    next_plat = np.zeros(2, dtype=float)
    next_del = np.zeros(2, dtype=float)

    for pos_w in range(model_init.N_WALKABLE):
        for ori in range(model_init.N_DIRECTIONS):
            for held in range(model_init.N_HELD_TYPES):
                for pot in range(model_init.N_POT_STATES):
                    w = q_pos[pos_w] * q_ori[ori] * q_held[held] * q_pot[pot]
                    if w <= 1e-16:
                        continue
                    did_put1, did_put2, did_put3, did_plated, did_deliver = _milestone_this_step(pos_w, ori, held, pot, action)
                    for c1 in range(2):
                        for c2 in range(2):
                            for c3 in range(2):
                                for cp in range(2):
                                    for cd in range(2):
                                        jw = w * q_ck1[c1] * q_ck2[c2] * q_ck3[c3] * q_plat[cp] * q_del[cd]
                                        if jw <= 1e-16:
                                            continue
                                        if did_deliver:
                                            # Global reset on delivery (for this agent's memory)
                                            next_ck1[0] += jw
                                            next_ck2[0] += jw
                                            next_ck3[0] += jw
                                            next_plat[0] += jw
                                            next_del[1] += jw
                                        else:
                                            # Monotonic progress driven by pot_state only:
                                            # pot>=1 -> ck_put1, pot>=2 -> ck_put2, pot==3 -> ck_put3
                                            has_put1 = pot in (model_init.POT_1, model_init.POT_2, model_init.POT_3)
                                            has_put2 = pot in (model_init.POT_2, model_init.POT_3)
                                            has_put3 = pot == model_init.POT_3

                                            n1 = 1 if has_put1 or c1 == 1 else 0
                                            n2 = 1 if has_put2 or c2 == 1 else 0
                                            n3 = 1 if has_put3 or c3 == 1 else 0

                                            # Plate and delivered remain event-based as before
                                            np_ = 1 if did_plated or cp == 1 else 0
                                            nd = cd  # keep delivered flag until we deliver (then reset above)

                                            next_ck1[n1] += jw
                                            next_ck2[n2] += jw
                                            next_ck3[n3] += jw
                                            next_plat[np_] += jw
                                            next_del[nd] += jw

    return {
        "ck_put1": normalize(next_ck1),
        "ck_put2": normalize(next_ck2),
        "ck_put3": normalize(next_ck3),
        "ck_plated": normalize(next_plat),
        "ck_delivered": normalize(next_del),
    }


def B_fn(qs: dict, action: int, B_NOISE_LEVEL: float = 0.03, **kwargs) -> dict[str, np.ndarray]:
    """
    Main transition model p(s' | s, a) for all hidden state factors.

    Parameters
    ----------
    qs : dict[str, np.ndarray]
        Belief over current state factors; keys must match model_init.states.
    action : int
        Primitive action index (0..5) as in model_init (NORTH,SOUTH,EAST,WEST,STAY,INTERACT).
    B_NOISE_LEVEL : float, optional
        Global transition noise level in [0,1]. Each factor's specific transition
        is mixed with a uniform distribution:
            q' = (1 - B_NOISE_LEVEL) * q_specific + B_NOISE_LEVEL * Uniform.

    Returns
    -------
    new_qs : dict[str, np.ndarray]
        Next-step beliefs over all state factors, factorised and normalized.
    """
    action = int(action)
    new_qs: dict[str, np.ndarray] = {}

    for factor, deps in state_state_dependencies.items():
        if factor in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
            if factor == "ck_put1":
                new_qs.update(B_checkboxes(qs, action))
            continue
        parents = {k: qs[k] for k in deps}
        if factor == "agent_pos":
            new_qs[factor] = B_agent_pos(parents, action)
        elif factor == "agent_orientation":
            new_qs[factor] = B_agent_orientation(parents, action)
        elif factor == "agent_held":
            new_qs[factor] = B_agent_held(parents, action)
        elif factor == "pot_state":
            new_qs[factor] = B_pot_state(parents, action)
        else:
            new_qs[factor] = normalize(np.array(parents[factor], dtype=float))

    # Optional global transition noise: mix each factor's next_q with uniform.
    if B_NOISE_LEVEL > 0.0:
        for k in new_qs:
            v = np.array(new_qs[k], dtype=float)
            S = v.shape[0]
            if S > 0:
                uniform = np.ones(S, dtype=float) / float(S)
                new_qs[k] = (1.0 - B_NOISE_LEVEL) * v + B_NOISE_LEVEL * uniform

    # Ensure all outputs are properly normalized numpy arrays
    for k in new_qs:
        new_qs[k] = normalize(np.array(new_qs[k], dtype=float))

    return new_qs

