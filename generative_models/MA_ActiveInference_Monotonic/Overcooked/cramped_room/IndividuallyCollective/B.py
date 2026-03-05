"""
Transition p(s' | s, a) model (B) for IndividuallyCollective paradigm — Cramped Room (Monotonic checkbox model).
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


def B_self_pos(parents: dict, action: int) -> np.ndarray:
    """
    B for self_pos (walkable 0..N_WALKABLE-1). Depends only on previous self_pos.
    Movement is deterministic within walkable cells.
    """
    q_pos = np.array(parents["self_pos"], dtype=float)
    S = q_pos.shape[0]
    next_q = np.zeros(S, dtype=float)

    for w in range(S):
        p = q_pos[w]
        if p <= 1e-16:
            continue
        new_w = _move_walkable(w, action)
        next_q[new_w] += p

    return normalize(next_q)


def B_self_orientation(parents: dict, action: int) -> np.ndarray:
    """
    B for self_orientation. Depends only on previous orientation, updated by action.
    """
    q_ori = np.array(parents["self_orientation"], dtype=float)
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


def B_self_held(parents: dict, action: int) -> np.ndarray:
    """
    B for self_held. Depends on: self_pos, self_orientation, self_held, pot_state.

    INTERACT rules (agent knows layout — front = compute_front_tile_type(pos, ori)):
    - Front ONION + held NONE -> pick onion.
    - Front DISH  + held NONE -> pick dish.
    - Front POT   + held ONION + pot in (0,1,2) -> put onion -> held NONE.
    - Front POT   + held DISH  + pot POT_3 -> take soup -> held SOUP.
    - Front SERVE + held SOUP -> deliver -> held NONE.
    - Front COUNTER + held in (ONION, DISH, SOUP) -> drop -> held NONE.
      (Counters are not explicitly modelled as state; they just clear held.)
    """
    q_pos = np.array(parents["self_pos"], dtype=float)
    q_ori = np.array(parents["self_orientation"], dtype=float)
    q_held = np.array(parents["self_held"], dtype=float)
    q_pot = np.array(parents["pot_state"], dtype=float)

    next_q = np.zeros(model_init.N_HELD_TYPES, dtype=float)

    for pos_w in range(model_init.N_WALKABLE):
        if q_pos[pos_w] <= 1e-16: # if the agent is not in this walkable position, skip
            continue
        for ori in range(model_init.N_DIRECTIONS): #  if the agent is in this position, for each orientation
            if q_ori[ori] <= 1e-16: # if the agent is not in this orientation, skip
                continue
            front = _get_front(pos_w, ori) # if the agent is in this position and orientation, get the front tile type
            for held in range(model_init.N_HELD_TYPES): # for each held type
                if q_held[held] <= 1e-16: # if the agent is not holding this type, skip
                    continue
                for pot in range(model_init.N_POT_STATES):
                    if q_pot[pot] <= 1e-16:
                        continue
                    w = q_pos[pos_w] * q_ori[ori] * q_held[held] * q_pot[pot]
                    if w <= 1e-16:
                        continue #skips this combination because it contributes negligibly to the next-state distribution

                    new_held = held
                    p_success = 1.0
                    if action == model_init.INTERACT:
                        if front == model_init.FRONT_ONION and held == model_init.HELD_NONE:
                            new_held = model_init.HELD_ONION
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_DISH and held == model_init.HELD_NONE:
                            new_held = model_init.HELD_DISH
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_POT and held == model_init.HELD_ONION and pot in (
                            model_init.POT_0,
                            model_init.POT_1,
                            model_init.POT_2,
                        ):
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_POT and held == model_init.HELD_DISH and pot == model_init.POT_3:
                            new_held = model_init.HELD_SOUP
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_SERVE and held == model_init.HELD_SOUP:
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_COUNTER and held in (
                            model_init.HELD_ONION,
                            model_init.HELD_DISH,
                            model_init.HELD_SOUP,
                        ):
                            # TODO: extend model with explicit counter occupancy so we
                            # can prevent dropping onto a full counter (like Independent B_counters).
                            # TODO: also model picking up from counters when held == HELD_NONE.
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                    next_q[new_held] += w * p_success
                    if p_success < 1.0:
                        next_q[held] += w * (1.0 - p_success)

    return normalize(next_q)


def B_pot_state(parents: dict, action: int) -> np.ndarray:
    """
    B for pot_state. Depends on: self_pos, self_orientation, self_held, pot_state.

    - POT_0 -> POT_1 when INTERACT at pot with onion.
    - POT_1 -> POT_2, POT_2 -> POT_3 same.
    - POT_3 -> POT_0 when INTERACT at pot with dish (take soup).
    """
    q_pos = np.array(parents["self_pos"], dtype=float)
    q_ori = np.array(parents["self_orientation"], dtype=float)
    q_held = np.array(parents["self_held"], dtype=float)
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






def B_checkboxes(parents: dict, action: int) -> dict[str, np.ndarray]:
    """
    Predictive belief update for checkbox factors (mean-field, marginal-based).

    Factors:
      - ck_put1/ck_put2/ck_put3: monotonic progress *within a cycle* inferred from pot_state
        (robust shaping for short inference horizons).
      - ck_plated: event-based (plated this step) OR persists until delivery.
      - ck_delivered: a 1-step pulse (preferred observation): becomes 1 iff delivery occurs now,
        otherwise 0. This prevents 'satisfice after first delivery' in long episodes.

    Reset semantics:
      - On delivery (this step), ck_put1/2/3 and ck_plated reset to 0 for the next cycle.
      - ck_delivered is a pulse so it resets automatically next step unless another delivery occurs.

    Requires:
      - pot_state uses POT_0..POT_3 where POT_3 means ready.
      - _get_front(pos_w, ori) returns FRONT_POT / FRONT_SERVE, etc.
    """
    q_pos = np.asarray(parents["self_pos"], dtype=float)
    q_ori = np.asarray(parents["self_orientation"], dtype=float)
    q_held = np.asarray(parents["self_held"], dtype=float)
    q_pot = np.asarray(parents["pot_state"], dtype=float)

    q_ck1 = np.asarray(parents["ck_put1"], dtype=float)
    q_ck2 = np.asarray(parents["ck_put2"], dtype=float)
    q_ck3 = np.asarray(parents["ck_put3"], dtype=float)
    q_plat = np.asarray(parents["ck_plated"], dtype=float)

    # ------------------------------------------------------------
    # 1) Compute probability that plating/delivery happens *this step*
    #    under current belief over (pos, ori, held, pot).
    # ------------------------------------------------------------
    p_deliver_now = 0.0
    p_plated_now = 0.0

    if action == model_init.INTERACT:
        for pos_w in range(model_init.N_WALKABLE):
            qp = q_pos[pos_w]
            if qp <= 1e-16:
                continue
            for ori in range(model_init.N_DIRECTIONS):
                qo = q_ori[ori]
                if qo <= 1e-16:
                    continue

                front = _get_front(pos_w, ori)
                base = qp * qo

                if front == model_init.FRONT_SERVE:
                    p_deliver_now += base * q_held[model_init.HELD_SOUP]

                elif front == model_init.FRONT_POT:
                    p_plated_now += (
                        base
                        * q_held[model_init.HELD_DISH]
                        * q_pot[model_init.POT_3]
                    )

    p_deliver_now = float(np.clip(p_deliver_now, 0.0, 1.0))
    p_plated_now = float(np.clip(p_plated_now, 0.0, 1.0))

    # ------------------------------------------------------------
    # 2) Pot-derived progress probabilities (robust shaping).
    # ------------------------------------------------------------
    p_has_put1 = float(q_pot[model_init.POT_1] + q_pot[model_init.POT_2] + q_pot[model_init.POT_3])
    p_has_put2 = float(q_pot[model_init.POT_2] + q_pot[model_init.POT_3])
    p_has_put3 = float(q_pot[model_init.POT_3])

    # ------------------------------------------------------------
    # 3) Predict next checkbox beliefs with soft reset on delivery.
    # ------------------------------------------------------------
    p_ck1_next_1_if_no_del = 1.0 - (1.0 - q_ck1[1]) * (1.0 - p_has_put1)
    p_ck2_next_1_if_no_del = 1.0 - (1.0 - q_ck2[1]) * (1.0 - p_has_put2)
    p_ck3_next_1_if_no_del = 1.0 - (1.0 - q_ck3[1]) * (1.0 - p_has_put3)

    p_ck1_next_1 = (1.0 - p_deliver_now) * p_ck1_next_1_if_no_del
    p_ck2_next_1 = (1.0 - p_deliver_now) * p_ck2_next_1_if_no_del
    p_ck3_next_1 = (1.0 - p_deliver_now) * p_ck3_next_1_if_no_del

    p_plat_next_1_if_no_del = 1.0 - (1.0 - q_plat[1]) * (1.0 - p_plated_now)
    p_plat_next_1 = (1.0 - p_deliver_now) * p_plat_next_1_if_no_del

    p_del_next_1 = p_deliver_now

    next_ck1 = np.array([1.0 - p_ck1_next_1, p_ck1_next_1], dtype=float)
    next_ck2 = np.array([1.0 - p_ck2_next_1, p_ck2_next_1], dtype=float)
    next_ck3 = np.array([1.0 - p_ck3_next_1, p_ck3_next_1], dtype=float)
    next_plat = np.array([1.0 - p_plat_next_1, p_plat_next_1], dtype=float)
    next_del = np.array([1.0 - p_del_next_1, p_del_next_1], dtype=float)

    return {
        "ck_put1": normalize(next_ck1),
        "ck_put2": normalize(next_ck2),
        "ck_put3": normalize(next_ck3),
        "ck_plated": normalize(next_plat),
        "ck_delivered": normalize(next_del),
    }




def B_fn(qs: dict, action: int, B_NOISE_LEVEL: float = 0.0, **kwargs) -> dict:
    """
    Main transition model p(s' | s, a) for all hidden state factors.
    """
    action = int(action)
    new_qs: dict[str, np.ndarray] = {}

    for factor, deps in state_state_dependencies.items():
        if factor in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
            if factor == "ck_put1":
                new_qs.update(B_checkboxes(qs, action))
            continue
        parents = {k: qs[k] for k in deps}
        if factor == "self_pos":
            new_qs[factor] = B_self_pos(parents, action)
        elif factor == "self_orientation":
            new_qs[factor] = B_self_orientation(parents, action)
        elif factor == "self_held":
            new_qs[factor] = B_self_held(parents, action)
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
