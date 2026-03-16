# B.py
"""
Transition p(s' | s, a) model (B) for IndividuallyCollective paradigm — Cramped Room.

Policy semantics:
- one policy step is (actor, primitive_action)
- if SELF acts, OTHER is STAY
- if OTHER acts, SELF is STAY

This B uses both effective actions:
    self_action, other_action = model_init.policy_step_to_actions(actor, action)
"""

import numpy as np
from . import model_init

state_state_dependencies = model_init.state_state_dependencies


def normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    return p / max(s, 1e-8)


def _decode_policy_step(step) -> tuple[int, int]:
    if isinstance(step, np.ndarray) and step.shape == (2,):
        actor, action = int(step[0]), int(step[1])
        return model_init.policy_step_to_actions(actor, action)

    if isinstance(step, (tuple, list)) and len(step) == 2:
        actor, action = int(step[0]), int(step[1])
        return model_init.policy_step_to_actions(actor, action)

    # Scalar encoding support:
    # - 0..5   => (SELF, primitive_action)
    # - 6..11  => (OTHER, primitive_action)
    a = int(step)
    n_actions = int(getattr(model_init, "N_ACTIONS", 6))
    n_actors = int(getattr(model_init, "N_ACTORS", 2))
    if 0 <= a < n_actions * n_actors:
        actor = a // n_actions
        action = a % n_actions
        return model_init.policy_step_to_actions(actor, action)

    # Back-compat: treat as primitive self action
    return a, model_init.STAY


def _update_orientation(ori_idx: int, action: int) -> int:
    if action in (model_init.NORTH, model_init.SOUTH, model_init.EAST, model_init.WEST):
        return int(action)
    return int(ori_idx)


def _move_walkable(walkable_idx: int, action: int) -> int:
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


def _get_front(pos_w: int, ori: int) -> int:
    return model_init.compute_front_tile_type(pos_w, ori)


INTERACT_SUCCESS_PROB = getattr(model_init, "INTERACT_SUCCESS_PROB", 0.9)

def _held_to_ctr_content(held: int) -> int:
    if held == model_init.HELD_ONION:
        return model_init.CTR_ONION
    if held == model_init.HELD_DISH:
        return model_init.CTR_DISH
    if held == model_init.HELD_SOUP:
        return model_init.CTR_SOUP
    return model_init.CTR_EMPTY


def _ctr_content_to_held(ctr_state: int) -> int:
    if ctr_state == model_init.CTR_ONION:
        return model_init.HELD_ONION
    if ctr_state == model_init.CTR_DISH:
        return model_init.HELD_DISH
    if ctr_state == model_init.CTR_SOUP:
        return model_init.HELD_SOUP
    return model_init.HELD_NONE


def B_self_pos(parents: dict, self_action: int) -> np.ndarray:
    q_pos = np.array(parents["self_pos"], dtype=float)
    q_other_pos = np.array(parents["other_pos"], dtype=float)

    S = q_pos.shape[0]
    next_q = np.zeros(S, dtype=float)

    for w in range(S):
        p = q_pos[w]
        if p <= 1e-16:
            continue

        intended_w = _move_walkable(w, self_action)
        p_blocked = float(q_other_pos[intended_w]) if 0 <= intended_w < S else 0.0
        p_blocked = float(np.clip(p_blocked, 0.0, 1.0))

        if intended_w == w:
            next_q[w] += p
        else:
            next_q[intended_w] += p * (1.0 - p_blocked)
            next_q[w] += p * p_blocked

    return normalize(next_q)


def B_self_orientation(parents: dict, self_action: int) -> np.ndarray:
    q_ori = np.array(parents["self_orientation"], dtype=float)
    next_q = np.zeros(model_init.N_DIRECTIONS, dtype=float)

    for ori in range(model_init.N_DIRECTIONS):
        p = q_ori[ori]
        if p <= 1e-16:
            continue
        next_q[_update_orientation(ori, self_action)] += p

    return normalize(next_q)


def B_self_held(parents: dict, self_action: int) -> np.ndarray:
    q_pos = np.array(parents["self_pos"], dtype=float)
    q_ori = np.array(parents["self_orientation"], dtype=float)
    q_held = np.array(parents["self_held"], dtype=float)
    q_pot = np.array(parents["pot_state"], dtype=float)

    q_ctr = {g: np.array(parents[f"ctr_{g}"], dtype=float) for g in model_init.MODELED_COUNTERS}
    next_q = np.zeros(model_init.N_HELD_TYPES, dtype=float)

    for pos_w in range(model_init.N_WALKABLE):
        if q_pos[pos_w] <= 1e-16:
            continue
        for ori in range(model_init.N_DIRECTIONS):
            if q_ori[ori] <= 1e-16:
                continue
            front = _get_front(pos_w, ori)
            front_ctr = model_init.modeled_counter_in_front(pos_w, ori)

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

                    if self_action == model_init.INTERACT:
                        if front == model_init.FRONT_ONION and held == model_init.HELD_NONE:
                            new_held = model_init.HELD_ONION
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_DISH and held == model_init.HELD_NONE:
                            new_held = model_init.HELD_DISH
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_POT and held == model_init.HELD_ONION and pot in (
                            model_init.POT_0, model_init.POT_1, model_init.POT_2
                        ):
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_POT and held == model_init.HELD_DISH and pot == model_init.POT_3:
                            new_held = model_init.HELD_SOUP
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_SERVE and held == model_init.HELD_SOUP:
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_COUNTER and front_ctr is not None:
                            if held != model_init.HELD_NONE:
                                p_empty = float(q_ctr[front_ctr][model_init.CTR_EMPTY])
                                next_q[model_init.HELD_NONE] += w * INTERACT_SUCCESS_PROB * p_empty
                                next_q[held] += w * (1.0 - INTERACT_SUCCESS_PROB * p_empty)
                                continue
                            else:
                                # Pick up from counter if it has a known item type.
                                qc = np.asarray(q_ctr[front_ctr], dtype=float)
                                # If counter is empty, interaction changes nothing (stay NONE).
                                p_empty = float(qc[model_init.CTR_EMPTY]) if qc.size else 1.0
                                if p_empty > 0.0:
                                    next_q[model_init.HELD_NONE] += w * p_empty
                                for cs in range(model_init.N_CTR_STATES):
                                    if cs == model_init.CTR_EMPTY:
                                        continue
                                    p_cs = float(qc[cs])
                                    if p_cs <= 1e-16:
                                        continue
                                    held_item = _ctr_content_to_held(cs)
                                    next_q[held_item] += w * INTERACT_SUCCESS_PROB * p_cs
                                    next_q[model_init.HELD_NONE] += w * (1.0 - INTERACT_SUCCESS_PROB) * p_cs
                                # Counter empty case falls through to default below.
                                if float(qc[model_init.CTR_EMPTY]) < 1.0:
                                    continue

                    next_q[new_held] += w * p_success
                    if p_success < 1.0:
                        next_q[held] += w * (1.0 - p_success)

    return normalize(next_q)


def B_other_pos(parents: dict, other_action: int) -> np.ndarray:
    q_other = np.array(parents["other_pos"], dtype=float)
    q_self_pos = np.array(parents["self_pos"], dtype=float)

    S = q_other.shape[0]
    next_q = np.zeros(S, dtype=float)

    for w in range(S):
        p = q_other[w]
        if p <= 1e-16:
            continue

        intended_w = _move_walkable(w, other_action)
        p_blocked = float(q_self_pos[intended_w]) if 0 <= intended_w < S else 0.0
        p_blocked = float(np.clip(p_blocked, 0.0, 1.0))

        if intended_w == w:
            next_q[w] += p
        else:
            next_q[intended_w] += p * (1.0 - p_blocked)
            next_q[w] += p * p_blocked

    return normalize(next_q)


def B_other_orientation(parents: dict, other_action: int) -> np.ndarray:
    q_ori = np.array(parents["other_orientation"], dtype=float)
    next_q = np.zeros(model_init.N_DIRECTIONS, dtype=float)

    for ori in range(model_init.N_DIRECTIONS):
        p = q_ori[ori]
        if p <= 1e-16:
            continue
        next_q[_update_orientation(ori, other_action)] += p

    return normalize(next_q)


def B_other_held(parents: dict, other_action: int) -> np.ndarray:
    q_pos = np.array(parents["other_pos"], dtype=float)
    q_ori = np.array(parents["other_orientation"], dtype=float)
    q_held = np.array(parents["other_held"], dtype=float)
    q_pot = np.array(parents["pot_state"], dtype=float)

    q_ctr = {g: np.array(parents[f"ctr_{g}"], dtype=float) for g in model_init.MODELED_COUNTERS}
    next_q = np.zeros(model_init.N_HELD_TYPES, dtype=float)

    for pos_w in range(model_init.N_WALKABLE):
        if q_pos[pos_w] <= 1e-16:
            continue
        for ori in range(model_init.N_DIRECTIONS):
            if q_ori[ori] <= 1e-16:
                continue
            front = _get_front(pos_w, ori)
            front_ctr = model_init.modeled_counter_in_front(pos_w, ori)

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

                    if other_action == model_init.INTERACT:
                        if front == model_init.FRONT_ONION and held == model_init.HELD_NONE:
                            new_held = model_init.HELD_ONION
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_DISH and held == model_init.HELD_NONE:
                            new_held = model_init.HELD_DISH
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_POT and held == model_init.HELD_ONION and pot in (
                            model_init.POT_0, model_init.POT_1, model_init.POT_2
                        ):
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_POT and held == model_init.HELD_DISH and pot == model_init.POT_3:
                            new_held = model_init.HELD_SOUP
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_SERVE and held == model_init.HELD_SOUP:
                            new_held = model_init.HELD_NONE
                            p_success = INTERACT_SUCCESS_PROB
                        elif front == model_init.FRONT_COUNTER and front_ctr is not None:
                            if held != model_init.HELD_NONE:
                                p_empty = float(q_ctr[front_ctr][model_init.CTR_EMPTY])
                                next_q[model_init.HELD_NONE] += w * INTERACT_SUCCESS_PROB * p_empty
                                next_q[held] += w * (1.0 - INTERACT_SUCCESS_PROB * p_empty)
                                continue
                            else:
                                qc = np.asarray(q_ctr[front_ctr], dtype=float)
                                p_empty = float(qc[model_init.CTR_EMPTY]) if qc.size else 1.0
                                if p_empty > 0.0:
                                    next_q[model_init.HELD_NONE] += w * p_empty
                                for cs in range(model_init.N_CTR_STATES):
                                    if cs == model_init.CTR_EMPTY:
                                        continue
                                    p_cs = float(qc[cs])
                                    if p_cs <= 1e-16:
                                        continue
                                    held_item = _ctr_content_to_held(cs)
                                    next_q[held_item] += w * INTERACT_SUCCESS_PROB * p_cs
                                    next_q[model_init.HELD_NONE] += w * (1.0 - INTERACT_SUCCESS_PROB) * p_cs
                                if float(qc[model_init.CTR_EMPTY]) < 1.0:
                                    continue

                    next_q[new_held] += w * p_success
                    if p_success < 1.0:
                        next_q[held] += w * (1.0 - p_success)

    return normalize(next_q)


def _apply_pot_interaction(pot: int, held: int, action: int, front_tile: int) -> tuple[int, float]:
    new_pot = pot
    p_success = 1.0

    if action == model_init.INTERACT and front_tile == model_init.FRONT_POT:
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

    return new_pot, p_success


def B_pot_state(parents: dict, self_action: int, other_action: int) -> np.ndarray:
    q_sp = np.array(parents["self_pos"], dtype=float)
    q_so = np.array(parents["self_orientation"], dtype=float)
    q_sh = np.array(parents["self_held"], dtype=float)

    q_op = np.array(parents["other_pos"], dtype=float)
    q_oo = np.array(parents["other_orientation"], dtype=float)
    q_oh = np.array(parents["other_held"], dtype=float)

    q_pot = np.array(parents["pot_state"], dtype=float)
    next_q = np.zeros(model_init.N_POT_STATES, dtype=float)

    for pot in range(model_init.N_POT_STATES):
        if q_pot[pot] <= 1e-16:
            continue

        for sp in range(model_init.N_WALKABLE):
            if q_sp[sp] <= 1e-16:
                continue
            for so in range(model_init.N_DIRECTIONS):
                if q_so[so] <= 1e-16:
                    continue
                s_front = _get_front(sp, so)
                for sh in range(model_init.N_HELD_TYPES):
                    if q_sh[sh] <= 1e-16:
                        continue

                    base_self = q_pot[pot] * q_sp[sp] * q_so[so] * q_sh[sh]
                    if base_self <= 1e-16:
                        continue

                    pot_after_self, p_self_success = _apply_pot_interaction(pot, sh, self_action, s_front)
                    self_branches = [(pot_after_self, p_self_success)]
                    if p_self_success < 1.0:
                        self_branches.append((pot, 1.0 - p_self_success))

                    for op in range(model_init.N_WALKABLE):
                        if q_op[op] <= 1e-16:
                            continue
                        for oo in range(model_init.N_DIRECTIONS):
                            if q_oo[oo] <= 1e-16:
                                continue
                            o_front = _get_front(op, oo)
                            for oh in range(model_init.N_HELD_TYPES):
                                if q_oh[oh] <= 1e-16:
                                    continue

                                base = base_self * q_op[op] * q_oo[oo] * q_oh[oh]
                                if base <= 1e-16:
                                    continue

                                for pot_mid, p_mid in self_branches:
                                    if p_mid <= 1e-16:
                                        continue

                                    pot_after_other, p_other_success = _apply_pot_interaction(
                                        pot_mid, oh, other_action, o_front
                                    )

                                    next_q[pot_after_other] += base * p_mid * p_other_success
                                    if p_other_success < 1.0:
                                        next_q[pot_mid] += base * p_mid * (1.0 - p_other_success)

    return normalize(next_q)


def _apply_counter_fill(ctr_state: int, held: int, action: int, front_ctr, counter_grid: int) -> tuple[int, float]:
    """
    Apply a single agent's INTERACT effect on one counter:
    - Place: held!=NONE and counter EMPTY -> counter becomes that item
    - Pickup: held==NONE and counter has item -> counter becomes EMPTY
    Returns (new_ctr_state, p_success_of_change). If no change possible, p_success=1.0.
    """
    new_ctr = int(ctr_state)
    p_success = 1.0

    if not (action == model_init.INTERACT and front_ctr == counter_grid):
        return new_ctr, p_success

    # Place onto empty counter
    if held != model_init.HELD_NONE and int(ctr_state) == int(model_init.CTR_EMPTY):
        new_ctr = _held_to_ctr_content(int(held))
        p_success = INTERACT_SUCCESS_PROB if new_ctr != model_init.CTR_EMPTY else 1.0
        return new_ctr, p_success

    # Pick up from non-empty counter
    if held == model_init.HELD_NONE and int(ctr_state) != int(model_init.CTR_EMPTY):
        new_ctr = model_init.CTR_EMPTY
        p_success = INTERACT_SUCCESS_PROB
        return new_ctr, p_success

    return new_ctr, p_success


def B_counter_occupancy(parents: dict, self_action: int, other_action: int, counter_grid: int) -> np.ndarray:
    q_ctr = np.asarray(parents[f"ctr_{counter_grid}"], dtype=float)

    q_sp = np.asarray(parents["self_pos"], dtype=float)
    q_so = np.asarray(parents["self_orientation"], dtype=float)
    q_sh = np.asarray(parents["self_held"], dtype=float)

    q_op = np.asarray(parents["other_pos"], dtype=float)
    q_oo = np.asarray(parents["other_orientation"], dtype=float)
    q_oh = np.asarray(parents["other_held"], dtype=float)

    next_q = np.zeros(model_init.N_CTR_STATES, dtype=float)

    for ctr_state in range(model_init.N_CTR_STATES):
        pc = q_ctr[ctr_state]
        if pc <= 1e-16:
            continue

        for sp in range(model_init.N_WALKABLE):
            if q_sp[sp] <= 1e-16:
                continue
            for so in range(model_init.N_DIRECTIONS):
                if q_so[so] <= 1e-16:
                    continue
                s_front_ctr = model_init.modeled_counter_in_front(sp, so)
                for sh in range(model_init.N_HELD_TYPES):
                    if q_sh[sh] <= 1e-16:
                        continue

                    base_self = pc * q_sp[sp] * q_so[so] * q_sh[sh]
                    if base_self <= 1e-16:
                        continue

                    ctr_after_self, p_self_success = _apply_counter_fill(
                        ctr_state, sh, self_action, s_front_ctr, counter_grid
                    )
                    self_branches = [(ctr_after_self, p_self_success)]
                    if p_self_success < 1.0:
                        self_branches.append((ctr_state, 1.0 - p_self_success))

                    for op in range(model_init.N_WALKABLE):
                        if q_op[op] <= 1e-16:
                            continue
                        for oo in range(model_init.N_DIRECTIONS):
                            if q_oo[oo] <= 1e-16:
                                continue
                            o_front_ctr = model_init.modeled_counter_in_front(op, oo)
                            for oh in range(model_init.N_HELD_TYPES):
                                if q_oh[oh] <= 1e-16:
                                    continue

                                base = base_self * q_op[op] * q_oo[oo] * q_oh[oh]
                                if base <= 1e-16:
                                    continue

                                for ctr_mid, p_mid in self_branches:
                                    if p_mid <= 1e-16:
                                        continue

                                    ctr_after_other, p_other_success = _apply_counter_fill(
                                        ctr_mid, oh, other_action, o_front_ctr, counter_grid
                                    )

                                    next_q[ctr_after_other] += base * p_mid * p_other_success
                                    if p_other_success < 1.0:
                                        next_q[ctr_mid] += base * p_mid * (1.0 - p_other_success)

    return normalize(next_q)


def B_checkboxes(parents: dict, self_action: int, other_action: int) -> dict[str, np.ndarray]:
    """
    Event semantics:
    - ck_delivered = probability delivery happened this imagined step
    - ck_plated    = probability plating happened this imagined step if no delivery reset
    - ck_put1/2/3  remain pot-driven as before
    """
    q_sp = np.asarray(parents["self_pos"], dtype=float)
    q_so = np.asarray(parents["self_orientation"], dtype=float)
    q_sh = np.asarray(parents["self_held"], dtype=float)

    q_op = np.asarray(parents["other_pos"], dtype=float)
    q_oo = np.asarray(parents["other_orientation"], dtype=float)
    q_oh = np.asarray(parents["other_held"], dtype=float)

    q_pot = np.asarray(parents["pot_state"], dtype=float)

    q_ck1 = np.asarray(parents["ck_put1"], dtype=float)
    q_ck2 = np.asarray(parents["ck_put2"], dtype=float)
    q_ck3 = np.asarray(parents["ck_put3"], dtype=float)
    q_plat = np.asarray(parents["ck_plated"], dtype=float)

    p_deliver_now = 0.0
    p_plated_now = 0.0

    if self_action == model_init.INTERACT:
        for sp in range(model_init.N_WALKABLE):
            if q_sp[sp] <= 1e-16:
                continue
            for so in range(model_init.N_DIRECTIONS):
                if q_so[so] <= 1e-16:
                    continue
                front = _get_front(sp, so)
                base = q_sp[sp] * q_so[so]

                if front == model_init.FRONT_SERVE:
                    p_deliver_now += base * q_sh[model_init.HELD_SOUP]
                elif front == model_init.FRONT_POT:
                    p_plated_now += base * q_sh[model_init.HELD_DISH] * q_pot[model_init.POT_3]

    if other_action == model_init.INTERACT:
        for op in range(model_init.N_WALKABLE):
            if q_op[op] <= 1e-16:
                continue
            for oo in range(model_init.N_DIRECTIONS):
                if q_oo[oo] <= 1e-16:
                    continue
                front = _get_front(op, oo)
                base = q_op[op] * q_oo[oo]

                if front == model_init.FRONT_SERVE:
                    p_deliver_now += base * q_oh[model_init.HELD_SOUP]
                elif front == model_init.FRONT_POT:
                    p_plated_now += base * q_oh[model_init.HELD_DISH] * q_pot[model_init.POT_3]

    p_deliver_now = float(np.clip(p_deliver_now, 0.0, 1.0))
    p_plated_now = float(np.clip(p_plated_now, 0.0, 1.0))

    p_has_put1 = float(q_pot[model_init.POT_1] + q_pot[model_init.POT_2] + q_pot[model_init.POT_3])
    p_has_put2 = float(q_pot[model_init.POT_2] + q_pot[model_init.POT_3])
    p_has_put3 = float(q_pot[model_init.POT_3])

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


def B_fn(qs: dict, action, B_NOISE_LEVEL: float = 0.0, **kwargs) -> dict:
    """
    Main transition model p(s' | s, a) for all hidden state factors.

    `action` can be either:
      - old primitive int action
      - new policy step tuple: (actor, primitive_action)
    """
    self_action, other_action = _decode_policy_step(action)
    new_qs: dict[str, np.ndarray] = {}

    for factor, deps in state_state_dependencies.items():
        if factor in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
            if factor == "ck_put1":
                new_qs.update(B_checkboxes(qs, self_action, other_action))
            continue

        parents = {k: qs[k] for k in deps}

        if factor == "self_pos":
            new_qs[factor] = B_self_pos(parents, self_action)
        elif factor == "self_orientation":
            new_qs[factor] = B_self_orientation(parents, self_action)
        elif factor == "self_held":
            new_qs[factor] = B_self_held(parents, self_action)
        elif factor == "other_pos":
            new_qs[factor] = B_other_pos(parents, other_action)
        elif factor == "other_orientation":
            new_qs[factor] = B_other_orientation(parents, other_action)
        elif factor == "other_held":
            new_qs[factor] = B_other_held(parents, other_action)
        elif factor == "pot_state":
            new_qs[factor] = B_pot_state(parents, self_action, other_action)
        elif factor.startswith("ctr_"):
            grid = int(factor.split("_")[1])
            new_qs[factor] = B_counter_occupancy(parents, self_action, other_action, grid)
        else:
            new_qs[factor] = normalize(np.array(parents[factor], dtype=float))

    if B_NOISE_LEVEL > 0.0:
        for k in new_qs:
            v = np.array(new_qs[k], dtype=float)
            S = v.shape[0]
            if S > 0:
                uniform = np.ones(S, dtype=float) / float(S)
                new_qs[k] = (1.0 - B_NOISE_LEVEL) * v + B_NOISE_LEVEL * uniform

    for k in new_qs:
        new_qs[k] = normalize(np.array(new_qs[k], dtype=float))

    return new_qs