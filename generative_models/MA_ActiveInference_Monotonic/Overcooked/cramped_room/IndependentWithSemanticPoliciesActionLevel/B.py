# B.py
"""
Transition model p(s' | s, a) for the Independent paradigm — Cramped Room.

Independent semantics:
- Each agent plans only over its own semantic actions.
- The other agent is treated as part of the environment: its pos/orientation/held
  are observed and inferred, but NOT controlled or co-planned.
- During policy rollout, the other agent's state factors are held fixed
  (identity transition). Only the ego agent's actions drive state changes.
- Shared factors (pot, counters, checkboxes) are updated by the ego agent's
  terminal action only; the other agent is assumed to STAY/not interact.

Policy step formats accepted by B_fn:
  - scalar int: self semantic action index 0..N_ACTIONS-1
  - (PRIMITIVE_POLICY_STEP, a_self): primitive ego action, no teleport
"""

import numpy as np
from . import model_init

state_state_dependencies = model_init.state_state_dependencies


def normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    return p / max(s, 1e-8)


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ego-agent (self) state transitions
# ---------------------------------------------------------------------------

def B_self_pos(parents: dict, self_action: int) -> np.ndarray:
    """Ego position transition with collision avoidance against inferred other pos."""
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


# ---------------------------------------------------------------------------
# Other-agent state transitions — identity (autonomous, uncontrolled)
#
# The independent agent does not model or control the other agent's behaviour.
# During planning, the other agent's latent states are held fixed; only
# perception/inference updates them via observations through A.
# ---------------------------------------------------------------------------

def B_other_pos(parents: dict) -> np.ndarray:
    return normalize(np.array(parents["other_pos"], dtype=float))


def B_other_orientation(parents: dict) -> np.ndarray:
    return normalize(np.array(parents["other_orientation"], dtype=float))


def B_other_held(parents: dict) -> np.ndarray:
    return normalize(np.array(parents["other_held"], dtype=float))


# ---------------------------------------------------------------------------
# Shared environment factors — driven by ego agent only (other assumed STAY)
# ---------------------------------------------------------------------------

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


def B_pot_state(parents: dict, self_action: int) -> np.ndarray:
    """Pot transition driven by ego agent only; other agent assumed STAY."""
    q_sp = np.array(parents["self_pos"], dtype=float)
    q_so = np.array(parents["self_orientation"], dtype=float)
    q_sh = np.array(parents["self_held"], dtype=float)
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
                    base = q_pot[pot] * q_sp[sp] * q_so[so] * q_sh[sh]
                    if base <= 1e-16:
                        continue
                    new_pot, p_success = _apply_pot_interaction(pot, sh, self_action, s_front)
                    next_q[new_pot] += base * p_success
                    if p_success < 1.0:
                        next_q[pot] += base * (1.0 - p_success)

    return normalize(next_q)


def _apply_counter_fill(
    ctr_state: int, held: int, action: int, front_ctr, counter_grid: int
) -> tuple[int, float]:
    new_ctr = int(ctr_state)
    p_success = 1.0

    if not (action == model_init.INTERACT and front_ctr == counter_grid):
        return new_ctr, p_success

    if held != model_init.HELD_NONE and int(ctr_state) == int(model_init.CTR_EMPTY):
        new_ctr = _held_to_ctr_content(int(held))
        p_success = INTERACT_SUCCESS_PROB if new_ctr != model_init.CTR_EMPTY else 1.0
        return new_ctr, p_success

    if held == model_init.HELD_NONE and int(ctr_state) != int(model_init.CTR_EMPTY):
        new_ctr = model_init.CTR_EMPTY
        p_success = INTERACT_SUCCESS_PROB
        return new_ctr, p_success

    return new_ctr, p_success


def B_counter_occupancy(parents: dict, self_action: int, counter_grid: int) -> np.ndarray:
    """Counter transition driven by ego agent only; other agent assumed STAY."""
    q_ctr = np.asarray(parents[f"ctr_{counter_grid}"], dtype=float)
    q_sp = np.asarray(parents["self_pos"], dtype=float)
    q_so = np.asarray(parents["self_orientation"], dtype=float)
    q_sh = np.asarray(parents["self_held"], dtype=float)
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
                    base = pc * q_sp[sp] * q_so[so] * q_sh[sh]
                    if base <= 1e-16:
                        continue
                    ctr_after, p_success = _apply_counter_fill(
                        ctr_state, sh, self_action, s_front_ctr, counter_grid
                    )
                    next_q[ctr_after] += base * p_success
                    if p_success < 1.0:
                        next_q[ctr_state] += base * (1.0 - p_success)

    return normalize(next_q)


def B_checkboxes(
    parents: dict,
    self_action: int,
    q_pot_next: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Monotonic checkbox transitions driven by ego agent only.

    - ck_put1/2/3: sticky toward 1 when pot mass crosses onion-count thresholds.
    - ck_plated: sticky toward 1 when self runs INTERACT facing pot while holding dish
      and pot is believed ready (POT_3). Masked by delivery reset.
    - ck_delivered: sticky toward 1 once self executes a delivery INTERACT.
    """
    q_sp = np.asarray(parents["self_pos"], dtype=float)
    q_so = np.asarray(parents["self_orientation"], dtype=float)
    q_sh = np.asarray(parents["self_held"], dtype=float)
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

    p_deliver_now = float(np.clip(p_deliver_now, 0.0, 1.0))
    p_plated_now = float(np.clip(p_plated_now, 0.0, 1.0))

    if q_pot_next is None:
        pot_parents = {
            "self_pos": q_sp,
            "self_orientation": q_so,
            "self_held": q_sh,
            "pot_state": q_pot,
        }
        q_pot_next = B_pot_state(pot_parents, self_action)
    else:
        q_pot_next = np.asarray(q_pot_next, dtype=float)

    p_has_put1 = float(q_pot_next[model_init.POT_1] + q_pot_next[model_init.POT_2] + q_pot_next[model_init.POT_3])
    p_has_put2 = float(q_pot_next[model_init.POT_2] + q_pot_next[model_init.POT_3])
    p_has_put3 = float(q_pot_next[model_init.POT_3])

    p_ck1_next_1 = (1.0 - p_deliver_now) * (1.0 - (1.0 - q_ck1[1]) * (1.0 - p_has_put1))
    p_ck2_next_1 = (1.0 - p_deliver_now) * (1.0 - (1.0 - q_ck2[1]) * (1.0 - p_has_put2))
    p_ck3_next_1 = (1.0 - p_deliver_now) * (1.0 - (1.0 - q_ck3[1]) * (1.0 - p_has_put3))
    p_plat_next_1 = (1.0 - p_deliver_now) * (1.0 - (1.0 - q_plat[1]) * (1.0 - p_plated_now))
    p_del_next_1 = float(np.clip(p_deliver_now, 0.0, 1.0))

    return {
        "ck_put1": normalize(np.array([1.0 - p_ck1_next_1, p_ck1_next_1], dtype=float)),
        "ck_put2": normalize(np.array([1.0 - p_ck2_next_1, p_ck2_next_1], dtype=float)),
        "ck_put3": normalize(np.array([1.0 - p_ck3_next_1, p_ck3_next_1], dtype=float)),
        "ck_plated": normalize(np.array([1.0 - p_plat_next_1, p_plat_next_1], dtype=float)),
        "ck_delivered": normalize(np.array([1.0 - p_del_next_1, p_del_next_1], dtype=float)),
    }


# ---------------------------------------------------------------------------
# Primitive-step rollout (no teleport)
# ---------------------------------------------------------------------------

def _try_primitive_policy_step(action) -> int | None:
    """
    Return the primitive ego action if action is a (PRIMITIVE_POLICY_STEP, a_self) tuple.
    Only the ego action is extracted; the independent agent carries no other-agent action.
    """
    lab = getattr(model_init, "PRIMITIVE_POLICY_STEP", None)
    if lab is None:
        return None
    if isinstance(action, (tuple, list)) and len(action) >= 2 and action[0] == lab:
        return int(action[1])
    return None


def B_fn_primitive_step(
    qs: dict,
    self_action: int,
    B_NOISE_LEVEL: float = 0.0,
    **kwargs,
) -> dict:
    """
    Single-timestep transition using a primitive ego action.
    Other agent's state factors are held as identity (autonomous/uncontrolled).
    """
    del kwargs
    parents = {k: np.asarray(qs[k], dtype=float) for k in qs}

    pot_next = B_pot_state(
        {k: parents[k] for k in state_state_dependencies["pot_state"]},
        self_action,
    )

    new_qs: dict[str, np.ndarray] = {}

    for factor, deps in state_state_dependencies.items():
        if factor in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
            if factor == "ck_put1":
                new_qs.update(B_checkboxes(parents, self_action, q_pot_next=pot_next))
            continue

        pdeps = {k: parents[k] for k in deps}

        if factor == "self_pos":
            new_qs[factor] = B_self_pos(pdeps, self_action)
        elif factor == "self_orientation":
            new_qs[factor] = B_self_orientation(pdeps, self_action)
        elif factor == "self_held":
            new_qs[factor] = B_self_held(pdeps, self_action)
        elif factor == "other_pos":
            new_qs[factor] = B_other_pos(pdeps)
        elif factor == "other_orientation":
            new_qs[factor] = B_other_orientation(pdeps)
        elif factor == "other_held":
            new_qs[factor] = B_other_held(pdeps)
        elif factor == "pot_state":
            new_qs[factor] = np.asarray(pot_next, dtype=float)
        elif factor.startswith("ctr_"):
            grid = int(factor.split("_")[1])
            new_qs[factor] = B_counter_occupancy(pdeps, self_action, grid)
        else:
            new_qs[factor] = normalize(np.asarray(parents[factor], dtype=float))

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


# ---------------------------------------------------------------------------
# Main B_fn — semantic macro-action for ego only
# ---------------------------------------------------------------------------

def B_fn(qs: dict, action, B_NOISE_LEVEL: float = 0.0, **kwargs) -> dict:
    """
    Transition model p(s' | s, a) for the independent agent.

    `action` can be:
      - scalar int: self semantic action index 0..N_ACTIONS-1
        → teleport self to target pose, apply terminal action, hold other fixed
      - (PRIMITIVE_POLICY_STEP, a_self): primitive ego action, no teleport

    The other agent's state factors are always held fixed (identity transition).
    Only the ego agent's action changes the world model.
    """
    prim = _try_primitive_policy_step(action)
    if prim is not None:
        return B_fn_primitive_step(qs, prim, B_NOISE_LEVEL=B_NOISE_LEVEL, **kwargs)

    # Decode scalar self semantic action
    if isinstance(action, (tuple, list)):
        # Unrecognised tuple format; fall back to STAY
        self_semantic = None
    else:
        self_semantic = int(action)

    if self_semantic is None or self_semantic < 0 or self_semantic >= model_init.N_ACTIONS:
        return {k: normalize(np.asarray(qs[k], dtype=float)) for k in qs}

    dst, mode = model_init.semantic_action_from_index(self_semantic)
    self_pos_tgt, self_ori_tgt = model_init.SEMANTIC_DEST_TARGET_POSE[dst]
    self_terminal = model_init.INTERACT if mode == "interact" else model_init.STAY

    # Teleport self to target pose; other agent is untouched
    qs_macro = {k: np.array(v, dtype=float) for k, v in qs.items()}
    qs_macro["self_pos"] = np.eye(model_init.N_WALKABLE, dtype=float)[int(self_pos_tgt)]
    qs_macro["self_orientation"] = np.eye(model_init.N_DIRECTIONS, dtype=float)[int(self_ori_tgt)]

    pot_next = B_pot_state(
        {k: qs_macro[k] for k in state_state_dependencies["pot_state"]},
        self_terminal,
    )

    new_qs: dict[str, np.ndarray] = {}

    for factor, deps in state_state_dependencies.items():
        if factor in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
            if factor == "ck_put1":
                new_qs.update(B_checkboxes(qs_macro, self_terminal, q_pot_next=pot_next))
            continue

        parents = {k: qs_macro[k] for k in deps}

        if factor == "self_pos":
            new_qs[factor] = np.array(qs_macro["self_pos"], dtype=float)
        elif factor == "self_orientation":
            new_qs[factor] = np.array(qs_macro["self_orientation"], dtype=float)
        elif factor == "self_held":
            new_qs[factor] = B_self_held(parents, self_terminal)
        elif factor == "other_pos":
            new_qs[factor] = B_other_pos(parents)
        elif factor == "other_orientation":
            new_qs[factor] = B_other_orientation(parents)
        elif factor == "other_held":
            new_qs[factor] = B_other_held(parents)
        elif factor == "pot_state":
            new_qs[factor] = np.array(pot_next, dtype=float)
        elif factor.startswith("ctr_"):
            grid = int(factor.split("_")[1])
            new_qs[factor] = B_counter_occupancy(parents, self_terminal, grid)
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
