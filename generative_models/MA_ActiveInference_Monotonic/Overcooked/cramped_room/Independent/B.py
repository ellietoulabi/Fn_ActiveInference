"""
Transition model (B = P(S_t+1 | S_t, a)) for Independent paradigm — Cramped Room (monotonic model).
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

def _mix_with_uniform(p: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Apply factor-local transition noise:
        p <- (1-noise)*p + noise*Uniform
    and normalize.
    """
    p = np.array(p, dtype=float)
    if noise_level <= 0.0:
        return normalize(p)
    S = p.shape[0]
    if S <= 0:
        return p
    uniform = np.ones(S, dtype=float) / float(S)
    return normalize((1.0 - noise_level) * p + noise_level * uniform)


def _walkable_neighbors(w: int) -> list[int]:
    """4-neighborhood (N,S,E,W) in walkable-index space."""
    grid_idx = model_init.walkable_idx_to_grid_idx(int(w))
    x, y = model_init.index_to_xy(grid_idx)
    nbrs: list[int] = []
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= model_init.GRID_WIDTH or ny < 0 or ny >= model_init.GRID_HEIGHT:
            continue
        ngrid = model_init.xy_to_index(nx, ny)
        nw = model_init.grid_idx_to_walkable_idx(ngrid)
        if nw is not None:
            nbrs.append(int(nw))
    return sorted(set(nbrs))


# Precompute local kernels for position noise (no teleporting).
_POS_KERNELS: list[np.ndarray] = []
POS_TELEPORT_EPS = 1e-3  # tiny mass everywhere so infeasible != 0
for _w in range(model_init.N_WALKABLE):
    support = [_w] + _walkable_neighbors(_w)
    k = np.zeros(model_init.N_WALKABLE, dtype=float)
    k[support] = 1.0 / float(len(support))
    _POS_KERNELS.append(k)


def _mix_pos_locally(
    p: np.ndarray, noise_level: float, teleport_eps: float = POS_TELEPORT_EPS
) -> np.ndarray:
    """
    Position-specific noise: diffuse mass locally on the walkable graph.

    For one-hot p at w, the noisy distribution becomes:
        (1-noise)*one_hot(w) + noise*Uniform({w}+neighbors(w))
    For general p, this is a mixture over kernels of each state.
    """
    p = normalize(np.array(p, dtype=float))
    noise_level = float(noise_level)
    noise_level = max(0.0, min(1.0, noise_level))
    if noise_level <= 0.0:
        out = p
    else:
        smooth = np.zeros_like(p)
        for w in range(p.shape[0]):
            pw = p[w]
            if pw <= 1e-16:
                continue
            smooth += pw * _POS_KERNELS[w]
        out = normalize((1.0 - noise_level) * p + noise_level * smooth)

    # Ensure no state is exactly 0 (tiny "teleport" leakage).
    teleport_eps = float(teleport_eps)
    if teleport_eps > 0.0:
        out = _mix_with_uniform(out, teleport_eps)
    return out


def _update_orientation(ori_idx: int, action: int) -> int:
    """
    Deterministic orientation update.
    Directional actions set orientation; others keep it.
    """
    if action in (model_init.NORTH, model_init.SOUTH, model_init.EAST, model_init.WEST):
        return int(action)
    return int(ori_idx)


def _orientation_noise_kernel(true_ori: int, noise_level: float) -> np.ndarray:
    """
    p(orientation' | true_orientation').

    With probability (1-noise) keep the true orientation; distribute the remaining
    probability uniformly across the other 3 directions.
    """
    n = model_init.N_DIRECTIONS
    p = np.full(n, 0.0, dtype=float)
    true_ori = int(true_ori)
    noise_level = float(noise_level)
    noise_level = max(0.0, min(1.0, noise_level))
    if n <= 0:
        return p
    off = noise_level / max(1, n - 1)
    p[:] = off
    if 0 <= true_ori < n:
        p[true_ori] = 1.0 - noise_level
    return normalize(p)


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


def B_agent_pos(
    parents: dict, action: int, noise_level: float = 0.0, teleport_eps: float = POS_TELEPORT_EPS
) -> np.ndarray:
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

    return _mix_pos_locally(next_q, noise_level, teleport_eps=teleport_eps)


def B_agent_orientation(parents: dict, action: int, noise_level: float = 0.0) -> np.ndarray:
    """
    B for agent_orientation. Depends only on previous orientation, updated by action.
    """
    q_ori = np.array(parents["agent_orientation"], dtype=float)
    next_q = np.zeros(model_init.N_DIRECTIONS, dtype=float)

    for ori in range(model_init.N_DIRECTIONS):
        p_ori = q_ori[ori]
        if p_ori <= 1e-16:
            continue
        true_next = _update_orientation(ori, action)
        # Apply a small stochasticity around the intended next orientation.
        next_q += p_ori * _orientation_noise_kernel(true_next, noise_level)

    return normalize(next_q)


def _get_front(pos_w: int, ori: int) -> int:
    """Front tile type for (walkable pos, orientation)."""
    return model_init.compute_front_tile_type(pos_w, ori)


def _front_counter_slot(pos_w: int, ori: int) -> int | None:
    """If front is one of the modeled counters, return slot 0..4 else None."""
    return model_init.front_counter_slot(pos_w, ori)


INTERACT_SUCCESS_PROB = getattr(model_init, "INTERACT_SUCCESS_PROB", 0.9)


def B_agent_held(parents: dict, action: int, noise_level: float = 0.0) -> np.ndarray:
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
    q_counters = [
        np.array(parents["counter_0"], dtype=float),
        np.array(parents["counter_1"], dtype=float),
        np.array(parents["counter_2"], dtype=float),
        np.array(parents["counter_3"], dtype=float),
        np.array(parents["counter_4"], dtype=float),
    ]

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

                    # Default: no change
                    if action != model_init.INTERACT:
                        next_q[held] += w
                        continue

                    # INTERACT: deterministic intent with success prob and counter occupancy
                    if front == model_init.FRONT_ONION and held == model_init.HELD_NONE:
                        next_q[model_init.HELD_ONION] += w * INTERACT_SUCCESS_PROB
                        next_q[held] += w * (1.0 - INTERACT_SUCCESS_PROB)
                    elif front == model_init.FRONT_DISH and held == model_init.HELD_NONE:
                        next_q[model_init.HELD_DISH] += w * INTERACT_SUCCESS_PROB
                        next_q[held] += w * (1.0 - INTERACT_SUCCESS_PROB)
                    elif front == model_init.FRONT_POT and held == model_init.HELD_ONION and pot in (
                        model_init.POT_0,
                        model_init.POT_1,
                        model_init.POT_2,
                    ):
                        next_q[model_init.HELD_NONE] += w * INTERACT_SUCCESS_PROB
                        next_q[held] += w * (1.0 - INTERACT_SUCCESS_PROB)
                    elif front == model_init.FRONT_POT and held == model_init.HELD_DISH and pot == model_init.POT_3:
                        next_q[model_init.HELD_SOUP] += w * INTERACT_SUCCESS_PROB
                        next_q[held] += w * (1.0 - INTERACT_SUCCESS_PROB)
                    elif front == model_init.FRONT_SERVE and held == model_init.HELD_SOUP:
                        next_q[model_init.HELD_NONE] += w * INTERACT_SUCCESS_PROB
                        next_q[held] += w * (1.0 - INTERACT_SUCCESS_PROB)
                    elif front == model_init.FRONT_COUNTER:
                        slot = _front_counter_slot(pos_w, ori)
                        if slot is None:
                            next_q[held] += w
                            continue
                        q_ctr = q_counters[slot]

                        if held != model_init.HELD_NONE:
                            # Drop only if counter is empty.
                            p_success = INTERACT_SUCCESS_PROB * float(q_ctr[model_init.HELD_NONE])
                            next_q[model_init.HELD_NONE] += w * p_success
                            next_q[held] += w * (1.0 - p_success)
                        else:
                            # Pick up if counter has an object (distributed by its belief).
                            p_obj = 1.0 - float(q_ctr[model_init.HELD_NONE])
                            for obj in (model_init.HELD_ONION, model_init.HELD_DISH, model_init.HELD_SOUP):
                                next_q[obj] += w * INTERACT_SUCCESS_PROB * float(q_ctr[obj])
                            next_q[model_init.HELD_NONE] += w * (1.0 - INTERACT_SUCCESS_PROB * p_obj)
                    else:
                        next_q[held] += w

    return _mix_with_uniform(next_q, noise_level)


def B_counters(qs: dict, action: int, noise_level: float = 0.0) -> dict[str, np.ndarray]:
    """
    Jointly update all counter slots (counter_0..counter_4).

    Counters change only when action==INTERACT and the agent is facing that counter.
    - If holding an object and the counter is empty -> place it (counter becomes object).
    - If holding nothing and the counter has an object -> pick it up (counter becomes empty).
    Otherwise, counter remains unchanged.
    """
    q_pos = np.array(qs["agent_pos"], dtype=float)
    q_ori = np.array(qs["agent_orientation"], dtype=float)
    q_held = np.array(qs["agent_held"], dtype=float)

    # Probability the agent's front tile is each counter slot.
    p_front = np.zeros(model_init.N_COUNTERS, dtype=float)
    for pos_w in range(model_init.N_WALKABLE):
        pp = q_pos[pos_w]
        if pp <= 1e-16:
            continue
        for ori in range(model_init.N_DIRECTIONS):
            po = q_ori[ori]
            if po <= 1e-16:
                continue
            slot = _front_counter_slot(pos_w, ori)
            if slot is not None:
                p_front[int(slot)] += pp * po

    out: dict[str, np.ndarray] = {}
    for i in range(model_init.N_COUNTERS):
        q_ctr = np.array(qs[f"counter_{i}"], dtype=float)

        if action != model_init.INTERACT or p_front[i] <= 1e-16:
            out[f"counter_{i}"] = _mix_with_uniform(q_ctr, noise_level)
            continue

        p_here = float(p_front[i])
        p_hold_none = float(q_held[model_init.HELD_NONE])
        p_hold_obj = 1.0 - p_hold_none

        # Transition matrix T: next = T @ q_ctr
        T = np.eye(model_init.N_HELD_TYPES, dtype=float)

        # If counter is empty, can become held object (drop)
        if p_hold_obj > 0.0:
            for obj in (model_init.HELD_ONION, model_init.HELD_DISH, model_init.HELD_SOUP):
                p_drop_obj = INTERACT_SUCCESS_PROB * p_here * float(q_held[obj])
                T[obj, model_init.HELD_NONE] += p_drop_obj
                T[model_init.HELD_NONE, model_init.HELD_NONE] -= p_drop_obj

        # If counter has object, can become empty (pickup) when holding none
        if p_hold_none > 0.0:
            p_pick = INTERACT_SUCCESS_PROB * p_here * p_hold_none
            for s in (model_init.HELD_ONION, model_init.HELD_DISH, model_init.HELD_SOUP):
                T[model_init.HELD_NONE, s] += p_pick
                T[s, s] -= p_pick

        next_ctr = T @ q_ctr
        out[f"counter_{i}"] = _mix_with_uniform(next_ctr, noise_level)

    return out


def B_pot_state(parents: dict, action: int, noise_level: float = 0.0) -> np.ndarray:
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
    return _mix_with_uniform(next_q, noise_level)


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


def B_checkboxes(parents: dict, action: int, noise_level: float = 0.0) -> dict[str, np.ndarray]:
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
        "ck_put1": _mix_with_uniform(next_ck1, noise_level),
        "ck_put2": _mix_with_uniform(next_ck2, noise_level),
        "ck_put3": _mix_with_uniform(next_ck3, noise_level),
        "ck_plated": _mix_with_uniform(next_plat, noise_level),
        "ck_delivered": _mix_with_uniform(next_del, noise_level),
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

    # Allow per-factor overrides, otherwise default to B_NOISE_LEVEL.
    nl_pos = float(kwargs.get("B_NOISE_agent_pos", B_NOISE_LEVEL))
    nl_ori = float(kwargs.get("B_NOISE_agent_orientation", B_NOISE_LEVEL))
    nl_held = float(kwargs.get("B_NOISE_agent_held", B_NOISE_LEVEL))
    nl_pot = float(kwargs.get("B_NOISE_pot_state", B_NOISE_LEVEL))
    nl_ck = float(kwargs.get("B_NOISE_checkboxes", B_NOISE_LEVEL))
    nl_ctr = float(kwargs.get("B_NOISE_counters", B_NOISE_LEVEL))

    for factor, deps in state_state_dependencies.items():
        if factor in ("ck_put1", "ck_put2", "ck_put3", "ck_plated", "ck_delivered"):
            if factor == "ck_put1":
                new_qs.update(B_checkboxes(qs, action, noise_level=nl_ck))
            continue
        if factor in ("counter_0", "counter_1", "counter_2", "counter_3", "counter_4"):
            if factor == "counter_0":
                new_qs.update(B_counters(qs, action, noise_level=0))
            continue
        parents = {k: qs[k] for k in deps}
        if factor == "agent_pos":
            new_qs[factor] = B_agent_pos(parents, action, noise_level=nl_pos)
        elif factor == "agent_orientation":
            new_qs[factor] = B_agent_orientation(parents, action, noise_level=nl_ori)
        elif factor == "agent_held":
            new_qs[factor] = B_agent_held(parents, action, noise_level=0)
        elif factor == "pot_state":
            new_qs[factor] = B_pot_state(parents, action, noise_level=nl_pot)
        else:
            new_qs[factor] = normalize(np.array(parents[factor], dtype=float))

    # Ensure all outputs are properly normalized numpy arrays
    for k in new_qs:
        new_qs[k] = normalize(np.array(new_qs[k], dtype=float))

    return new_qs




