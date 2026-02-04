# B.py
"""
Transition model (B) for Independent paradigm (single-agent actions) — Cramped Room (Stage 1).

Key Stage-1 changes:
- Movement respects walls.
- Interact effects are context-dependent via:
    front_type = compute_front_tile_type(agent_pos, agent_orientation, other_agent_pos)
- Pot model supports 3 onions + cook_time=1:
    POT_0 -> POT_1 -> POT_2 -> POT_COOKING -> POT_READY
- soup_delivered is OBSERVATION-ONLY (event), NOT a hidden state.
  => B_fn returns only hidden state factors, no "soup_delivered".
"""

import numpy as np
from . import model_init


# ----------------------------
# Utils
# ----------------------------
def normalize(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    return p / max(s, 1e-8)


def _update_orientation(ori_idx: int, action: int) -> int:
    if action in (model_init.NORTH, model_init.SOUTH, model_init.EAST, model_init.WEST):
        return action
    return ori_idx


def _compute_new_pos(pos_idx: int, action: int) -> int:
    """
    Attempted move respecting bounds and walls.
    If invalid, returns original pos_idx.
    """
    if action in (model_init.STAY, model_init.INTERACT):
        return pos_idx

    x, y = model_init.index_to_xy(pos_idx)

    if action == model_init.NORTH:
        y -= 1
    elif action == model_init.SOUTH:
        y += 1
    elif action == model_init.EAST:
        x += 1
    elif action == model_init.WEST:
        x -= 1

    if x < 0 or x >= model_init.GRID_WIDTH or y < 0 or y >= model_init.GRID_HEIGHT:
        return pos_idx

    new_idx = model_init.xy_to_index(x, y)

    # Env: only " " (empty) is walkable; X, P, O, D, S are not (counters, pot, dispensers, serving).
    if new_idx in model_init.BLOCKED_MOVEMENT_INDICES:
        return pos_idx

    return new_idx


def _valid_neighbors(idx: int) -> list[int]:
    """
    Valid neighbor cells for other-agent dynamics: N,S,E,W that are walkable
    (not counters, pot, dispensers, or serving — matches env).
    """
    x, y = model_init.index_to_xy(idx)
    candidates = [(x, y - 1), (x, y + 1), (x + 1, y), (x - 1, y)]
    out = []
    for nx, ny in candidates:
        if 0 <= nx < model_init.GRID_WIDTH and 0 <= ny < model_init.GRID_HEIGHT:
            nidx = model_init.xy_to_index(nx, ny)
            if nidx not in model_init.BLOCKED_MOVEMENT_INDICES:
                out.append(nidx)
    return out


def _manhattan_distance(pos1: int, pos2: int) -> int:
    x1, y1 = model_init.index_to_xy(pos1)
    x2, y2 = model_init.index_to_xy(pos2)
    return abs(x1 - x2) + abs(y1 - y2)


# ----------------------------
# B for each hidden factor
# ----------------------------
def B_agent_pos(qs: dict, action: int, collision_awareness: float = 0.5) -> np.ndarray:
    """
    Update agent position given action, with soft collision avoidance against other agent.

    - If the agent attempts to move into other agent's current cell, it stays.
    - If the other agent is near, add some probability of hesitation (stay).
    - If my proposed cell is adjacent to the other (they could move into it or we could swap),
      env would block both on collision → add probability I stay (same-target / swap collision).
    """
    q_pos = np.array(qs["agent_pos"], dtype=float)
    q_other = np.array(qs["other_agent_pos"], dtype=float)
    S = model_init.GRID_SIZE
    next_q = np.zeros(S, dtype=float)

    is_move = action in (model_init.NORTH, model_init.SOUTH, model_init.EAST, model_init.WEST)

    for pos in range(S):
        if q_pos[pos] <= 1e-16:
            continue
        for other_pos in range(S):
            if q_other[other_pos] <= 1e-16:
                continue
            if pos == other_pos:
                continue

            w = q_pos[pos] * q_other[other_pos]
            if w <= 1e-16:
                continue

            proposed = _compute_new_pos(pos, action)

            # Hesitation when near other agent
            p_stay = 0.0
            if is_move:
                d = _manhattan_distance(pos, other_pos)
                if d == 1:
                    p_stay = max(p_stay, collision_awareness * 0.7)
                elif d == 2:
                    p_stay = max(p_stay, collision_awareness * 0.4)
                # Same-target or swap: my target cell is adjacent to other → they might move there too → collision → I stay
                other_neighbors = _valid_neighbors(other_pos)
                if proposed in other_neighbors:
                    p_stay = max(p_stay, 0.6)  # high chance we collide and both stay (env blocks both)

            p_move = 1.0 - p_stay

            if p_stay > 0:
                next_q[pos] += w * p_stay

            # Direct collision: moving into other's current cell
            if proposed == other_pos:
                next_q[pos] += w * p_move
            else:
                next_q[proposed] += w * p_move

    return normalize(next_q)


def B_agent_orientation(qs: dict, action: int) -> np.ndarray:
    q_ori = np.array(qs["agent_orientation"], dtype=float)
    next_q = np.zeros(model_init.N_DIRECTIONS, dtype=float)

    for ori in range(model_init.N_DIRECTIONS):
        if q_ori[ori] <= 1e-16:
            continue
        new_ori = _update_orientation(ori, action)
        next_q[new_ori] += q_ori[ori]

    return normalize(next_q)


def B_agent_held(qs: dict, action: int) -> np.ndarray:
    """
    Held-object transitions based on FRONT tile type (not "standing on" tile).

    INTERACT rules (with optional noise via INTERACT_SUCCESS_PROB):
      front ONION + held NONE -> held ONION (or stay NONE with prob 1 - INTERACT_SUCCESS_PROB)
      front DISH  + held NONE -> held DISH  (or stay NONE with prob 1 - INTERACT_SUCCESS_PROB)
      front POT   + held ONION + pot in {POT_0,POT_1,POT_2} -> held NONE (or stay ONION w/ noise)
      front POT   + held DISH  + pot POT_READY -> held SOUP (or stay DISH w/ noise)
      front SERVE + held SOUP -> held NONE (or stay SOUP w/ noise)
      front COUNTER + held in {ONION,DISH,SOUP} -> held NONE (drop on counter, or stay w/ noise)
    """
    q_pos = np.array(qs["agent_pos"], dtype=float)
    q_ori = np.array(qs["agent_orientation"], dtype=float)
    q_other = np.array(qs["other_agent_pos"], dtype=float)
    q_held = np.array(qs["agent_held"], dtype=float)
    q_pot = np.array(qs["pot_state"], dtype=float)

    next_q = np.zeros(model_init.N_HELD_TYPES, dtype=float)

    for pos in range(model_init.GRID_SIZE):
        if q_pos[pos] <= 1e-16:
            continue
        for ori in range(model_init.N_DIRECTIONS):
            if q_ori[ori] <= 1e-16:
                continue
            for other_pos in range(model_init.GRID_SIZE):
                if q_other[other_pos] <= 1e-16:
                    continue

                front = model_init.compute_front_tile_type(pos, ori, other_pos)

                for held in range(model_init.N_HELD_TYPES):
                    if q_held[held] <= 1e-16:
                        continue
                    for pot_state in range(model_init.N_POT_STATES):
                        if q_pot[pot_state] <= 1e-16:
                            continue

                        w = q_pos[pos] * q_ori[ori] * q_other[other_pos] * q_held[held] * q_pot[pot_state]
                        if w <= 1e-16:
                            continue

                        new_held = held
                        interact_success_prob = getattr(
                            model_init, "INTERACT_SUCCESS_PROB", 1.0
                        )

                        if action == model_init.INTERACT:
                            if front == model_init.FRONT_BLOCKED_BY_OTHER:
                                new_held = held
                                interact_success_prob = 1.0  # no change, no noise

                            elif front == model_init.FRONT_ONION and held == model_init.HELD_NONE:
                                new_held = model_init.HELD_ONION

                            elif front == model_init.FRONT_DISH and held == model_init.HELD_NONE:
                                new_held = model_init.HELD_DISH

                            elif (
                                front == model_init.FRONT_POT
                                and held == model_init.HELD_ONION
                                and pot_state in (model_init.POT_0, model_init.POT_1, model_init.POT_2)
                            ):
                                new_held = model_init.HELD_NONE

                            elif (
                                front == model_init.FRONT_POT
                                and held == model_init.HELD_DISH
                                and pot_state == model_init.POT_READY
                            ):
                                new_held = model_init.HELD_SOUP

                            elif front == model_init.FRONT_SERVE and held == model_init.HELD_SOUP:
                                new_held = model_init.HELD_NONE

                            elif (
                                front == model_init.FRONT_COUNTER
                                and held in (model_init.HELD_ONION, model_init.HELD_DISH, model_init.HELD_SOUP)
                            ):
                                new_held = model_init.HELD_NONE  # drop on counter

                            else:
                                interact_success_prob = 1.0  # no-op interact, no noise

                            # Apply INTERACT outcome noise: success with prob, else stay at current held
                            next_q[new_held] += w * interact_success_prob
                            if interact_success_prob < 1.0:
                                next_q[held] += w * (1.0 - interact_success_prob)
                        else:
                            next_q[new_held] += w

    return normalize(next_q)


def B_other_agent_pos(qs: dict, stay_prob: float = 0.7, move_prob: float = 0.3) -> np.ndarray:
    """
    Simple environment dynamics for the other agent:
      - stay with stay_prob
      - otherwise move to a random valid neighbor (not walls)
    """
    q_other = np.array(qs["other_agent_pos"], dtype=float)
    S = model_init.GRID_SIZE
    next_q = np.zeros(S, dtype=float)

    for pos in range(S):
        p = q_other[pos]
        if p <= 1e-16:
            continue

        next_q[pos] += p * stay_prob

        neigh = _valid_neighbors(pos)
        if len(neigh) == 0:
            next_q[pos] += p * move_prob
        else:
            share = (p * move_prob) / len(neigh)
            for npos in neigh:
                next_q[npos] += share

    return normalize(next_q)


def _cook_to_ready_prob() -> float:
    """
    Convert cook_time to a geometric-step approximation.
    For cook_time=1 -> 1.0 (ready next step).
    """
    ct = int(getattr(model_init, "COOK_TIME", 1))
    ct = max(1, ct)
    return min(1.0, 1.0 / ct)


def B_pot_state(qs: dict, action: int) -> np.ndarray:
    """
    Pot transitions for 3-onion recipe + cook_time.

    Deposit onion (requires being in front of pot):
      POT_0 -> POT_1
      POT_1 -> POT_2
      POT_2 -> POT_COOKING

    Cooking:
      POT_COOKING -> POT_READY with prob p_cook_to_ready (1.0 when cook_time=1)

    Taking soup:
      INTERACT in front of pot + holding DISH + pot READY -> POT_0
    """
    q_pos = np.array(qs["agent_pos"], dtype=float)
    q_ori = np.array(qs["agent_orientation"], dtype=float)
    q_other = np.array(qs["other_agent_pos"], dtype=float)
    q_held = np.array(qs["agent_held"], dtype=float)
    q_pot = np.array(qs["pot_state"], dtype=float)

    next_q = np.zeros(model_init.N_POT_STATES, dtype=float)
    p_cook_to_ready = _cook_to_ready_prob()

    for pot_state in range(model_init.N_POT_STATES):
        if q_pot[pot_state] <= 1e-16:
            continue

        for pos in range(model_init.GRID_SIZE):
            if q_pos[pos] <= 1e-16:
                continue
            for ori in range(model_init.N_DIRECTIONS):
                if q_ori[ori] <= 1e-16:
                    continue
                for other_pos in range(model_init.GRID_SIZE):
                    if q_other[other_pos] <= 1e-16:
                        continue

                    front = model_init.compute_front_tile_type(pos, ori, other_pos)

                    for held in range(model_init.N_HELD_TYPES):
                        if q_held[held] <= 1e-16:
                            continue

                        w = q_pot[pot_state] * q_pos[pos] * q_ori[ori] * q_other[other_pos] * q_held[held]
                        if w <= 1e-16:
                            continue

                        # default: stay same
                        if pot_state == model_init.POT_0:
                            if action == model_init.INTERACT and front == model_init.FRONT_POT and held == model_init.HELD_ONION:
                                next_q[model_init.POT_1] += w
                            else:
                                next_q[model_init.POT_0] += w

                        elif pot_state == model_init.POT_1:
                            if action == model_init.INTERACT and front == model_init.FRONT_POT and held == model_init.HELD_ONION:
                                next_q[model_init.POT_2] += w
                            else:
                                next_q[model_init.POT_1] += w

                        elif pot_state == model_init.POT_2:
                            if action == model_init.INTERACT and front == model_init.FRONT_POT and held == model_init.HELD_ONION:
                                next_q[model_init.POT_COOKING] += w
                            else:
                                next_q[model_init.POT_2] += w

                        elif pot_state == model_init.POT_COOKING:
                            # cooking progression independent of our action
                            next_q[model_init.POT_READY] += w * p_cook_to_ready
                            next_q[model_init.POT_COOKING] += w * (1.0 - p_cook_to_ready)

                        elif pot_state == model_init.POT_READY:
                            if action == model_init.INTERACT and front == model_init.FRONT_POT and held == model_init.HELD_DISH:
                                # soup taken -> reset pot
                                next_q[model_init.POT_0] += w
                            else:
                                next_q[model_init.POT_READY] += w

    return normalize(next_q)


# ----------------------------
# Main B
# ----------------------------
def B_fn(qs: dict, action: int, collision_awareness: float = 0.5, **kwargs) -> dict[str, np.ndarray]:
    """
    Main transition model function for hidden state factors only.
    kwargs (e.g. width, height from env_params) are ignored for this model.
    """
    action = int(action)
    next_state: dict[str, np.ndarray] = {}

    next_state["agent_pos"] = B_agent_pos(qs, action, collision_awareness=collision_awareness)
    next_state["agent_orientation"] = B_agent_orientation(qs, action)
    next_state["agent_held"] = B_agent_held(qs, action)
    next_state["other_agent_pos"] = B_other_agent_pos(qs)
    next_state["pot_state"] = B_pot_state(qs, action)

    for k in next_state:
        next_state[k] = normalize(np.array(next_state[k], dtype=float))

    return next_state
