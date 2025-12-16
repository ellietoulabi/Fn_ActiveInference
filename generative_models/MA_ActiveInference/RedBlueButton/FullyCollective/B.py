"""
Transition model (B) for FullyCollective paradigm (JOINT actions).

Keep structure close to SA B.py:
- Factor-wise updates driven by model_init.state_state_dependencies
- JOINT action is encoded as a single integer in [0, 35] representing (a1, a2)
- Movement matches TwoAgentRedBlueButton sequential rules with collision blocking:
    Agent1 moves first, then Agent2 (blocked if moving into the other's current/new cell)
"""

import numpy as np
from . import model_init

state_state_dependencies = model_init.state_state_dependencies


def normalize(p):
    return p / np.maximum(np.sum(p), 1e-8)


def decode_joint_action(joint_action):
    a = int(joint_action)
    return a // model_init.N_ACTIONS, a % model_init.N_ACTIONS


def _compute_new_xy(x, y, action, width, height):
    if action == model_init.UP:
        y = max(0, y - 1)
    elif action == model_init.DOWN:
        y = min(height - 1, y + 1)
    elif action == model_init.LEFT:
        x = max(0, x - 1)
    elif action == model_init.RIGHT:
        x = min(width - 1, x + 1)
    return x, y


def _compute_new_pos(pos, action, width, height):
    if action not in (model_init.UP, model_init.DOWN, model_init.LEFT, model_init.RIGHT):
        return int(pos)
    x, y = int(pos) % width, int(pos) // width
    nx, ny = _compute_new_xy(x, y, action, width, height)
    return ny * width + nx


def _joint_move_step(p1, p2, a1, a2, width, height):
    """
    Deterministic joint move with sequential order and collision blocking.
    """
    p1 = int(p1)
    p2 = int(p2)
    a1 = int(a1)
    a2 = int(a2)

    # Agent1 moves first; blocked if moving into agent2 current cell.
    np1 = _compute_new_pos(p1, a1, width, height)
    if np1 == p2:
        np1 = p1

    # Agent2 moves next; blocked if moving into agent1 new cell.
    np2 = _compute_new_pos(p2, a2, width, height)
    if np2 == np1:
        np2 = p2

    return np1, np2


def B_agent_positions(parents, joint_action, width, height):
    """
    Update marginals for (agent1_pos, agent2_pos) given joint action.
    Assumes factorised belief q(p1)q(p2) and returns marginal next beliefs.
    """
    q1 = np.array(parents["agent1_pos"], dtype=float)
    q2 = np.array(parents["agent2_pos"], dtype=float)
    S = q1.shape[0]

    a1, a2 = decode_joint_action(joint_action)

    next_q1 = np.zeros(S, dtype=float)
    next_q2 = np.zeros(S, dtype=float)

    for p1 in range(S):
        if q1[p1] <= 1e-16:
            continue
        for p2 in range(S):
            w = q1[p1] * q2[p2]
            if w <= 1e-16:
                continue
            np1, np2 = _joint_move_step(p1, p2, a1, a2, width, height)
            next_q1[np1] += w
            next_q2[np2] += w

    next_q1 = normalize(next_q1)
    next_q2 = normalize(next_q2)
    return next_q1, next_q2


def B_static_with_noise(parents, self_key, noise):
    q = np.array(parents[self_key], dtype=float)
    S = q.shape[0]
    if S <= 1:
        return q
    new = np.zeros(S, dtype=float)
    for s in range(S):
        stay = q[s] * (1.0 - noise)
        leak = q[s] * noise / (S - 1)
        new[s] += stay
        new += leak
        new[s] -= leak
    return normalize(new)


def _press_update(q_state, p_any_press):
    # Deterministic press: if not pressed and someone presses -> pressed
    q0, q1 = float(q_state[0]), float(q_state[1])
    next0 = q0 * (1.0 - p_any_press)
    next1 = q1 + q0 * p_any_press
    return normalize(np.array([next0, next1], dtype=float))


def B_red_button_state(parents, joint_action, noise):
    q_state = np.array(parents["red_button_state"], dtype=float)
    a1, a2 = decode_joint_action(joint_action)
    if a1 != model_init.PRESS and a2 != model_init.PRESS:
        return q_state

    q1 = np.array(parents["agent1_pos"], dtype=float)
    q2 = np.array(parents["agent2_pos"], dtype=float)
    q_red_pos = np.array(parents["red_button_pos"], dtype=float)

    p1 = float(np.sum(q1 * q_red_pos)) if a1 == model_init.PRESS else 0.0
    p2 = float(np.sum(q2 * q_red_pos)) if a2 == model_init.PRESS else 0.0
    p_any = 1.0 - (1.0 - p1) * (1.0 - p2)
    return _press_update(q_state, p_any)


def B_blue_button_state(parents, joint_action, noise):
    q_state = np.array(parents["blue_button_state"], dtype=float)
    a1, a2 = decode_joint_action(joint_action)
    if a1 != model_init.PRESS and a2 != model_init.PRESS:
        return q_state

    q1 = np.array(parents["agent1_pos"], dtype=float)
    q2 = np.array(parents["agent2_pos"], dtype=float)
    q_blue_pos = np.array(parents["blue_button_pos"], dtype=float)

    p1 = float(np.sum(q1 * q_blue_pos)) if a1 == model_init.PRESS else 0.0
    p2 = float(np.sum(q2 * q_blue_pos)) if a2 == model_init.PRESS else 0.0
    p_any = 1.0 - (1.0 - p1) * (1.0 - p2)
    return _press_update(q_state, p_any)


def B_fn(qs, action, width=3, height=3, B_NOISE_LEVEL=0.0):
    """
    JOINT transition. `action` is the JOINT action index in [0, 35].

    Note: We set movement noise to 0.0 by default for the collective planner to
    better match the deterministic environment.
    """
    joint_action = int(action)
    new_qs = {}

    BUTTON_POS_NOISE = 0.0  # env has static buttons within config

    # Positions (coupled)
    q1_next, q2_next = B_agent_positions(
        {"agent1_pos": qs["agent1_pos"], "agent2_pos": qs["agent2_pos"]},
        joint_action,
        width,
        height,
    )
    new_qs["agent1_pos"] = q1_next
    new_qs["agent2_pos"] = q2_next

    # Static positions (buttons)
    new_qs["red_button_pos"] = B_static_with_noise({"red_button_pos": qs["red_button_pos"]}, "red_button_pos", BUTTON_POS_NOISE)
    new_qs["blue_button_pos"] = B_static_with_noise({"blue_button_pos": qs["blue_button_pos"]}, "blue_button_pos", BUTTON_POS_NOISE)

    # Button state transitions depend on both agents + joint action
    new_qs["red_button_state"] = B_red_button_state(
        {
            "red_button_state": qs["red_button_state"],
            "agent1_pos": qs["agent1_pos"],
            "agent2_pos": qs["agent2_pos"],
            "red_button_pos": qs["red_button_pos"],
        },
        joint_action,
        B_NOISE_LEVEL,
    )
    new_qs["blue_button_state"] = B_blue_button_state(
        {
            "blue_button_state": qs["blue_button_state"],
            "agent1_pos": qs["agent1_pos"],
            "agent2_pos": qs["agent2_pos"],
            "blue_button_pos": qs["blue_button_pos"],
        },
        joint_action,
        B_NOISE_LEVEL,
    )

    # Final normalization safety
    for f in new_qs:
        new_qs[f] = normalize(np.array(new_qs[f], dtype=float))
    return new_qs


