"""
Transition model (B) for ObservablePartner variant.

This keeps the structure similar to the SA "standard" B.py:
- Factor-wise updates based on model_init.state_state_dependencies
- Movement diffusion noise
- Button-state transitions use p(at button) under beliefs
"""

import numpy as np
from . import model_init

state_state_dependencies = model_init.state_state_dependencies


def normalize(p):
    return p / np.maximum(np.sum(p), 1e-8)


def B_my_pos(parents, action, width, height, noise):
    q = np.array(parents["my_pos"], dtype=float)
    S = q.shape[0]
    new_q = np.zeros(S, dtype=float)

    noise = noise * 0.2 if action == model_init.NOOP else noise

    def next_idx(i):
        x, y = i % width, i // width
        if action == model_init.UP:
            y = max(0, y - 1)
        elif action == model_init.DOWN:
            y = min(height - 1, y + 1)
        elif action == model_init.LEFT:
            x = max(0, x - 1)
        elif action == model_init.RIGHT:
            x = min(width - 1, x + 1)
        return y * width + x

    for cur in range(S):
        p_cur = q[cur]
        nxt = next_idx(cur) if action in (model_init.UP, model_init.DOWN, model_init.LEFT, model_init.RIGHT) else cur
        new_q[nxt] += p_cur * (1.0 - noise)
        if S > 1:
            noise_share = p_cur * noise / (S - 1)
            new_q += noise_share
            new_q[nxt] -= noise_share

    return new_q / np.maximum(np.sum(new_q), 1e-8)


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
    return new / np.maximum(np.sum(new), 1e-8)


def _press_update(q_state, p_at_button):
    # Deterministic press: if at button and not pressed -> pressed
    q0, q1 = float(q_state[0]), float(q_state[1])
    next0 = q0 * (1.0 - p_at_button)
    next1 = q1 + q0 * p_at_button
    out = np.array([next0, next1], dtype=float)
    return out / np.maximum(np.sum(out), 1e-8)


def B_red_button_state(parents, action, noise):
    q_state = np.array(parents["red_button_state"], dtype=float)
    if action != model_init.PRESS:
        return q_state
    q_me = np.array(parents["my_pos"], dtype=float)
    q_red_pos = np.array(parents["red_button_pos"], dtype=float)
    p_at = float(np.sum(q_me * q_red_pos))
    return _press_update(q_state, p_at)


def B_blue_button_state(parents, action, noise):
    q_state = np.array(parents["blue_button_state"], dtype=float)
    if action != model_init.PRESS:
        return q_state
    q_me = np.array(parents["my_pos"], dtype=float)
    q_blue_pos = np.array(parents["blue_button_pos"], dtype=float)
    p_at = float(np.sum(q_me * q_blue_pos))
    return _press_update(q_state, p_at)


def B_fn(qs, action, width=3, height=3, B_NOISE_LEVEL=0.05):
    action = int(action)
    new_qs = {}

    BUTTON_POS_NOISE = 0.01
    OTHER_POS_NOISE = 0.0

    for factor, deps in state_state_dependencies.items():
        parents = {k: qs[k] for k in deps}

        if factor == "my_pos":
            new_qs[factor] = B_my_pos(parents, action, width, height, B_NOISE_LEVEL)
        elif factor == "other_pos":
            if OTHER_POS_NOISE > 0:
                new_qs[factor] = B_static_with_noise(parents, factor, OTHER_POS_NOISE)
            else:
                new_qs[factor] = np.array(parents[factor], dtype=float).copy()
        elif factor in ("red_button_pos", "blue_button_pos"):
            new_qs[factor] = B_static_with_noise(parents, factor, BUTTON_POS_NOISE)
        elif factor == "red_button_state":
            new_qs[factor] = B_red_button_state(parents, action, B_NOISE_LEVEL)
        elif factor == "blue_button_state":
            new_qs[factor] = B_blue_button_state(parents, action, B_NOISE_LEVEL)
        else:
            new_qs[factor] = np.array(qs[factor], dtype=float).copy()

    for f in new_qs:
        new_qs[f] = normalize(np.array(new_qs[f], dtype=float))
    return new_qs


