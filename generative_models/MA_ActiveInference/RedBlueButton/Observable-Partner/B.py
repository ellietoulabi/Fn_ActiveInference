"""
Transition Model (B) for Multi-Agent RedBlueButton.

B_fn computes expected next state given current beliefs and action.
"""

import numpy as np
from . import model_init

state_state_dependencies = model_init.state_state_dependencies


def normalize(p):
    return p / np.maximum(np.sum(p), 1e-8)


def B_my_pos(parents, action, width, height, noise):
    """
    Update belief over my position given action (similar to SA B_agent_pos).
    parents['my_pos'] is a probability vector over grid cells.
    """
    q = np.array(parents["my_pos"])
    S = q.shape[0]
    new_q = np.zeros(S)

    # Less noise for NOOP (more certain when staying)
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
        # PRESS and NOOP: stay
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
    """Static factor update with small diffusion noise (same pattern as SA)."""
    q = np.array(parents[self_key])
    S = q.shape[0]
    if S <= 1:
        return q
    new = np.zeros(S)
    for s in range(S):
        stay = q[s] * (1.0 - noise)
        leak = q[s] * noise / (S - 1)
        new[s] += stay
        new += leak
        new[s] -= leak
    return new / np.maximum(np.sum(new), 1e-8)


def B_red_button_state(parents, action, noise):
    """
    Update red button state belief under PRESS action using p(at red button).
    Mirrors SA logic (probabilistic success if at button; deterministic otherwise).
    """
    q_state = np.array(parents["red_button_state"])
    q_me = np.array(parents["my_pos"])
    q_red_pos = np.array(parents["red_button_pos"])

    # Non-PRESS actions: button state stays exactly the same (deterministic)
    if action in (model_init.UP, model_init.DOWN, model_init.LEFT, model_init.RIGHT, model_init.NOOP):
        return q_state

    p_at_button = float(np.sum(q_me * q_red_pos))
    q0, q1 = float(q_state[0]), float(q_state[1])

    # If at button and not pressed -> pressed (95%); else state persists.
    next0 = q0 * (p_at_button * 0.05 + (1.0 - p_at_button) * 1.0)
    next1 = q0 * p_at_button * 0.95 + q1 * 1.0

    result = np.array([next0, next1], dtype=float)
    return result / np.maximum(np.sum(result), 1e-8)


def B_blue_button_state(parents, action, noise):
    """Same as B_red_button_state but for the blue button."""
    q_state = np.array(parents["blue_button_state"])
    q_me = np.array(parents["my_pos"])
    q_blue_pos = np.array(parents["blue_button_pos"])

    # Non-PRESS actions: button state stays exactly the same (deterministic)
    if action in (model_init.UP, model_init.DOWN, model_init.LEFT, model_init.RIGHT, model_init.NOOP):
        return q_state

    p_at_button = float(np.sum(q_me * q_blue_pos))
    q0, q1 = float(q_state[0]), float(q_state[1])

    next0 = q0 * (p_at_button * 0.05 + (1.0 - p_at_button) * 1.0)
    next1 = q0 * p_at_button * 0.95 + q1 * 1.0

    result = np.array([next0, next1], dtype=float)
    return result / np.maximum(np.sum(result), 1e-8)


def B_fn(qs, action, width=3, height=3, B_NOISE_LEVEL=0.05):
    """
    Factorized transition model consistent with the SA (standard) implementation.

    Signature mirrors SA: B_fn(qs, action, width, height, B_NOISE_LEVEL)
    so `control.get_expected_state(..., env_params)` can pass width/height by name.
    """
    # Base noise levels (match SA flavor)
    BUTTON_POS_NOISE = 0.01  # slight drift / forgetting
    OTHER_POS_NOISE = 0.00   # treat other_pos as observed (identity)

    action = int(action)
    new_qs = {}

    for factor, deps in state_state_dependencies.items():
        parents = {k: qs[k] for k in deps}

        if factor == "my_pos":
            new_qs[factor] = B_my_pos(parents, action, width, height, B_NOISE_LEVEL)

        elif factor == "other_pos":
            # We don't control the other agent; keep as-is (optionally with tiny diffusion).
            if OTHER_POS_NOISE > 0:
                new_qs[factor] = B_static_with_noise(parents, factor, OTHER_POS_NOISE)
            else:
                new_qs[factor] = np.array(parents[factor]).copy()

        elif factor in ("red_button_pos", "blue_button_pos"):
            new_qs[factor] = B_static_with_noise(parents, factor, BUTTON_POS_NOISE)

        elif factor == "red_button_state":
            new_qs[factor] = B_red_button_state(parents, action, B_NOISE_LEVEL)

        elif factor == "blue_button_state":
            new_qs[factor] = B_blue_button_state(parents, action, B_NOISE_LEVEL)

        else:
            # Fallback: identity
            new_qs[factor] = np.array(qs[factor]).copy()

    # Final normalization safety
    for f in new_qs:
        new_qs[f] = normalize(np.array(new_qs[f], dtype=float))

    return new_qs
