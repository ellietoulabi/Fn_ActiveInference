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


def _joint_move_step(p1, p2, a1, a2, width, height, ego_agent_index=1):
    """
    Deterministic joint move with sequential order and collision blocking,
    matching TwoAgentRedBlueButton's FIXED physical execution order: the real
    environment always processes physical agent 1's move before physical
    agent 2's, regardless of which agent is doing the reasoning.

    `ego_agent_index` tells us which physical agent the labels p1/a1
    ("agent1_pos"/its action) in this call actually refer to as "self".
    FullyCollective (single brain, no relabeling) and IndividuallyCollective's
    own agent 1 both have ego_agent_index=1 -- p1/a1 IS physical agent 1, the
    real first mover, so behavior here is byte-identical to the original
    (pre-fix) code. IndividuallyCollective's agent 2 relabels its observations
    so that ITS OWN position is stored under "agent1_pos" (see
    IndividuallyCollective/env_utils.py::env_obs_to_model_obs_for_agent) --
    for that belief, p2/a2 is the one that's physically agent 1 (the real
    first mover), not p1/a1. ego_agent_index=2 resolves in that true physical
    order instead.
    """
    p1 = int(p1)
    p2 = int(p2)
    a1 = int(a1)
    a2 = int(a2)

    if ego_agent_index == 2:
        # p2/a2 is physically agent 1 (real first mover); p1/a1 is physically agent 2.
        np2 = _compute_new_pos(p2, a2, width, height)
        if np2 == p1:
            np2 = p2
        np1 = _compute_new_pos(p1, a1, width, height)
        if np1 == np2:
            np1 = p1
        return np1, np2

    # ego_agent_index == 1 (default): p1/a1 is physically agent 1, the real first mover.
    np1 = _compute_new_pos(p1, a1, width, height)
    if np1 == p2:
        np1 = p1

    np2 = _compute_new_pos(p2, a2, width, height)
    if np2 == np1:
        np2 = p2

    return np1, np2


def B_agent_positions(parents, joint_action, width, height, ego_agent_index=1):
    """
    Update marginals for (agent1_pos, agent2_pos) given joint action.
    Assumes factorised belief q(p1)q(p2) and returns marginal next beliefs.
    See _joint_move_step's docstring for what ego_agent_index means.
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
            np1, np2 = _joint_move_step(p1, p2, a1, a2, width, height, ego_agent_index=ego_agent_index)
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


# Genuine action-effect noise: even certainly standing on the button and
# pressing, there's a small chance it doesn't register -- the same kind of
# deliberate, task-relevant uncertainty as Overcooked's PROGRESS_SUCCESS_PROB
# (ai/02-debug.md, MA Overcooked section A / J.9), reusing the *proven* value
# directly (0.85) rather than picking a new number. Previously this model had
# zero action-effect noise anywhere (100% deterministic press given position),
# unlike Independent which already had an analogous, if disconnected, 95/5
# hardcoded pattern -- this brings all three paradigms onto the same
# principled noise structure. See ai/02-debug.md, MA Red-Blue-Button noise
# inventory + redesign entry.
PRESS_SUCCESS_PROB = 0.85


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


def B_button_states(parents, joint_action, width, height, ego_agent_index=1):
    """
    Order-aware button state transition that better matches the environment logic.

    Approximation of TwoAgentRedBlueButtonEnv press loop:
    - Presses are processed in a fixed order: agent1, then agent2
    - Only the FIRST successful press in that loop is applied
    - Blue can be pressed while red is still unpressed, leading to lose

    We approximate this at the belief level by enumerating over factorised
    beliefs for positions and button positions and applying the same
    sequential 'first press wins' logic per joint state sample, then
    aggregating back to marginals over red/blue button states.

    `ego_agent_index` -- see _joint_move_step's docstring. The real
    environment always processes physical agent 1's press before physical
    agent 2's; when this belief belongs to IndividuallyCollective's agent 2
    (ego_agent_index=2), (p1,a1)/(p2,a2) have been relabeled so p2/a2 is the
    one that's actually physical agent 1, and the press-order loop below
    must be evaluated in that true physical order instead.
    """
    q_red_state = np.array(parents["red_button_state"], dtype=float)
    q_blue_state = np.array(parents["blue_button_state"], dtype=float)
    q1 = np.array(parents["agent1_pos"], dtype=float)
    q2 = np.array(parents["agent2_pos"], dtype=float)
    q_red_pos = np.array(parents["red_button_pos"], dtype=float)
    q_blue_pos = np.array(parents["blue_button_pos"], dtype=float)

    a1, a2 = decode_joint_action(joint_action)

    # If nobody presses, keep current states
    if a1 != model_init.PRESS and a2 != model_init.PRESS:
        return normalize(q_red_state), normalize(q_blue_state)

    S = q1.shape[0]
    next_red = np.zeros_like(q_red_state, dtype=float)
    next_blue = np.zeros_like(q_blue_state, dtype=float)

    # Enumerate over factorised joint beliefs:
    # red_state × blue_state × agent1_pos × agent2_pos × red_pos × blue_pos
    for r in range(2):
        pr_r = float(q_red_state[r])
        if pr_r <= 1e-16:
            continue
        for b in range(2):
            pr_rb = pr_r * float(q_blue_state[b])
            if pr_rb <= 1e-16:
                continue
            for pr_idx in range(S):  # red button position
                w_pr = pr_rb * float(q_red_pos[pr_idx])
                if w_pr <= 1e-16:
                    continue
                for pb_idx in range(S):  # blue button position
                    w_pr_pb = w_pr * float(q_blue_pos[pb_idx])
                    if w_pr_pb <= 1e-16:
                        continue
                    for p1 in range(S):
                        w_pr_pb_p1 = w_pr_pb * float(q1[p1])
                        if w_pr_pb_p1 <= 1e-16:
                            continue
                        for p2 in range(S):
                            w = w_pr_pb_p1 * float(q2[p2])
                            if w <= 1e-16:
                                continue

                            # Sequential press processing, in TRUE physical order
                            # (agent1 then agent2) regardless of which physical
                            # agent's ego-relabeled belief this is. Each
                            # individual press attempt branches on
                            # PRESS_SUCCESS_PROB instead of succeeding
                            # unconditionally -- a failed press leaves the
                            # button state unchanged and does NOT block the
                            # other agent's own press attempt this same step.
                            #
                            # A successful BLUE press (0->1) always ends the
                            # episode immediately (win if red is already 1,
                            # lose if red is still 0) -- matching the real
                            # environment's `if not terminated: # Agent 2's
                            # turn` guard, once a press ends the episode this
                            # step, any later press in press_order never
                            # executes at all, not even attempted. A
                            # successful RED press (0->1) never ends the
                            # episode by itself (win needs blue too), so
                            # processing continues normally after it. Without
                            # this, two agents pressing DIFFERENT buttons in
                            # the same step were modelled as both succeeding
                            # independently, which could make a losing
                            # blue-before-red joint action look like a win to
                            # the belief model (ai/02-debug.md, MA
                            # Red-Blue-Button "cross-button simultaneous press"
                            # entry).
                            press_order = ((p2, a2), (p1, a1)) if ego_agent_index == 2 else ((p1, a1), (p2, a2))

                            # branches: list of (red_s, blue_s, terminated, branch_weight)
                            branches = [(r, b, False, 1.0)]
                            for pos, act in press_order:
                                if act != model_init.PRESS:
                                    continue
                                next_branches = []
                                for red_s, blue_s, terminated, bw in branches:
                                    if terminated:
                                        # Episode already ended by an earlier
                                        # press this step -- this agent's press
                                        # never executes.
                                        next_branches.append((red_s, blue_s, terminated, bw))
                                        continue
                                    if pos == pr_idx and red_s == 0:
                                        next_branches.append((1, blue_s, False, bw * PRESS_SUCCESS_PROB))
                                        next_branches.append((red_s, blue_s, False, bw * (1.0 - PRESS_SUCCESS_PROB)))
                                    elif pos == pb_idx and blue_s == 0:
                                        next_branches.append((red_s, 1, True, bw * PRESS_SUCCESS_PROB))
                                        next_branches.append((red_s, blue_s, False, bw * (1.0 - PRESS_SUCCESS_PROB)))
                                    else:
                                        next_branches.append((red_s, blue_s, False, bw))
                                branches = next_branches

                            for red_s, blue_s, terminated, bw in branches:
                                next_red[red_s] += w * bw
                                next_blue[blue_s] += w * bw

    return normalize(next_red), normalize(next_blue)


def B_fn(qs, action, width=3, height=3, B_NOISE_LEVEL=0.0, ego_agent_index=1):
    """
    JOINT transition. `action` is the JOINT action index in [0, 35].

    Note: We set movement noise to 0.0 by default for the collective planner to
    better match the deterministic environment.

    `ego_agent_index=1` (default) preserves the original behavior exactly --
    correct for FullyCollective (single brain, physical labels always) and
    for IndividuallyCollective's own agent 1. IndividuallyCollective's agent 2
    must pass ego_agent_index=2 (via env_params) so the fixed real-environment
    execution order (physical agent 1 always first) is respected even though
    that agent's own belief has "agent1_pos"/"agent2_pos" relabeled to mean
    "me"/"other". See _joint_move_step and B_button_states docstrings.
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
        ego_agent_index=ego_agent_index,
    )
    new_qs["agent1_pos"] = q1_next
    new_qs["agent2_pos"] = q2_next

    # Static positions (buttons)
    new_qs["red_button_pos"] = B_static_with_noise({"red_button_pos": qs["red_button_pos"]}, "red_button_pos", BUTTON_POS_NOISE)
    new_qs["blue_button_pos"] = B_static_with_noise({"blue_button_pos": qs["blue_button_pos"]}, "blue_button_pos", BUTTON_POS_NOISE)

    # Order-aware button state transitions that approximate env press loop
    new_red_state, new_blue_state = B_button_states(
        {
            "red_button_state": qs["red_button_state"],
            "blue_button_state": qs["blue_button_state"],
            "agent1_pos": qs["agent1_pos"],
            "agent2_pos": qs["agent2_pos"],
            "red_button_pos": qs["red_button_pos"],
            "blue_button_pos": qs["blue_button_pos"],
        },
        joint_action,
        width,
        height,
        ego_agent_index=ego_agent_index,
    )
    new_qs["red_button_state"] = new_red_state
    new_qs["blue_button_state"] = new_blue_state

    # Final normalization safety
    for f in new_qs:
        new_qs[f] = normalize(np.array(new_qs[f], dtype=float))
    return new_qs


