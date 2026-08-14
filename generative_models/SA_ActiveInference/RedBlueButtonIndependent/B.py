import numpy as np
from . import model_init
state_state_dependencies = model_init.state_state_dependencies




# --------------------------------
# Dependencies
# --------------------------------
# state_state_dependencies = {
#     "agent_pos": ["agent_pos"],

#     "red_button_pos": ["red_button_pos"],
#     "blue_button_pos": ["blue_button_pos"],

#     "red_button_state": ["agent_pos", "red_button_pos", "red_button_state"],
#     "blue_button_state": ["agent_pos", "blue_button_pos", "blue_button_state"],

#     "goal_context": ["goal_context"],
# }



# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize(p):
    # Pure NumPy normalization (avoid importing JAX).
    return p / np.maximum(np.sum(p), 1e-8)


# --------------------------------
# Factor updates
# --------------------------------
def B_agent_pos(parents, action, width, height, noise):
    q = np.array(parents["agent_pos"])   #a probability vector over all grid cells (size S = width * height)
    S = q.shape[0]
    new_q = np.zeros(S)

    # Use LESS noise for NOOP (more certain when staying)
    noise = noise * 0.2 if action == 5 else noise  # 1% for NOOP vs 5% for movement


    def next_idx(i):
        x, y = i % width, i // width
        if action == 0:      # up
            y = max(0, y - 1)
        elif action == 1:    # down
            y = min(height - 1, y + 1)
        elif action == 2:    # left
            x = max(0, x - 1)
        elif action == 3:    # right
            x = min(width - 1, x + 1)
        # 4=open, 5=noop: stay
        return y * width + x

    q_other = parents.get("other_pos")
    if q_other is None:
        # Original behavior, unchanged -- no other_pos dependency wired in.
        for cur in range(S): #  for each pissible position:
            p_cur = q[cur] # prob of being in that pos
            nxt = next_idx(cur) if action in (0, 1, 2, 3) else cur # next pos
            new_q[nxt] += p_cur * (1.0 - noise)  # move prob to next state

            if S > 1:
                noise_share = p_cur * noise / (S - 1) # spread noise to other states
                new_q += noise_share
                new_q[nxt] -= noise_share

        # Normalize
        return new_q / np.maximum(np.sum(new_q), 1e-8)

    # other_pos wired in: treat the other agent's (believed) CURRENT position
    # as a static obstacle, same in kind as a wall -- no modeling of the
    # other agent's own intended move or turn order, since Independent
    # doesn't co-plan with the partner (see ai/02-debug.md, MA Red-Blue-Button
    # collision-blindness entry).
    q_other = np.array(q_other)
    for cur in range(S):
        p_cur = q[cur]
        if p_cur <= 1e-16:
            continue
        raw_nxt = next_idx(cur) if action in (0, 1, 2, 3) else cur
        for other in range(S):
            p_other = q_other[other]
            if p_other <= 1e-16:
                continue
            w = p_cur * p_other
            # Blocked if the deterministic target cell is (believed) occupied
            # by the other agent -- stay put, exactly like bumping a wall.
            nxt = cur if (action in (0, 1, 2, 3) and raw_nxt == other) else raw_nxt
            new_q[nxt] += w * (1.0 - noise)
            if S > 1:
                noise_share = w * noise / (S - 1)
                new_q += noise_share
                new_q[nxt] -= noise_share

    return new_q / np.maximum(np.sum(new_q), 1e-8)


def B_static_with_noise(parents, self_key, noise):
    q = np.array(parents[self_key])  # Convert to numpy first
    S = q.shape[0]
    if S <= 1:
        return q
    new = np.zeros(S)
    for s in range(S):
        stay = q[s] * (1.0 - noise)
        leak = q[s] * noise / (S - 1)
        new[s] += stay  # Direct assignment
        new += leak
        new[s] -= leak
    # Normalize in numpy, return numpy
    return new / np.maximum(np.sum(new), 1e-8)


# Genuine action-effect noise: even certainly standing on the button and
# pressing, there's a small chance it doesn't register -- the same kind of
# deliberate, task-relevant uncertainty as Overcooked's PROGRESS_SUCCESS_PROB
# (ai/02-debug.md, MA Overcooked section A / J.9), and the *proven* value is
# reused directly (0.85) rather than picking a new number. This gives the
# epistemic-value term something genuine and consistent to resolve right at
# the one action that actually decides the task, instead of (or in addition
# to) relying on movement noise, which is the pattern already found harmful
# in Overcooked (MOVE_UNCERTAINTY) and in this file (see the noise inventory
# entry, MA Red-Blue-Button section).
PRESS_SUCCESS_PROB = 0.85


def B_red_button_state(parents, action, noise):
    """
    parents = {
      "agent_pos": prob over grid,
      "red_button_pos": prob over grid,
      "red_button_state": [p(not_pressed), p(pressed)]
    }
    """
    # Convert to numpy to avoid JAX recompilation
    q_state = np.array(parents["red_button_state"])
    q_agent = np.array(parents["agent_pos"])
    q_button = np.array(parents["red_button_pos"])

    # Non-OPEN actions: button state stays exactly the same (deterministic)
    if action in (0, 1, 2, 3, 5):
        return q_state  # Already numpy

    # OPEN action: check if agent is at button position
    # Compute P(agent_pos == red_button_pos)
    p_at_button = np.sum(q_agent * q_button)

    q0, q1 = q_state[0], q_state[1]

    # If at button and not pressed -> pressed, with probability PRESS_SUCCESS_PROB
    # If at button and pressed -> stays pressed (100%)
    # If not at button -> stays exactly the same (deterministic)
    next0 = q0 * (p_at_button * (1.0 - PRESS_SUCCESS_PROB) + (1.0 - p_at_button) * 1.0)
    next1 = q0 * p_at_button * PRESS_SUCCESS_PROB + q1 * 1.0

    result = np.array([next0, next1])
    result = result / np.maximum(np.sum(result), 1e-8)
    return result


def B_blue_button_state(parents, action, noise):
    """
    parents = {
      "agent_pos": prob over grid,
      "blue_button_pos": prob over grid,
      "blue_button_state": [p(not_pressed), p(pressed)]
    }
    """
    # Convert to numpy to avoid JAX recompilation
    q_state = np.array(parents["blue_button_state"])
    q_agent = np.array(parents["agent_pos"])
    q_button = np.array(parents["blue_button_pos"])

    # Non-OPEN actions: button state stays exactly the same (deterministic)
    if action in (0, 1, 2, 3, 5):
        return q_state  # Already numpy

    # OPEN action: check if agent is at button position
    # Compute P(agent_pos == blue_button_pos)
    p_at_button = np.sum(q_agent * q_button)

    q0, q1 = q_state[0], q_state[1]

    # If at button and not pressed -> pressed, with probability PRESS_SUCCESS_PROB
    # If at button and pressed -> stays pressed (100%)
    # If not at button -> stays exactly the same (deterministic)
    next0 = q0 * (p_at_button * (1.0 - PRESS_SUCCESS_PROB) + (1.0 - p_at_button) * 1.0)
    next1 = q0 * p_at_button * PRESS_SUCCESS_PROB + q1 * 1.0

    result = np.array([next0, next1])
    result = result / np.maximum(np.sum(result), 1e-8)
    return result  # Return numpy, not jax


# --------------------------------
# apply_B
# --------------------------------
def B_fn(qs, action, width, height, B_NOISE_LEVEL=0.0):
    """
    B_NOISE_LEVEL now controls ONLY movement noise (via B_agent_pos), decoupled
    from press-success noise (which is its own named PRESS_SUCCESS_PROB
    constant above -- previously these were accidentally conflated: the same
    parameter was threaded into both, but press-success silently ignored it
    and used a hardcoded value anyway).

    Default changed from 0.05 to 0.0: the real environment's movement is
    100% deterministic, and spreading belief uniformly across all 8 other
    cells on every move is the same structural pattern already found to
    backfire in Overcooked (MOVE_UNCERTAINTY -- a spurious, path-length-
    proportional "distance bias" leaked into info gain). Matches
    FullyCollective/IndividuallyCollective, whose movement has always been
    exactly deterministic. See ai/02-debug.md, MA Red-Blue-Button noise
    inventory + this redesign entry.
    """
    new_qs = {}

    for factor, deps in state_state_dependencies.items():
        parents = {k: qs[k] for k in deps}

        if factor == "agent_pos":
            new_qs[factor] = B_agent_pos(parents, action, width, height, B_NOISE_LEVEL)

        elif factor == "other_pos":
            # No transition model of the partner's actual movement (Independent
            # doesn't co-plan) -- treated as static-with-noise, same pattern as
            # button positions, so uncertainty grows slightly each planning
            # step until corrected by the next direct observation.
            OTHER_POS_NOISE = 0.05
            new_qs[factor] = B_static_with_noise(parents, factor, OTHER_POS_NOISE)

        elif factor in ("red_button_pos", "blue_button_pos"):
            # Buttons are exactly static within an episode (they only relocate
            # at config/episode boundaries, which belief retention via
            # reset(keep_factors=...) already handles correctly). No noise
            # needed here -- matches FullyCollective/IndividuallyCollective's
            # BUTTON_POS_NOISE=0.0 exactly. Previously 0.01, an unnecessary
            # drift term for a quantity that never actually drifts intra-episode.
            BUTTON_POS_NOISE = 0.0
            new_qs[factor] = B_static_with_noise(parents, factor, BUTTON_POS_NOISE)

        elif factor == "red_button_state":
            new_qs[factor] = B_red_button_state(parents, action, B_NOISE_LEVEL)

        elif factor == "blue_button_state":
            new_qs[factor] = B_blue_button_state(parents, action, B_NOISE_LEVEL)

    return new_qs



