import jax.numpy as jnp
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
    return p / jnp.maximum(jnp.sum(p), 1e-8)


# --------------------------------
# Factor updates
# --------------------------------
def B_agent_pos(parents, action, width, height, noise):
    q = parents["agent_pos"]              # shape [S]
    S = q.shape[0]
    new_q = jnp.zeros_like(q)

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

    for cur in range(S):
        p_cur = q[cur]
        nxt = next_idx(cur) if action in (0, 1, 2, 3) else cur
        new_q = new_q.at[nxt].add(p_cur * (1.0 - noise))

        if S > 1:
            noise_share = p_cur * noise / (S - 1)
            new_q = new_q + noise_share
            new_q = new_q.at[nxt].add(-noise_share)

    return normalize(new_q)


def B_static_with_noise(parents, self_key, noise):
    q = parents[self_key]
    S = q.shape[0]
    if S <= 1:
        return q
    new = jnp.zeros_like(q)
    for s in range(S):
        stay = q[s] * (1.0 - noise)
        leak = q[s] * noise / (S - 1)
        new = new.at[s].add(stay)
        new = new + leak
        new = new.at[s].add(-leak)
    return normalize(new)


def B_red_button_state(parents, action, noise):
    """
    parents = {
      "agent_pos": prob over grid,
      "red_button_pos": prob over grid,
      "red_button_state": [p(not_pressed), p(pressed)]
    }
    """
    q_state = parents["red_button_state"]
    q_agent = parents["agent_pos"]
    q_button = parents["red_button_pos"]

    # Non-OPEN actions: button state stays exactly the same (deterministic)
    if action in (0, 1, 2, 3, 5):
        return q_state

    # OPEN action: check if agent is at button position
    # Compute P(agent_pos == red_button_pos)
    p_at_button = jnp.sum(q_agent * q_button)
    
    q0, q1 = q_state[0], q_state[1]
    
    # If at button and not pressed → pressed (80% success, 20% fail)
    # If at button and pressed → stays pressed (100%)
    # If not at button → stays exactly the same (deterministic)
    next0 = q0 * (p_at_button * 0.2 + (1.0 - p_at_button) * 1.0)
    next1 = q0 * p_at_button * 0.8 + q1 * 1.0
    
    return normalize(jnp.array([next0, next1]))


def B_blue_button_state(parents, action, noise):
    """
    parents = {
      "agent_pos": prob over grid,
      "blue_button_pos": prob over grid,
      "blue_button_state": [p(not_pressed), p(pressed)]
    }
    """
    q_state = parents["blue_button_state"]
    q_agent = parents["agent_pos"]
    q_button = parents["blue_button_pos"]

    # Non-OPEN actions: button state stays exactly the same (deterministic)
    if action in (0, 1, 2, 3, 5):
        return q_state

    # OPEN action: check if agent is at button position
    # Compute P(agent_pos == blue_button_pos)
    p_at_button = jnp.sum(q_agent * q_button)
    
    q0, q1 = q_state[0], q_state[1]
    
    # If at button and not pressed → pressed (80% success, 20% fail)
    # If at button and pressed → stays pressed (100%)
    # If not at button → stays exactly the same (deterministic)
    next0 = q0 * (p_at_button * 0.2 + (1.0 - p_at_button) * 1.0)
    next1 = q0 * p_at_button * 0.8 + q1 * 1.0
    
    return normalize(jnp.array([next0, next1]))


# --------------------------------
# apply_B
# --------------------------------
def B_fn(qs, action, width, height, B_NOISE_LEVEL=0.05):
    new_qs = {}

    for factor, deps in state_state_dependencies.items():
        parents = {k: qs[k] for k in deps}

        if factor == "agent_pos":
            new_qs[factor] = B_agent_pos(parents, action, width, height, B_NOISE_LEVEL)

        elif factor in ("red_button_pos", "blue_button_pos"):
            new_qs[factor] = B_static_with_noise(parents, factor, B_NOISE_LEVEL)

        elif factor == "red_button_state":
            new_qs[factor] = B_red_button_state(parents, action, B_NOISE_LEVEL)

        elif factor == "blue_button_state":
            new_qs[factor] = B_blue_button_state(parents, action, B_NOISE_LEVEL)

    return new_qs



