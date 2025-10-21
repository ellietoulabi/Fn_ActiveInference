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
    q = np.array(parents["agent_pos"])   #a probability vector over all grid cells (size S = width * height)
    S = q.shape[0]
    new_q = np.zeros(S)

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
    
    # If at button and not pressed ->  pressed (80% success, 20% fail)
    # If at button and pressed -> stays pressed (100%)
    # If not at button -> stays exactly the same (deterministic)
    next0 = q0 * (p_at_button * 0.2 + (1.0 - p_at_button) * 1.0)
    next1 = q0 * p_at_button * 0.8 + q1 * 1.0
    
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
    
    # If at button and not pressed → pressed (80% success, 20% fail)
    # If at button and pressed → stays pressed (100%)
    # If not at button → stays exactly the same (deterministic)
    next0 = q0 * (p_at_button * 0.05 + (1.0 - p_at_button) * 1.0)
    next1 = q0 * p_at_button * 0.95 + q1 * 1.0
    
    result = np.array([next0, next1])
    result = result / np.maximum(np.sum(result), 1e-8)
    return result  # Return numpy, not jax


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
            # Small noise for non-stationary environment (buttons may slowly drift)
            # Beliefs decay slowly unless confirmed by observations
            BUTTON_POS_NOISE = 0.01  # 1% uncertainty per step
            new_qs[factor] = B_static_with_noise(parents, factor, BUTTON_POS_NOISE)

        elif factor == "red_button_state":
            new_qs[factor] = B_red_button_state(parents, action, B_NOISE_LEVEL)

        elif factor == "blue_button_state":
            new_qs[factor] = B_blue_button_state(parents, action, B_NOISE_LEVEL)

    return new_qs



