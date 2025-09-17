import jax.numpy as jnp
import numpy as np
import model_init as mi



# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize(p):
    return p / jnp.maximum(jnp.sum(p), 1e-8)

def grid_to_pos(width, height):
    """Return a list of (x,y) positions in row-major order."""
    return [(x, y) for y in range(height) for x in range(width)]

def prob_adjacent(q_agent_pos, q_door_pos, width, height):
    """Probability agent is adjacent to door (works for width × height grid)."""
    S = width * height
    total_prob = 0.0
    for i in range(S):
        x1, y1 = i % width, i // width
        for j in range(S):
            x2, y2 = j % width, j // width
            is_adj = (abs(x1 - x2) + abs(y1 - y2) == 1)
            total_prob += jnp.where(is_adj, q_agent_pos[i] * q_door_pos[j], 0.0)
    return total_prob

def update_agent_pos(q_pos, action, width, height, noise=0.05):
    """Move agent on a width×height grid."""
    S = q_pos.shape[0]
    new_pos = jnp.zeros_like(q_pos)
    for i in range(S):
        prob = q_pos[i]
        x, y = i % width, i // width
        if action == 0:    # UP
            y = max(0, y - 1)
        elif action == 1:  # DOWN
            y = min(height - 1, y + 1)
        elif action == 2:  # LEFT
            x = max(0, x - 1)
        elif action == 3:  # RIGHT
            x = min(width - 1, x + 1)
        # 4=OPEN, 5=NOOP: no move
        target_idx = y * width + x
        new_pos = new_pos.at[target_idx].add(prob * (1.0 - noise))
    new_pos += noise / S
    return normalize(new_pos)

def update_door_state(q_state, q_agent_pos, q_door_pos, action,
                      width, height, open_success=0.8, dep_open_prob=None):
    """Update door state based on adjacency and action. Optionally depend on another door being open."""
    if action != 4:  # OPEN
        return q_state
    p_adj = prob_adjacent(q_agent_pos, q_door_pos, width, height)
    p_open = open_success * p_adj
    if dep_open_prob is not None:
        p_open *= dep_open_prob
    closed, opened = q_state
    new_closed = closed * (1.0 - p_open)
    new_open = opened + closed * p_open
    return normalize(jnp.array([new_closed, new_open]))


# -------------------------------------------------
# Functional B update
# -------------------------------------------------
def apply_B(qs, action, width, height):
    new_pos = update_agent_pos(qs["agent_pos"], action, width, height)

    new_red_door_pos = qs["red_door_pos"]  # static
    new_blue_door_pos = qs["blue_door_pos"]  # static

    new_red_state = update_door_state(
        qs["red_door_state"], qs["agent_pos"], qs["red_door_pos"], action,
        width, height
    )
    new_blue_state = update_door_state(
        qs["blue_door_state"], qs["agent_pos"], qs["blue_door_pos"], action,
        width, height, dep_open_prob=qs["red_door_state"][1]  # blue depends on red
    )

    new_goal = qs["goal_context"]

    return {
        "agent_pos": new_pos,
        "red_door_pos": new_red_door_pos,
        "blue_door_pos": new_blue_door_pos,
        "red_door_state": new_red_state,
        "blue_door_state": new_blue_state,
        "goal_context": new_goal,
    }

