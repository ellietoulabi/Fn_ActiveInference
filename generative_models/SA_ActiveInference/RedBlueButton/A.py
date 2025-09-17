import jax.numpy as jnp

def A_agent_pos(state, S):
    """Observation = agent position index"""
    agent_pos, *_ = state
    dist = jnp.zeros(S)
    return dist.at[agent_pos].set(1.0)

def A_on_red_button(state, S):
    agent_pos, red_pos, *_ = state
    return jnp.array([1.0, 0.0]) if agent_pos != red_pos else jnp.array([0.0, 1.0])

def A_on_blue_button(state, S):
    agent_pos, _, blue_pos, *_ = state
    return jnp.array([1.0, 0.0]) if agent_pos != blue_pos else jnp.array([0.0, 1.0])

def A_red_button_state(state, S):
    _, _, _, red_state, *_ = state
    return jnp.array([1.0, 0.0]) if red_state == 0 else jnp.array([0.0, 1.0])

def A_blue_button_state(state, S):
    _, _, _, _, blue_state, *_ = state
    return jnp.array([1.0, 0.0]) if blue_state == 0 else jnp.array([0.0, 1.0])

def A_game_result(state, S):
    _, _, _, red_state, blue_state, goal_ctx = state
    if red_state == 1 and blue_state == 1:
        return jnp.array([0.0, 1.0, 0.0]) if goal_ctx == 0 else jnp.array([0.0, 0.0, 1.0])
    return jnp.array([1.0, 0.0, 0.0])  # neutral

def A_button_just_pressed(prev_state, state, S):
    # Example: if any button changed from not_pressed → pressed
    red_changed = prev_state[3] == 0 and state[3] == 1
    blue_changed = prev_state[4] == 0 and state[4] == 1
    return jnp.array([0.0, 1.0]) if (red_changed or blue_changed) else jnp.array([1.0, 0.0])

# Bundle A
A_funcs = {
    "agent_pos": A_agent_pos,
    "on_red_button": A_on_red_button,
    "on_blue_button": A_on_blue_button,
    "red_button_state": A_red_button_state,
    "blue_button_state": A_blue_button_state,
    "game_result": A_game_result,
    # Note: button_just_pressed depends on prev+curr state
}
