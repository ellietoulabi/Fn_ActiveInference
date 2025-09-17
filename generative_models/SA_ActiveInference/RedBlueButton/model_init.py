import jax.numpy as jnp


# -------------------------------------------------
# Grid size
# -------------------------------------------------
n, m = 3, 3
S = n * m  # total cells

# -------------------------------------------------
# Actions
# -------------------------------------------------
UP, DOWN, LEFT, RIGHT, OPEN, NOOP = 0, 1, 2, 3, 4, 5



# States
States = {
    "agent_pos": list(range(S)),  # 0..S-1 positions
    "red_button_pos": list(range(S)),
    "blue_button_pos": list(range(S)),
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
    "goal_context": ["red_then_blue", "blue_then_red"],
}

Observations = {
    "agent_pos": list(range(S)),
    "on_red_button": ["FALSE", "TRUE"],
    "on_blue_button": ["FALSE", "TRUE"],
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
    "game_result": ["neutral", "win", "lose"],
    "button_just_pressed": ["TRUE", "FALSE"],
}


