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
states = {
    "agent_pos": list(range(S)),  # 0..S-1 positions
    "red_button_pos": list(range(S)),
    "blue_button_pos": list(range(S)),
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
    # "goal_context": ["red_then_blue", "blue_then_red"],
}

observations = {
    "agent_pos": list(range(S)),
    "on_red_button": ["FALSE", "TRUE"],
    "on_blue_button": ["FALSE", "TRUE"],
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
    "game_result": ["neutral", "win", "lose"],
    "button_just_pressed": ["TRUE", "FALSE"],
}

observation_state_dependencies = {
    "agent_pos": ["agent_pos"],

    "on_red_button": ["agent_pos", "red_button_pos"],
    "on_blue_button": ["agent_pos", "blue_button_pos"],

    "red_button_state": ["red_button_state"],
    "blue_button_state": ["blue_button_state"],

    "game_result": ["red_button_state", "blue_button_state"],

    "button_just_pressed": ["agent_pos", "red_button_pos", "blue_button_pos", "red_button_state", "blue_button_state"]
}

state_state_dependencies = {
    "agent_pos": ["agent_pos"],

    "red_button_pos": ["red_button_pos"],
    "blue_button_pos": ["blue_button_pos"],

    "red_button_state": ["agent_pos", "red_button_pos", "red_button_state"],
    "blue_button_state": ["agent_pos", "blue_button_pos", "blue_button_state"],

    # "goal_context":  ["goal_context"],
}