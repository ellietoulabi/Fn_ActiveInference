"""
ObservablePartner (decentralised) MA model init for RedBlueButton.

Each agent has its own model where hidden state includes:
- my_pos (hidden)
- other_pos (treated as observed / exogenous)
- button positions + button states

Actions are still SINGLE-agent actions (6 actions).
"""

# -------------------------------------------------
# Grid size
# -------------------------------------------------
n, m = 3, 3
S = n * m  # total cells (9)

# -------------------------------------------------
# Actions
# -------------------------------------------------
UP, DOWN, LEFT, RIGHT, PRESS, NOOP = 0, 1, 2, 3, 4, 5

# -------------------------------------------------
# States (from this agent's perspective)
# -------------------------------------------------
states = {
    "my_pos": list(range(S)),
    "other_pos": list(range(S)),
    "red_button_pos": list(range(S)),
    "blue_button_pos": list(range(S)),
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
}

# -------------------------------------------------
# Observations
# -------------------------------------------------
observations = {
    "my_pos": list(range(S)),
    "other_pos": list(range(S)),
    "my_on_red_button": ["FALSE", "TRUE"],
    "my_on_blue_button": ["FALSE", "TRUE"],
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
    "game_result": ["neutral", "win", "lose"],
    "button_just_pressed": ["FALSE", "TRUE"],
}

# -------------------------------------------------
# Dependencies
# -------------------------------------------------
observation_state_dependencies = {
    "my_pos": ["my_pos"],
    "other_pos": ["other_pos"],
    "my_on_red_button": ["my_pos", "red_button_pos"],
    "my_on_blue_button": ["my_pos", "blue_button_pos"],
    "red_button_state": ["red_button_state"],
    "blue_button_state": ["blue_button_state"],
    "game_result": ["red_button_state", "blue_button_state"],
    "button_just_pressed": ["my_pos", "red_button_pos", "blue_button_pos", "red_button_state", "blue_button_state"],
}

state_state_dependencies = {
    "my_pos": ["my_pos"],
    "other_pos": ["other_pos"],  # observed / exogenous
    "red_button_pos": ["red_button_pos"],
    "blue_button_pos": ["blue_button_pos"],
    "red_button_state": ["my_pos", "red_button_pos", "red_button_state"],
    "blue_button_state": ["my_pos", "blue_button_pos", "blue_button_state"],
}


