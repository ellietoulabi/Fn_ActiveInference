"""
Multi-Agent Active Inference Model Initialization for RedBlueButton.

Defines state factors, observations, and their dependencies for a single agent
in the two-agent environment. Each agent treats the other agent's position
as an observed variable.
"""

import numpy as np

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
    "my_pos": list(range(S)),  # 0..S-1 positions (my position)
    "other_pos": list(range(S)),  # Other agent's position (treated as observable)
    "red_button_pos": list(range(S)),  # Red button position
    "blue_button_pos": list(range(S)),  # Blue button position
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
}

# -------------------------------------------------
# Observations
# -------------------------------------------------
observations = {
    "my_pos": list(range(S)),  # My position (directly observed)
    "other_pos": list(range(S)),  # Other agent's position (directly observed)
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
    "button_just_pressed": ["my_pos", "red_button_pos", "blue_button_pos", 
                            "red_button_state", "blue_button_state"],
}

state_state_dependencies = {
    "my_pos": ["my_pos"],  # My position depends on previous position + action
    "other_pos": ["other_pos"],  # Other's position - we observe it, don't predict
    "red_button_pos": ["red_button_pos"],  # Static
    "blue_button_pos": ["blue_button_pos"],  # Static
    "red_button_state": ["my_pos", "red_button_pos", "red_button_state"],
    "blue_button_state": ["my_pos", "blue_button_pos", "blue_button_state"],
}
