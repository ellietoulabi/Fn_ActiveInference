"""
FullyCollective paradigm model init.

Centralised model: one planner chooses the full joint action.
- Joint state: both agent positions + button pos/state.
- Joint action (a1, a2) encoded as one integer in [0, 35]; a1, a2 in [0..5].
- At execution: agent_0 takes a1, agent_1 takes a2 (agent1↔agent_0, agent2↔agent_1).
"""

# -------------------------------------------------
# Grid size
# -------------------------------------------------
n, m = 3, 3
S = n * m  # total cells (9)

# -------------------------------------------------
# Primitive actions (per-agent)
# -------------------------------------------------
UP, DOWN, LEFT, RIGHT, PRESS, NOOP = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
N_JOINT_ACTIONS = N_ACTIONS * N_ACTIONS  # 36

# -------------------------------------------------
# States (JOINT)
# -------------------------------------------------
states = {
    "agent1_pos": list(range(S)),
    "agent2_pos": list(range(S)),
    "red_button_pos": list(range(S)),
    "blue_button_pos": list(range(S)),
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
}

# -------------------------------------------------
# Observations (full joint observation)
# -------------------------------------------------
observations = {
    "agent1_pos": list(range(S)),
    "agent2_pos": list(range(S)),
    "agent1_on_red_button": ["FALSE", "TRUE"],
    "agent1_on_blue_button": ["FALSE", "TRUE"],
    "agent2_on_red_button": ["FALSE", "TRUE"],
    "agent2_on_blue_button": ["FALSE", "TRUE"],
    "red_button_state": ["not_pressed", "pressed"],
    "blue_button_state": ["not_pressed", "pressed"],
    "game_result": ["neutral", "win", "lose"],
    "button_just_pressed": ["FALSE", "TRUE"],
}

observation_state_dependencies = {
    "agent1_pos": ["agent1_pos"],
    "agent2_pos": ["agent2_pos"],
    "agent1_on_red_button": ["agent1_pos", "red_button_pos"],
    "agent1_on_blue_button": ["agent1_pos", "blue_button_pos"],
    "agent2_on_red_button": ["agent2_pos", "red_button_pos"],
    "agent2_on_blue_button": ["agent2_pos", "blue_button_pos"],
    "red_button_state": ["red_button_state"],
    "blue_button_state": ["blue_button_state"],
    "game_result": ["red_button_state", "blue_button_state"],
    "button_just_pressed": [
        "agent1_pos",
        "agent2_pos",
        "red_button_pos",
        "blue_button_pos",
        "red_button_state",
        "blue_button_state",
    ],
}

state_state_dependencies = {
    # Movement depends on both positions due to collision + sequential turns.
    "agent1_pos": ["agent1_pos", "agent2_pos"],
    "agent2_pos": ["agent1_pos", "agent2_pos"],
    "red_button_pos": ["red_button_pos"],
    "blue_button_pos": ["blue_button_pos"],
    "red_button_state": ["agent1_pos", "agent2_pos", "red_button_pos", "red_button_state"],
    "blue_button_state": ["agent1_pos", "agent2_pos", "blue_button_pos", "blue_button_state"],
}