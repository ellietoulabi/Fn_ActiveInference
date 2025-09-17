import jax.numpy as jnp

def make_C_prefs():
    """Return functional C: preference distributions over each modality."""

    def C_agent_pos(obs_idx):
        # Neutral about where the agent is
        return 0.0

    def C_on_red_button(obs_idx):
        return 0.1 if obs_idx == 1 else 0.0  # 1 = TRUE

    def C_on_blue_button(obs_idx):
        return 0.1 if obs_idx == 1 else 0.0  # 1 = TRUE

    def C_red_button_state(obs_idx):
        # Slight encouragement to press red button
        return 0.2 if obs_idx == 1 else 0.0

    def C_blue_button_state(obs_idx):
        # Slight encouragement to press blue button
        return 0.2 if obs_idx == 1 else 0.0

    def C_game_result(obs_idx):
        if obs_idx == 1:   # win
            return 1.0
        elif obs_idx == 2: # lose
            return -1.0
        else:
            return 0.0     # neutral

    def C_button_just_pressed(obs_idx):
        return 0.05 if obs_idx == 0 else 0.0

    return {
        "agent_pos": C_agent_pos,
        "on_red_button": C_on_red_button,
        "on_blue_button": C_on_blue_button,
        "red_button_state": C_red_button_state,
        "blue_button_state": C_blue_button_state,
        "game_result": C_game_result,
        "button_just_pressed": C_button_just_pressed,
    }
