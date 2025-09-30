import jax.numpy as jnp

# Noise parameters
A_NOISE_LEVEL = 0.01  # 1% noise level
EPSILON = 1e-8  # Small constant to avoid division by zero






def add_noise_to_distribution(clean_dist, noise_level=A_NOISE_LEVEL):
    """Add uniform noise to a probability distribution and normalize"""
    # Add uniform noise to all elements
    noisy_dist = clean_dist + noise_level
    
    # Normalize to ensure it sums to 1
    return noisy_dist / jnp.sum(noisy_dist)

def add_noise_to_binary(clean_dist, noise_level=A_NOISE_LEVEL):
    """Add noise to binary distributions (2-element vectors)"""
    # For binary distributions, we can add noise more carefully
    # Add noise to the "wrong" element and subtract from the "right" element
    is_first_larger = clean_dist[0] > clean_dist[1]
    noisy = jnp.where(is_first_larger, 
                      jnp.array([1.0 - noise_level, noise_level]),
                      jnp.array([noise_level, 1.0 - noise_level]))
    
    return noisy

def add_noise_to_ternary(clean_dist, noise_level=A_NOISE_LEVEL):
    """Add noise to ternary distributions (3-element vectors)"""
    # Find the index with maximum probability
    max_idx = jnp.argmax(clean_dist)
    
    # Add noise to all elements
    noisy = clean_dist + noise_level
    
    # Subtract extra noise from the maximum element to maintain balance
    noisy = noisy.at[max_idx].set(clean_dist[max_idx] - noise_level * (len(clean_dist) - 1))
    
    # Normalize
    return noisy / jnp.sum(noisy)

def A_agent_pos_noisy(state, S, noise_level=A_NOISE_LEVEL, key=None):
    """Observation = agent position index with noise"""
    agent_pos, *_ = state
    dist = jnp.zeros(S)
    dist = dist.at[agent_pos].set(1.0)
    
    # Add noise
    noisy_dist = add_noise_to_distribution(dist, noise_level)
    return noisy_dist

def A_on_red_button_noisy(state, S, noise_level=A_NOISE_LEVEL, key=None):
    """On red button observation with noise"""
    agent_pos, red_pos, *_ = state
    clean_dist = jnp.where(agent_pos != red_pos, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    
    # Add noise to binary distribution
    return add_noise_to_binary(clean_dist, noise_level)

def A_on_blue_button_noisy(state, S, noise_level=A_NOISE_LEVEL, key=None):
    """On blue button observation with noise"""
    agent_pos, _, blue_pos, *_ = state
    clean_dist = jnp.where(agent_pos != blue_pos, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    
    # Add noise to binary distribution
    return add_noise_to_binary(clean_dist, noise_level)

def A_red_button_state_noisy(state, S, noise_level=A_NOISE_LEVEL, key=None):
    """Red button state observation with noise"""
    _, _, _, red_state, *_ = state
    clean_dist = jnp.where(red_state == 0, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    
    # Add noise to binary distribution
    return add_noise_to_binary(clean_dist, noise_level)

def A_blue_button_state_noisy(state, S, noise_level=A_NOISE_LEVEL, key=None):
    """Blue button state observation with noise"""
    _, _, _, _, blue_state, *_ = state
    clean_dist = jnp.where(blue_state == 0, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    
    # Add noise to binary distribution
    return add_noise_to_binary(clean_dist, noise_level)

def A_game_result_noisy(state, S, noise_level=A_NOISE_LEVEL, key=None):
    """Game result observation with noise"""
    _, _, _, red_state, blue_state, goal_ctx = state
    both_pressed = jnp.logical_and(red_state == 1, blue_state == 1)
    win_dist = jnp.array([0.0, 1.0, 0.0])
    lose_dist = jnp.array([0.0, 0.0, 1.0])
    neutral_dist = jnp.array([1.0, 0.0, 0.0])
    
    # Choose win/lose based on goal context when both pressed
    when_pressed = jnp.where(goal_ctx == 0, win_dist, lose_dist)
    clean_dist = jnp.where(both_pressed, when_pressed, neutral_dist)
    
    # Add noise to ternary distribution
    return add_noise_to_ternary(clean_dist, noise_level)

def A_button_just_pressed_noisy(prev_state, state, S, noise_level=A_NOISE_LEVEL, key=None):
    """Button just pressed observation with noise"""
    red_changed = jnp.logical_and(prev_state[3] == 0, state[3] == 1)
    blue_changed = jnp.logical_and(prev_state[4] == 0, state[4] == 1)
    any_changed = jnp.logical_or(red_changed, blue_changed)
    clean_dist = jnp.where(any_changed, jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0]))
    
    # Add noise to binary distribution
    return add_noise_to_binary(clean_dist, noise_level)

# Bundle noisy A functions
A_funcs_noisy = {
    "agent_pos": A_agent_pos_noisy,
    "on_red_button": A_on_red_button_noisy,
    "on_blue_button": A_on_blue_button_noisy,
    "red_button_state": A_red_button_state_noisy,
    "blue_button_state": A_blue_button_state_noisy,
    "game_result": A_game_result_noisy,
    # Note: button_just_pressed depends on prev+curr state
}

