import jax.numpy as jnp
import model_init as mi


    
D = {
    # Agent position: certain at index 0
    "agent_pos": jnp.array([1.0] + [0.0] * (mi.S - 1)),

    # Doors: uniform position
    "red_door_pos": jnp.ones(mi.S) / mi.S,
    "blue_door_pos": jnp.ones(mi.S) / mi.S,

    # Door states: closed with certainty
    "red_door_state": jnp.array([1.0, 0.0]),
    "blue_door_state": jnp.array([1.0, 0.0]),

    # Goal context: uniform
    "goal_context": jnp.ones(3) / 3,
}


