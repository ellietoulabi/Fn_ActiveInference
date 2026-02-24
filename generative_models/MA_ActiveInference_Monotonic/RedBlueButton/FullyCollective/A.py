"""
Observation model (A) for FullyCollective paradigm (JOINT state).
"""

import numpy as np
from . import model_init

# Keep small observation noise for positions (SA-like)
A_NOISE_LEVEL = 0.01


def _noisy_pos_obs(pos_idx, num_obs, noise_level=A_NOISE_LEVEL):
    p = np.full(num_obs, noise_level / max(1, (num_obs - 1)), dtype=float)
    p[pos_idx] = 1.0 - noise_level
    return p


def A_fn(state_indices):
    """
    Compute observation likelihoods for the full joint observation.
    """
    a1 = int(state_indices["agent1_pos"])
    a2 = int(state_indices["agent2_pos"])
    red_pos = int(state_indices["red_button_pos"])
    blue_pos = int(state_indices["blue_button_pos"])
    red_state = int(state_indices["red_button_state"])
    blue_state = int(state_indices["blue_button_state"])

    S = model_init.S
    obs = {}

    obs["agent1_pos"] = _noisy_pos_obs(a1, S)
    obs["agent2_pos"] = _noisy_pos_obs(a2, S)

    obs["agent1_on_red_button"] = np.array([0.0, 1.0]) if a1 == red_pos else np.array([1.0, 0.0])
    obs["agent1_on_blue_button"] = np.array([0.0, 1.0]) if a1 == blue_pos else np.array([1.0, 0.0])
    obs["agent2_on_red_button"] = np.array([0.0, 1.0]) if a2 == red_pos else np.array([1.0, 0.0])
    obs["agent2_on_blue_button"] = np.array([0.0, 1.0]) if a2 == blue_pos else np.array([1.0, 0.0])

    obs["red_button_state"] = np.zeros(2)
    obs["red_button_state"][red_state] = 1.0
    obs["blue_button_state"] = np.zeros(2)
    obs["blue_button_state"][blue_state] = 1.0

    obs["game_result"] = np.zeros(3)
    if blue_state == 1:
        obs["game_result"][1 if red_state == 1 else 2] = 1.0
    else:
        obs["game_result"][0] = 1.0

    # Dynamic observation; approximate using "on any button" heuristic (SA-like)
    on_any_button = (a1 == red_pos) or (a1 == blue_pos) or (a2 == red_pos) or (a2 == blue_pos)
    obs["button_just_pressed"] = np.array([0.5, 0.5]) if on_any_button else np.array([1.0, 0.0])

    return obs


# =============================================================================
# Inference utilities: apply A to belief distributions
# =============================================================================

def get_observation_likelihood(modality, state_indices):
    """
    Get observation likelihood for a specific (joint) state configuration.
    """
    obs_likelihoods = A_fn(state_indices)
    return np.array(obs_likelihoods[modality], dtype=float)


def predict_obs_from_beliefs(modality, state_beliefs, prev_state_beliefs=None):
    """
    Predicted observation distribution: q(o) = Σ_s q(s) p(o|s)
    over joint state with factorized beliefs.
    """
    import itertools
    state_factors = list(model_init.states.keys())
    factor_sizes = [len(state_beliefs[f]) for f in state_factors]
    num_obs = len(model_init.observations[modality])
    q_obs = np.zeros(num_obs, dtype=float)

    for state_combo in itertools.product(*[range(s) for s in factor_sizes]):
        state_indices = {f: state_combo[i] for i, f in enumerate(state_factors)}
        joint_prob = 1.0
        for i, f in enumerate(state_factors):
            joint_prob *= float(state_beliefs[f][state_combo[i]])
        if joint_prob < 1e-12:
            continue
        p_obs = get_observation_likelihood(modality, state_indices)
        q_obs += joint_prob * p_obs

    s = np.sum(q_obs)
    return q_obs / s if s > 0 else q_obs


def predict_all_obs_from_beliefs(state_beliefs, prev_state_beliefs=None):
    """Predict observation distributions for all modalities from joint state beliefs."""
    obs_predictions = {}
    for modality in model_init.observation_state_dependencies.keys():
        obs_predictions[modality] = predict_obs_from_beliefs(
            modality, state_beliefs, prev_state_beliefs
        )
    return obs_predictions

