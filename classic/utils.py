import numpy as np
import itertools


def construct_policies(actions: list = [], policy_len: int = 1) -> np.ndarray:
    """
    Generate all possible policies (action sequences) of given length.
    Each policy is a list of actions.
    """
    return np.array(list(itertools.product(actions, repeat=policy_len)), dtype=object)


def format_observations(observations: list, num_obs: list) -> np.ndarray:
    """
    observations: list of integers, one per modality, each integer is an observation index
    num_obs: list of integers, number of observations per modality
    Returns: list of one-hot encoded observations, one per modality
    """
    one_hot_observations = []
    for obs_idx, obs in enumerate(observations):
        one_hot_obs = np.zeros(num_obs[obs_idx])
        one_hot_obs[obs] = 1
        one_hot_observations.append(one_hot_obs)

    return np.array(one_hot_observations,dtype=object)


def sample(probabilities):
    """
    Sample from the probabilities.
    Example:
        probabilities = [0.1, 0.2, 0.3, 0.4]
        with chance of 0.1, sample action 0
        with chance of 0.2, sample action 1
        with chance of 0.3, sample action 2
        with chance of 0.4, sample action 3

    Args:
        probabilities (np.ndarray): probabilities of the actions

    Returns:
       int: index of the sampled action
    """
    probabilities = probabilities.squeeze() if len(probabilities) > 1 else probabilities #remove extra dimensions
    sample_onehot = np.random.multinomial(1, probabilities) # sample from the probabilities one-hot encoded (like 1 time fliping a coin)
    return np.where(sample_onehot == 1)[0][0] # return the index of the sampled action