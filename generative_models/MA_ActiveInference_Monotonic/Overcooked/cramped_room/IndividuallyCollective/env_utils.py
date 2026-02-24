"""
Env <-> model utilities for the IndividuallyCollective paradigm - Cramped Room.

This paradigm uses the same JOINT model and helpers as FullyCollective, but each
agent reasons over joint actions and then executes only its own action component.

Observation perspective:
- Each agent can receive an observation "from their perspective" so that
  (agent1_pos, agent2_pos) in the model means (my_pos, other_pos) for that agent.
  This allows agents to share the same joint model structure while maintaining
  different beliefs and preferences in multi-agent experiments.
"""

import numpy as np

from ..FullyCollective.env_utils import *  # noqa: F401,F403
from . import model_init


def env_obs_to_model_obs_for_agent(env_state, reward_info, agent_id):
    """
    Convert OvercookedState to model observation indices from a specific agent's
    perspective.

    So that in the JOINT model:
    - agent1_* corresponds to \"me\"
    - agent2_* corresponds to \"other\"

    For agent 1 (agent_id=1): keep the original mapping.
    For agent 2 (agent_id=2): swap agent1/agent2 roles so that in the model
    agent1_* = env's player 2, agent2_* = env's player 1.

    Returns a dict in the same format as FullyCollective.env_obs_to_model_obs,
    so the same JOINT generative model (A, B, C, D) can be used; Agent 2 just
    receives a different perspective.
    """
    base = env_obs_to_model_obs(env_state, reward_info=reward_info)
    if agent_id == 1:
        return base

    # Agent 2: swap so that in the model \"agent1\" = me (env player 2),
    # and \"agent2\" = other (env player 1).
    return {
        "agent1_pos": base["agent2_pos"],
        "agent2_pos": base["agent1_pos"],
        "agent1_orientation": base["agent2_orientation"],
        "agent2_orientation": base["agent1_orientation"],
        "agent1_held": base["agent2_held"],
        "agent2_held": base["agent1_held"],
        "pot_state": base["pot_state"],
        "soup_delivered": base["soup_delivered"],
    }


def sample_my_component(
    q_pi,
    policies,
    my_agent_id,
    n_actions: int | None = None,
    action_selection: str = "deterministic",
    alpha: float = 16.0,
) -> int:
    """
    Marginalize the joint policy posterior to the distribution over my action
    component and return a single primitive action index in [0, n_actions-1].

    Used in IndividuallyCollective: each agent has the same joint model and
    q_pi over joint actions; this returns only \"my\" component (a1 or a2).

    Args
    ----
    q_pi : array-like
        1D array of policy posterior probabilities (over joint policies).
    policies : list
        List of policies, each policy is a list of joint action indices
        (we use only the first timestep: policies[i][0]).
    my_agent_id : int
        0 for agent 1 (a1 = joint // N_ACTIONS),
        1 for agent 2 (a2 = joint % N_ACTIONS).
    n_actions : int, optional
        Number of primitive actions per agent (default: model_init.N_ACTIONS).
    action_selection : {\"deterministic\", \"stochastic\"}
        - \"deterministic\": return argmax of the marginal.
        - \"stochastic\": sample from a sharpened marginal (precision alpha).
    alpha : float
        Precision for stochastic sampling (higher -> greedier).

    Returns
    -------
    int
        Primitive action index in [0, n_actions-1].
    """
    if n_actions is None:
        n_actions = model_init.N_ACTIONS

    q_pi = np.asarray(q_pi, dtype=float)
    marginal = np.zeros(n_actions, dtype=float)

    for pol_idx, policy in enumerate(policies):
        joint = int(policy[0])
        if my_agent_id == 0:
            a = joint // n_actions
        else:
            a = joint % n_actions
        marginal[a] += q_pi[pol_idx]

    total = float(np.sum(marginal))
    if total <= 0.0 or not np.isfinite(total):
        # Fallback to uniform if something goes wrong
        return int(np.random.randint(0, n_actions))

    marginal /= total

    if action_selection == "deterministic":
        return int(np.argmax(marginal))

    # Stochastic: sharpen with precision alpha and sample
    log_m = np.log(marginal + 1e-16)
    p = np.exp(log_m * alpha)
    p /= np.sum(p)
    return int(np.random.choice(n_actions, p=p))


