"""
IndividuallyCollective paradigm uses the same env_utils as FullyCollective.

Execution-time logic: each agent reasons over the joint model but executes only
its own action component. Helpers here support marginalizing the joint policy
posterior to P(a1) or P(a2) and sampling/argmax.

Observation perspective: each agent can receive an observation "from their perspective"
so that (agent1_pos, agent2_pos) means (my_pos, other_pos). Then agent 1 and agent 2
get different observation values and can form different beliefs.
"""

import numpy as np
from ..FullyCollective.env_utils import *  # noqa: F401,F403
from . import model_init


def env_obs_to_model_obs_for_agent(joint_obs, width, agent_id):
    """
    Convert env observation to model observation from a specific agent's perspective.

    So that (agent1_pos, agent2_pos) in the model = (my_pos, other_pos) for that agent.
    - Agent 1 (agent_id=1): model sees agent1_pos=my_pos, agent2_pos=other_pos (unchanged).
    - Agent 2 (agent_id=2): model sees agent1_pos=my_pos, agent2_pos=other_pos by
      swapping env's agent1/agent2, so agent1_pos=env's agent2, agent2_pos=env's agent1.

    Returns a dict in the same format as env_obs_to_model_obs, so the same
    generative model (A, B, C, D) can be used; Agent 2 just gets different input.
    """
    base = env_obs_to_model_obs(joint_obs, width=width)
    if agent_id == 1:
        return base
    # Agent 2: swap so that in the model "agent1" = me (env's agent2), "agent2" = other (env's agent1)
    return {
        "agent1_pos": base["agent2_pos"],
        "agent2_pos": base["agent1_pos"],
        "agent1_on_red_button": base["agent2_on_red_button"],
        "agent1_on_blue_button": base["agent2_on_blue_button"],
        "agent2_on_red_button": base["agent1_on_red_button"],
        "agent2_on_blue_button": base["agent1_on_blue_button"],
        "red_button_state": base["red_button_state"],
        "blue_button_state": base["blue_button_state"],
        "game_result": base["game_result"],
        "button_just_pressed": base["button_just_pressed"],
    }


def sample_my_component(q_pi, policies, my_agent_id, n_actions=None, action_selection="deterministic", alpha=16.0):
    """
    Marginalize the joint policy posterior to the distribution over my action
    component and return a single action in [0, n_actions-1].

    Used in Individually Collective: each agent has the same joint model and
    q_pi over joint actions; this returns only "my" component (a1 or a2).

    Args:
        q_pi: 1D array of policy posterior probabilities (over joint actions).
        policies: list of policies, each policy is a list of joint action indices
            (first step only is used: policies[i][0]).
        my_agent_id: 0 for agent 1 (a1 = joint // N_ACTIONS), 1 for agent 2 (a2 = joint % N_ACTIONS).
        n_actions: number of primitive actions per agent (default from model_init.N_ACTIONS).
        action_selection: "deterministic" (argmax) or "stochastic" (sample).
        alpha: precision for stochastic sampling.

    Returns:
        action: int in [0, n_actions-1].
    """
    if n_actions is None:
        n_actions = model_init.N_ACTIONS
    q_pi = np.asarray(q_pi)
    marginal = np.zeros(n_actions)
    for pol_idx, policy in enumerate(policies):
        joint = int(policy[0])
        if my_agent_id == 0:
            a = joint // n_actions
        else:
            a = joint % n_actions
        marginal[a] += q_pi[pol_idx]
    marginal = marginal / (np.sum(marginal) + 1e-16)
    if action_selection == "deterministic":
        return int(np.argmax(marginal))
    # stochastic
    log_m = np.log(marginal + 1e-16)
    p = np.exp(log_m * alpha)
    p = p / np.sum(p)
    return int(np.random.choice(n_actions, p=p))


