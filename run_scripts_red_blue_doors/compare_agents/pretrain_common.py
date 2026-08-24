"""
Shared pretraining harness for the "fair adaptation-speed" SA nine-agent
comparison (compare_nine_agents_pretrained.py). New file; does not edit
compare_nine_agents.py or any agent class.

Purpose (see ai/02-debug.md, 2026-08-23 "fair adaptation framework" entry):
the cold-start protocol conflates "can this baseline learn the task
mechanics at all" with "how fast does it re-adapt after a relocation" --
only the second is what the AIF-vs-RL adaptation comparison actually claims
to test, since AIF is deployed with a hand-specified, already-correct model
and never has to learn mechanics from scratch. This mirrors the pretrained
OPSRL variants built earlier this session (agents/OPSRL/agent_pretrained_*)
and the run_two_ppo_agents.py --mode pretrained/online precedent.

Important, disclosed asymmetry (not a bug -- a real property of tabular
state representations, verified directly, kept in mind when interpreting
results): none of these Q-learning-family states include button *position*
(only agent_pos + on_red/on_blue/red_pressed/blue_pressed, all locally
observable). Pretraining on a domain-randomized STREAM of configs can only
teach "press when the local observation says I should" -- a transferable
fact, true regardless of where the button physically is -- it CANNOT teach
"which direction to walk," since that is config-specific and pretraining
sees a different (random) direction as "correct" from the same tabular
state on almost every pretraining episode, so no consistent Q-value for
movement-in-a-particular-direction can be learned. Concretely: pretraining
should remove essentially all of the "learn to press correctly" cost while
leaving the "physically locate the button this time" cost fully intact --
this is the tabular-RL analogue of AIF's own per-config belief needing to
be resolved fresh at every relocation, and is expected to still take real
per-config episodes even post-pretraining, just not as many as cold-start.
"""

import numpy as np

from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv
from generative_models.SA_ActiveInference.RedBlueButton import env_utils


def generate_random_config_np(rng_state_seed=None):
    """Matches compare_nine_agents.py's own convention exactly (uniform over
    cells 1-8, avoiding the agent's fixed start cell 0), but driven by the
    passed-in numpy Generator `rng_state_seed` (a np.random.Generator) so
    pretraining draws from an independent stream from the scored-eval RNG,
    the same separation already used for OPSRL's pretrained variants."""
    available_positions = list(range(1, 9))
    rng_state_seed.shuffle(available_positions)
    red_idx, blue_idx = available_positions[0], available_positions[1]
    return {
        'red_pos': (red_idx // 3, red_idx % 3),
        'blue_pos': (blue_idx // 3, blue_idx % 3),
    }


def run_ql_family_episode(env, agent, max_steps=50):
    """One episode for a QLearningAgent or any DynaQAgent subclass, matching
    compare_nine_agents.py::run_episode's step loop exactly (same
    hasattr(agent, 'update_model') gating so plain QLearningAgent never
    accidentally gets Dyna-Q's planning bonus), minus CSV logging."""
    env_obs, _ = env.reset()
    obs_dict = env_utils.env_obs_to_model_obs(env_obs)
    state = agent.get_state(obs_dict)
    episode_reward = 0.0
    step = 0
    for step in range(1, max_steps + 1):
        action = agent.choose_action(state)
        env_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        next_obs_dict = env_utils.env_obs_to_model_obs(env_obs)
        next_state = agent.get_state(next_obs_dict) if not done else None
        agent.update_q_table(state, action, reward, next_state)
        if hasattr(agent, 'update_model'):
            agent.update_model(state, action, next_state, reward, terminated)
            agent.planning()
        state = next_state
        if done:
            break
    return episode_reward, step


def pretrain_ql_agent_to_convergence(agent, pretrain_rng, max_steps=50,
                                      window=50, patience=3, min_delta=0.02,
                                      max_episodes=1500, min_episodes=150,
                                      progress_desc=None):
    """Domain-randomized pretraining until windowed win rate plateaus --
    same convergence-detection design as OPSRLAgentPretrainedConvergence's
    driving loop (run_nonstationary_opsrl_pretrained.py), applied to any
    QLearningAgent-family instance. Epsilon decays via the agent's own
    decay_exploration() each pretraining episode, exactly as it would during
    real play -- no special-casing.

    Returns dict: n_episodes, converged (bool), final_windowed_win_rate.
    """
    history = []
    win_rate_windows = []
    episode = 0
    plateaued_count = 0
    converged = False

    while episode < max_episodes:
        episode += 1
        config = generate_random_config_np(pretrain_rng)
        env = SingleAgentRedBlueButtonEnv(
            width=3, height=3,
            red_button_pos=config['red_pos'], blue_button_pos=config['blue_pos'],
            agent_start_pos=(0, 0), max_steps=max_steps,
        )
        reward, _steps = run_ql_family_episode(env, agent, max_steps=max_steps)
        agent.decay_exploration()
        history.append(1 if reward >= 1.0 else 0)

        if episode % window == 0 and episode >= min_episodes:
            recent_wr = sum(history[-window:]) / window
            win_rate_windows.append(recent_wr)
            if len(win_rate_windows) >= 2:
                delta = abs(win_rate_windows[-1] - win_rate_windows[-2])
                if delta < min_delta:
                    plateaued_count += 1
                else:
                    plateaued_count = 0
                if plateaued_count >= patience:
                    converged = True
                    break

    if not converged:
        tail = win_rate_windows[-5:] if win_rate_windows else []
        print(f"    [pretrain] WARNING ({progress_desc}): did not converge within "
              f"max_episodes={max_episodes}; stopping anyway. Last windows: {tail}")

    return {
        'n_episodes': episode,
        'converged': converged,
        'final_windowed_win_rate': win_rate_windows[-1] if win_rate_windows else None,
    }
