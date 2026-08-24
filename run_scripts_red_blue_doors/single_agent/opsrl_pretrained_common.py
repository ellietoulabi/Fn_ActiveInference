"""
Shared, protocol-agnostic helpers for the two SA (Stage 1, single-agent)
OPSRL pretrained-variant run scripts (run_nonstationary_opsrl_pretrained.py
and run_nonstationary_opsrl_pretrained_sweep.py). Both new files, so sharing
this module is ordinary DRY within new work -- not the "isolate before
touching shared/already-reported code" concern that applies to
agents/OPSRL/agent.py or compare_nine_agents.py, neither of which this
module imports from or edits.

Config generation matches compare_nine_agents.py's own convention exactly
(random red/blue cell from 1-8, avoiding the agent's fixed start cell 0) so
pretraining/scoring configs are drawn from the same distribution the
already-reported nine-agent table uses -- just from an independent RNG
stream (see each run script's own seed-splitting for how pretrain vs. eval
streams are kept separate).

run_opsrl_episode drives the agent's own posterior-resample + backward-
induction preamble externally (rather than calling agent._run_episode(),
which owns its own env.reset()/env.step() loop internally) purely so this
module can add CSV logging and a returned outcome/deadlock summary in the
same shape run_nonstationary_aif_sa_exact_test.py already established this
session. The preamble itself and the per-step hh indexing are unchanged from
(and exercise the same, now-fixed) agents/OPSRL/agent.py::_run_episode /
compare_nine_agents.py::run_opsrl_episode logic -- see ai/02-debug.md,
2026-08-22 OPSRL entry, for why sum(-2) (not sum(-1)) is required here.
"""

import csv as _csv

import numpy as np

from agents.OPSRL.utils import backward_induction_in_place, backward_induction_sd

ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PRESS', 5: 'NOOP'}


def generate_random_config(rng):
    """One (red_pos, blue_pos) config, matching compare_nine_agents.py's own
    convention: uniform over cells 1-8 (cell 0 is the fixed agent start),
    red and blue distinct."""
    available = list(range(1, 9))
    rng.shuffle(available)
    red_idx, blue_idx = available[0], available[1]
    return {
        'red_pos': (red_idx // 3, red_idx % 3),
        'blue_pos': (blue_idx // 3, blue_idx % 3),
        'red_idx': red_idx,
        'blue_idx': blue_idx,
    }


def resample_and_plan(agent):
    """Draw one Thompson sample of the transition/reward posterior and run
    backward induction -- exactly OPSRLAgent._run_episode()'s own preamble,
    duplicated here (not called via _run_episode() itself) so this module
    can drive the environment step loop externally. See module docstring."""
    B = agent.thompson_samples

    M_sab_zero = np.repeat(agent.M_sa[..., 0, np.newaxis], B, -1)
    M_sab_one = np.repeat(agent.M_sa[..., 1, np.newaxis], B, -1)
    N_sasb = np.repeat(agent.N_sas[..., np.newaxis], B, axis=-1)

    R_samples = agent.rng.beta(M_sab_zero, M_sab_one)
    P_samples = agent.rng.gamma(N_sasb)
    P_samples = P_samples + 1e-10
    # Next-state axis is always second-to-last (Thompson-sample axis B is
    # always appended last) -- see ai/02-debug.md, 2026-08-22 OPSRL entry.
    sums = P_samples.sum(-2, keepdims=True)
    P_samples = P_samples / sums

    R_samples = 2.0 * R_samples - 1.0

    if agent.stage_dependent:
        backward_induction_sd(agent.Q, agent.V, R_samples, P_samples, agent.gamma, agent.v_max[0])
    else:
        backward_induction_in_place(agent.Q, agent.V, R_samples, P_samples,
                                     agent.horizon, agent.gamma, agent.v_max[0])


def run_opsrl_episode(env, agent, max_steps=50, csv_writer=None, seed=None,
                       episode_num=None, agent_label='OPSRL', config_idx=None):
    """Run one OPSRL episode: resample posterior, plan, then step through the
    real env with correct per-step hh indexing (agents.OPSRL.agent's own
    fixed hh=min(step-1, horizon-1) convention -- matches the already-fixed
    compare_nine_agents.py::run_opsrl_episode, and _run_episode()'s own
    internal for-hh-in-range(horizon) loop)."""
    agent.env = env
    resample_and_plan(agent)

    result = env.reset()
    obs, info = result if isinstance(result, tuple) else (result, {})

    episode_reward = 0.0
    outcome = 'timeout'
    positions = []

    step = 0
    for step in range(1, max_steps + 1):
        state = agent._obs_to_state(obs)
        hh = min(step - 1, agent.horizon - 1)
        action = agent._get_action(state, hh=hh)

        step_result = env.step(action)
        next_obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        episode_reward += reward
        positions.append(env.agent_position)

        if csv_writer is not None:
            _x, _y = env.agent_position
            csv_writer.writerow({
                'seed': seed,
                'agent': agent_label,
                'episode': episode_num,
                'step': step,
                'action': action,
                'action_name': ACTION_NAMES[action],
                'reward': reward,
                'agent_pos': _y * 3 + _x,  # matches env_utils.xy_to_index(x, y, width=3)
            })

        next_state = agent._obs_to_state(next_obs) if not done else None
        agent._update(state, action, next_state, reward, hh=hh)

        obs = next_obs
        if done:
            outcome = info.get('result', 'neutral')
            break

    agent.episode += 1

    deadlock = False
    if outcome != 'win' and step >= max_steps:
        tail = positions[-20:]
        if len(set(tail)) <= 4:
            deadlock = True

    return {'outcome': outcome, 'reward': episode_reward, 'steps': step,
            'success': outcome == 'win', 'deadlock': deadlock,
            'seed': seed, 'episode': episode_num, 'config_idx': config_idx}


def open_csv_writer(path, fieldnames=('seed', 'agent', 'episode', 'step',
                                       'action', 'action_name', 'reward', 'agent_pos')):
    f = open(path, 'w', newline='')
    writer = _csv.DictWriter(f, fieldnames=list(fieldnames))
    writer.writeheader()
    return f, writer
