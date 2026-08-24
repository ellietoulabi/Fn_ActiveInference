"""
Warm-started variant of OPSRLAgent (Stage 1, single-agent): identical learning
algorithm (Thompson-sampled posterior + backward induction, inherited
unchanged, including the 2026-08-22 transition-normalization fix -- see
ai/02-debug.md), with a distinct class identity so warm-started agents can
never be confused with, or accidentally share code-path risk with, the
cold-start baseline behind the already-reported nine-agent table.

Isolation, mirroring agents/OPSRL/ma_agent_pretrained_convergence.py's own
isolation note: this file does not edit agents/OPSRL/agent.py or
run_scripts_red_blue_doors/compare_agents/compare_nine_agents.py -- only
subclasses OPSRLAgent. No Active Inference agent or script is touched by
this file or anything that imports it.

Motivation (same underlying fairness question as the MA variant, and as
run_two_ppo_agents.py's --mode pretrained/online precedent): the cold-start
protocol conflates "can OPSRL learn navigation/press mechanics at all" with
"can OPSRL adapt when the button relocates" -- and only the second is what
H1 actually claims to test, since AIF is deployed with a hand-specified,
already-correct generative model and never has to solve the first problem
from scratch. This class exists so a run script can give OPSRL a
domain-randomized pretraining phase BEFORE the real, scored evaluation
protocol begins -- letting its posterior converge on general task mechanics
first -- while still leaving per-config button-location belief to be
genuinely discovered during the scored protocol itself, exactly as AIF must
also discover it per config (belief retention within a config, no privileged
reset across relocations).

All of the actual episode-stepping mechanics (_run_episode, _get_action,
_update, _obs_to_state, fit) are inherited unchanged from OPSRLAgent -- a
"pretraining episode" and a "scored episode" are mechanically identical
calls into this class. What differs is entirely in the driving run script
(run_nonstationary_opsrl_pretrained.py): which env/config sequence it points
episodes at, and whether an episode's result counts toward reported metrics.
"""

from agents.OPSRL.agent import OPSRLAgent


class OPSRLAgentPretrainedConvergence(OPSRLAgent):
    """OPSRLAgent, warm-started to convergence before scoring begins.

    See run_nonstationary_opsrl_pretrained.py::pretrain_to_convergence for
    the actual pretraining loop.
    """

    PARADIGM_TAG = "pretrained_convergence"
