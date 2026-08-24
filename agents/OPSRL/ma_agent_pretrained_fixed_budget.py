"""
Fixed-pretraining-budget variant of MAOPSRLAgent, for the budget-sweep
protocol (run_two_opsrl_agents_pretrained_sweep.py). Same learning algorithm
as MAOPSRLAgent, inherited unchanged; distinct class purely for identity
separation, so a sweep over several pretraining budgets can never be
confused with, or share risk with, the cold-start baseline or the
convergence-based warm-start variant.

Isolation, per explicit direction: does not edit agents/OPSRL/agent.py,
agents/OPSRL/ma_agent.py, or agents/OPSRL/ma_agent_pretrained_convergence.py
-- only subclasses MAOPSRLAgent. No Active Inference agent or script is
touched by this file or anything that imports it.

Where this differs from MAOPSRLAgentPretrainedConvergence (see that file for
the shared rationale): convergence-based pretraining discovers its own
budget (stops once win rate plateaus); this variant is pretrained for an
EXTERNALLY CHOSEN, fixed number of episodes, so a run script can construct
several independent instances at different budgets (e.g. 50, 200, 500, 1500
pretraining episodes) and compare scored, post-relocation adaptation across
them -- tracing out an adaptation-quality-vs-pretraining-budget curve rather
than reporting a single warm-started number.
"""

from agents.OPSRL.ma_agent import MAOPSRLAgent


class MAOPSRLAgentPretrainedFixedBudget(MAOPSRLAgent):
    """MAOPSRLAgent, warm-started for an externally-specified fixed number
    of pretraining episodes before scoring begins.

    See run_two_opsrl_agents_pretrained_sweep.py::pretrain_pair_fixed_budget
    for the actual pretraining loop.
    """

    PARADIGM_TAG = "pretrained_sweep"
