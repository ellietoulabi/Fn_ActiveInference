"""
Warm-started variant of MAOPSRLAgent: identical learning algorithm
(Thompson-sampled posterior + backward induction, inherited unchanged), with
a distinct class identity so warm-started agents can never be confused with,
or accidentally share code-path risk with, the cold-start baseline.

Isolation, per explicit direction: this file does not edit, and never will
edit, agents/OPSRL/agent.py (Stage 1's shared single-agent class, behind the
already-reported nine-agent table) or agents/OPSRL/ma_agent.py (the
cold-start MAOPSRLAgent, behind the already-run 30-seed cluster comparison
in ai/02-debug.md). It only subclasses MAOPSRLAgent. No Active Inference
agent or script is touched by this file or anything that imports it.

Motivation (see ai/02-debug.md, MA Red-Blue-Button, for the fuller
discussion): the cold-start protocol conflates two different questions --
"can OPSRL learn navigation/press mechanics at all" and "can OPSRL adapt
when the button relocates" -- and only the second is what H1/H2 actually
claim to test (AIF is deployed with a hand-specified, already-correct
model, so it never has to solve the first problem). This class exists so a
run script can give OPSRL a domain-randomized pretraining phase BEFORE the
real, scored evaluation protocol begins -- letting its posterior converge on
general task mechanics first -- while still leaving per-config button-
location belief to be genuinely discovered during the scored protocol,
exactly as AIF must also discover it per config. This mirrors the existing
MAPPO --mode pretrained/online precedent (run_two_ppo_agents.py) for the
same underlying fairness question.

All of the actual episode-stepping mechanics (resample_and_plan,
_get_action, _update, _obs_to_state) are inherited unchanged from
MAOPSRLAgent -- a "pretraining episode" and a "scored episode" are
mechanically identical calls into this class. What differs is entirely in
the driving run script (run_two_opsrl_agents_pretrained.py): which
env/config sequence it points episodes at, and whether an episode's result
counts toward reported metrics. Nothing about that protocol lives on this
class, deliberately, so the class stays a trivial, low-risk subclass.
"""

from agents.OPSRL.ma_agent import MAOPSRLAgent


class MAOPSRLAgentPretrainedConvergence(MAOPSRLAgent):
    """MAOPSRLAgent, warm-started to convergence before scoring begins.

    See run_two_opsrl_agents_pretrained.py::pretrain_pair_to_convergence for
    the actual pretraining loop (which drives two instances of this class
    jointly against a shared env, same as the cold-start runner drives two
    MAOPSRLAgent instances).
    """

    PARADIGM_TAG = "pretrained_convergence"
