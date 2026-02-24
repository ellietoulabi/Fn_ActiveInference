"""
IndividuallyCollective paradigm for Overcooked - Cramped Room layout.

Uses the same JOINT generative model as FullyCollective:
- Joint hidden state over both agents (positions, orientations, held objects, pot state, soup_delivered)
- Joint actions (a1, a2) encoded as a single integer in [0, 35]

Difference is in execution time (in the run script), not in the model:
- Each agent reasons over the joint model and q_pi over joint actions
- But executes only its own action component (marginalising the joint policy posterior).
"""

from ..FullyCollective.model_init import *  # noqa: F401,F403


