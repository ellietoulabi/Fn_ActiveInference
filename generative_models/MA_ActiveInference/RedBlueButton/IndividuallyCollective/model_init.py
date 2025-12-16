"""
IndividuallyCollective paradigm uses the same JOINT model as FullyCollective.

Difference is in execution (run-time): each agent plans over joint actions but
executes only its own component.
"""

from ..FullyCollective.model_init import *  # noqa: F401,F403


