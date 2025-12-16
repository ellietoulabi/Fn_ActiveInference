"""
IndividuallyCollective paradigm uses the same env_utils as FullyCollective.

Execution-time logic differs (each agent takes its own component of a joint action),
but conversion utilities and joint action encoding are shared.
"""

from ..FullyCollective.env_utils import *  # noqa: F401,F403


