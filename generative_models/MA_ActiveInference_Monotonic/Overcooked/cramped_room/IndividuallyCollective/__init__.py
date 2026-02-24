"""
IndividuallyCollective paradigm.

Each agent plans using the same joint model (joint actions), but executes only its own
component of the joint action. Coordination can emerge without explicit communication.

Implementation note:
This reuses the FullyCollective model components.
"""

# Reuse FullyCollective model components, but use this package's env_utils
from ..FullyCollective import A_fn, B_fn, C_fn, D_fn, model_init  # noqa: F401
from . import env_utils  # noqa: F401

__all__ = ["A_fn", "B_fn", "C_fn", "D_fn", "model_init", "env_utils"]


