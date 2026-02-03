"""
IndividuallyCollective paradigm.

Each agent plans using the same joint model (joint actions), but executes only its own
component of the joint action. Coordination can emerge without explicit communication.

Implementation note:
This reuses the FullyCollective model components.
"""

# Reuse FullyCollective model components
from ..FullyCollective import A_fn, B_fn, C_fn, D_fn, model_init, env_utils

__all__ = ["A_fn", "B_fn", "C_fn", "D_fn", "model_init", "env_utils"]


