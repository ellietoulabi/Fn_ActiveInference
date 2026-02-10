"""
Multi-Agent Active Inference generative models for the TwoAgentRedBlueButton environment.

This package contains multiple modelling / coordination paradigms:

- `Independent`:
    Each agent runs its own *single-agent* model and selects only its own action.
    The other agent is treated as part of the environment.

- `FullyCollective`:
    A centralized planner selects a JOINT action (a1, a2) encoded as a single action index.

- `IndividuallyCollective`:
    Each agent plans using the same joint model (joint actions), but executes only its own
    component of the joint action. Coordination can emerge without explicit communication.

Default export: `FullyCollective`. For other paradigms, import explicitly from the
subpackage, e.g. `from ...RedBlueButton.Independent import A_fn, ...`
"""

# Default export: FullyCollective (centralized joint planner)
from .FullyCollective import A_fn, B_fn, C_fn, D_fn, model_init, env_utils

__all__ = [
    "A_fn",
    "B_fn",
    "C_fn",
    "D_fn",
    "model_init",
    "env_utils",
    "Independent",
    "FullyCollective",
    "IndividuallyCollective",
]

# Expose subpackages for explicit imports
from . import Independent, FullyCollective, IndividuallyCollective


