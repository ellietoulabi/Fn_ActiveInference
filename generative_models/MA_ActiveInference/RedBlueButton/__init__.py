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

- `ObservablePartner` (decentralised, optional extra):
    Each agent represents the partner position (`other_pos`) explicitly but still selects
    only its own action (6 actions).

Backwards compatibility:
This module re-exports the `ObservablePartner` variant by default so existing imports like:
`from generative_models.MA_ActiveInference.RedBlueButton import A_fn, ...`
continue to work.
"""

# Default / backwards-compatible export: ObservablePartner
from .ObservablePartner import A_fn, B_fn, C_fn, D_fn, model_init, env_utils

__all__ = [
    "A_fn",
    "B_fn",
    "C_fn",
    "D_fn",
    "model_init",
    "env_utils",
    # Variants
    "Independent",
    "FullyCollective",
    "IndividuallyCollective",
    "ObservablePartner",
]

# Expose subpackages for explicit imports
from . import Independent, FullyCollective, IndividuallyCollective, ObservablePartner


