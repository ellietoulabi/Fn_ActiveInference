"""
FullyCollective paradigm.

A centralized planner selects a JOINT action (a1, a2) encoded as a single action index.
Both agents execute the same joint action plan.
"""

from .A import A_fn
from .B import B_fn
from .C import C_fn
from .D import D_fn
from . import model_init
from . import env_utils

__all__ = ["A_fn", "B_fn", "C_fn", "D_fn", "model_init", "env_utils"]


