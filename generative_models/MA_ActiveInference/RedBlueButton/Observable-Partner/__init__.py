"""
Multi-Agent Active Inference Generative Model for RedBlueButton Environment.

Each agent has its own model where:
- State includes: my position, other agent's position (observed), button positions, button states
- Other agent's position is observed directly (not inferred)
"""

from .A import A_fn
from .B import B_fn
from .C import C_fn
from .D import D_fn
from . import model_init
from . import env_utils

__all__ = ['A_fn', 'B_fn', 'C_fn', 'D_fn', 'model_init', 'env_utils']
