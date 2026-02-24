"""
Individually Collective paradigm for Two-Agent RedBlueButton.

Each agent maintains the same joint generative model (state + joint actions) as
FullyCollective but executes only its own action component. Coordination emerges
through shared model structure: both reason over joint state and joint actions,
then marginalize to P(a1) and P(a2) and act. Supports heterogeneous partners
(one AIF agent can use step_individual_component while the other is non-AIF).
"""

from . import model_init
from . import env_utils
from .A import A_fn
from .B import B_fn
from .C import C_fn
from .D import D_fn

__all__ = [
    "model_init",
    "env_utils",
    "A_fn",
    "B_fn",
    "C_fn",
    "D_fn",
]
