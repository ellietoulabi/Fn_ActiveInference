"""
Fully Collective paradigm for Two-Agent RedBlueButton.

One AIF agent (central planner) and one follower: the AIF chooses the full joint
action (a1, a2); the follower only executes the action the AIF assigns to it.
Centralized generative model over joint state and joint action space; policies
minimize expected free energy over joint trajectories. Upper bound on coordination
(reference for maximal coordination); assumes centralized control.
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
