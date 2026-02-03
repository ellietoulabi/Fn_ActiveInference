"""
Independent paradigm (baseline).

Each agent runs its own *single-agent* generative model and selects only its own action.
The other agent is treated as part of the environment (no explicit modelling / planning about them).

"""

from .A import A_fn
from .B import B_fn
from .C import C_fn
from .D import D_fn
from . import model_init
from . import env_utils

__all__ = ["A_fn", "B_fn", "C_fn", "D_fn", "model_init", "env_utils"]


