"""
ObservablePartner variant (decentralised):

Each agent has a model that explicitly represents the partner position (other_pos)
as an observed factor, but still selects only its own action.
"""

from .A import A_fn
from .B import B_fn
from .C import C_fn
from .D import D_fn
from . import model_init
from . import env_utils

__all__ = ["A_fn", "B_fn", "C_fn", "D_fn", "model_init", "env_utils"]


