"""
Independent paradigm — single-agent Monotonic cramped_room.

Each agent uses its own generative model and treats the partner as part of
the environment (identity transition; observed and inferred, not co-planned).

Uses this package's model_init, A, B, C, D, and env_utils.
"""

from . import model_init  # local (monotonic cramped_room)
from .A import A_fn
from .B import B_fn
from .C import C_fn
from .D import D_fn
from . import env_utils  # noqa: F401

__all__ = ["A_fn", "B_fn", "C_fn", "D_fn", "model_init", "env_utils"]


