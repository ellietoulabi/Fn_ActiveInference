"""
FullyCollective paradigm with semantic policies at the action level
(single-agent Monotonic cramped_room).

Mirrors the model files of `IndividuallyCollectiveWithSemanticPoliciesActionLevel`.
The structural model is identical — the FC paradigm differs only at the
runner level: a single IC brain plans joint semantic pairs
`(JOINT_PAIR_LABEL, a_self, a_other)` for both agents, and the partner
("puppet") executes the IC brain's prescription for it without running
inference of its own.

Uses this package's model_init, A, B, C, D, and env_utils.
"""

from . import model_init  # local (monotonic cramped_room)
from .A import A_fn
from .B import B_fn
from .C import C_fn
from .D import D_fn
from . import env_utils  # noqa: F401

__all__ = ["A_fn", "B_fn", "C_fn", "D_fn", "model_init", "env_utils"]
