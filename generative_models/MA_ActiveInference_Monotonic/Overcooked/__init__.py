"""
Multi-Agent Active Inference generative models for the Overcooked-AI environment.

Layouts are organised by name, e.g.:

- `cramped_room/Independent`
- `cramped_room/FullyCollective`
- `cramped_room/IndividuallyCollective`

High-level code should import layout-specific models explicitly, e.g.:

    from generative_models.MA_ActiveInference.Overcooked.cramped_room.Independent import (
        A_fn, B_fn, C_fn, D_fn, model_init, env_utils
    )
"""

from . import cramped_room  # noqa: F401

__all__ = ["cramped_room"]

