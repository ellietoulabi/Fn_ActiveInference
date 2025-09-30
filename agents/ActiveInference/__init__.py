"""
Functional Active Inference Agent Package

This package provides a functional implementation of Active Inference that works
with functional A, B, C, D components instead of matrices. The agent can be used
with the functional generative models defined in the generative_models directory.
"""

from .agent import FunctionalAgent

__all__ = ['FunctionalAgent']
