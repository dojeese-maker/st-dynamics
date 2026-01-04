"""
Post-hoc inference modules for ST-Dynamics.

This package provides:
- LatentTimeInference: Post-hoc latent time inference methods
- infer_latent_time_simple: Simple inference interface
"""

from .latent_time import LatentTimeInference, infer_latent_time_simple

__all__ = [
    "LatentTimeInference",
    "infer_latent_time_simple"
]