"""
Neural network models and components for ST-Dynamics.

This package provides:
- STDynamicsModel: Main model with strict train/inference separation
- Encoder architectures: MLP, Variational, Attention encoders  
- Loss functions: Reconstruction, temporal consistency, spatial smoothness
- Decoder: Gene expression reconstruction
"""

from .model import STDynamicsModel
from .encoder import MLPEncoder, VariationalEncoder, AttentionEncoder, Decoder, create_encoder
from .losses import (
    STDynamicsLoss, 
    ReconstructionLoss, 
    TemporalConsistencyLoss, 
    SpatialSmoothnessLoss,
    VAEKLDivergenceLoss
)

__all__ = [
    "STDynamicsModel",
    "MLPEncoder", 
    "VariationalEncoder",
    "AttentionEncoder", 
    "Decoder",
    "create_encoder",
    "STDynamicsLoss",
    "ReconstructionLoss",
    "TemporalConsistencyLoss", 
    "SpatialSmoothnessLoss",
    "VAEKLDivergenceLoss"
]