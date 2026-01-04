"""
Configuration management for ST-Dynamics research codebase.

This module centralizes all hyperparameters and experimental settings
for reproducible research at Nature Machine Intelligence standard.
"""

import dataclasses
from typing import Optional, Tuple, List


@dataclasses.dataclass
class Config:
    """Central configuration for ST-Dynamics experiments.
    
    All hyperparameters are explicitly defined here for reproducibility.
    Follows the principle: no magic numbers in code.
    """
    
    # Data preprocessing
    n_hvgs: int = 2000  # Number of highly variable genes
    normalize_counts: bool = True
    log_transform: bool = True
    scale_features: bool = True
    
    # Spatial neighborhood
    n_spatial_neighbors: int = 6  # For spatial smoothness
    spatial_method: str = "knn"  # "knn" or "radius" 
    spatial_radius: Optional[float] = None
    
    # Model architecture
    input_dim: int = 2000  # Gene expression dimension
    latent_dim: int = 64   # Latent embedding dimension
    hidden_dims: Tuple[int, ...] = (512, 256, 128)  # Encoder hidden layers
    dropout_rate: float = 0.1
    activation: str = "relu"
    
    # Loss function weights
    lambda_recon: float = 1.0      # Reconstruction loss weight
    lambda_temporal: float = 0.5   # Temporal consistency weight  
    lambda_spatial: float = 0.3    # Spatial smoothness weight
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    early_stopping_patience: int = 20
    early_stopping_delta: float = 1e-4
    
    # Temporal inference (post-hoc)
    latent_time_method: str = "diffusion_map"  # "diffusion_map" or "projection"
    n_diffusion_components: int = 10
    diffusion_alpha: float = 1.0
    
    # Evaluation
    n_bootstrap_samples: int = 1000
    test_split_ratio: float = 0.2
    random_seed: int = 42
    
    # Baselines to compare
    baseline_methods: List[str] = dataclasses.field(default_factory=lambda: [
        "diffusion_pseudotime",
        "optimal_transport", 
        "paste_alignment",
        "dest_ot"
    ])
    
    # Experimental settings
    device: str = "cuda"  # "cuda" or "cpu"
    n_jobs: int = -1      # Parallel processing
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.latent_dim >= self.input_dim:
            raise ValueError("Latent dimension must be smaller than input dimension")
        
        if self.lambda_recon + self.lambda_temporal + self.lambda_spatial == 0:
            raise ValueError("At least one loss component must be non-zero")
            
        if not 0 <= self.test_split_ratio <= 1:
            raise ValueError("Test split ratio must be between 0 and 1")