"""
Neural network encoder for spatial transcriptomics representation learning.

This module implements the encoder architecture that maps gene expression
to latent representations for temporal dynamics analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

from ..config import Config


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for gene expression to latent embedding.
    
    Maps high-dimensional gene expression vectors to lower-dimensional 
    latent representations that capture temporal dynamics information.
    """
    
    def __init__(self, config: Config):
        super(MLPEncoder, self).__init__()
        self.config = config
        
        # Build encoder layers
        layers = []
        input_dim = config.input_dim
        
        # Hidden layers
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(config.dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # Final projection to latent space
        layers.append(nn.Linear(input_dim, config.latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif self.config.activation == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")
    
    def _init_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.activation == "relu":
                    # He initialization for ReLU
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    # Xavier initialization for other activations
                    nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, n_genes)
            Gene expression input
            
        Returns:
        --------
        z : torch.Tensor, shape (batch_size, latent_dim)
            Latent representation
        """
        return self.encoder(x)


class VariationalEncoder(nn.Module):
    """
    Variational autoencoder (VAE) style encoder with uncertainty quantification.
    
    Encodes gene expression to a probabilistic latent representation,
    useful for capturing uncertainty in temporal dynamics.
    """
    
    def __init__(self, config: Config):
        super(VariationalEncoder, self).__init__()
        self.config = config
        
        # Build shared encoder layers
        layers = []
        input_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(config.dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*layers)
        
        # Separate heads for mean and log-variance
        self.mu_head = nn.Linear(input_dim, config.latent_dim)
        self.logvar_head = nn.Linear(input_dim, config.latent_dim)
        
        self._init_weights()
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Parameters:
        -----------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log-variance of latent distribution
            
        Returns:
        --------
        z : torch.Tensor
            Sampled latent representation
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through variational encoder.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, n_genes)
            Gene expression input
            
        Returns:
        --------
        z : torch.Tensor, shape (batch_size, latent_dim)
            Sampled latent representation
        mu : torch.Tensor, shape (batch_size, latent_dim) 
            Mean of latent distribution
        logvar : torch.Tensor, shape (batch_size, latent_dim)
            Log-variance of latent distribution
        """
        h = self.shared_encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar


class AttentionEncoder(nn.Module):
    """
    Attention-based encoder for capturing spatial context in gene expression.
    
    Uses self-attention mechanisms to model spatial relationships
    while encoding to latent representation.
    """
    
    def __init__(self, config: Config, n_heads: int = 8):
        super(AttentionEncoder, self).__init__()
        self.config = config
        self.n_heads = n_heads
        
        # Initial projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dims[0])
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_dims[0],
                num_heads=n_heads,
                dropout=config.dropout_rate,
                batch_first=True
            )
            for _ in range(len(config.hidden_dims) - 1)
        ])
        
        # Feed-forward layers after attention
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dims[0], config.hidden_dims[i+1]),
                self._get_activation(),
                nn.Dropout(config.dropout_rate),
                nn.LayerNorm(config.hidden_dims[i+1])
            )
            for i in range(len(config.hidden_dims) - 1)
        ])
        
        # Final projection to latent space
        final_dim = config.hidden_dims[-1] if config.hidden_dims else config.hidden_dims[0]
        self.output_projection = nn.Linear(final_dim, config.latent_dim)
        
        self._init_weights()
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through attention encoder.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, n_genes)
            Gene expression input
        attention_mask : torch.Tensor, optional
            Attention mask for spatial relationships
            
        Returns:
        --------
        z : torch.Tensor, shape (batch_size, latent_dim)
            Latent representation with spatial context
        """
        # Initial projection
        h = self.input_projection(x)  # (batch_size, hidden_dim)
        
        # Add sequence dimension for attention (treat each sample as sequence of length 1)
        h = h.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Apply attention layers
        for attention, ff in zip(self.attention_layers, self.ff_layers):
            # Self-attention
            attn_output, _ = attention(h, h, h, attn_mask=attention_mask)
            h = h + attn_output  # Residual connection
            
            # Feed-forward
            ff_output = ff(h.squeeze(1))  # Remove sequence dimension
            h = ff_output.unsqueeze(1)  # Add back sequence dimension
        
        # Remove sequence dimension and project to latent space
        h = h.squeeze(1)  # (batch_size, hidden_dim)
        z = self.output_projection(h)  # (batch_size, latent_dim)
        
        return z


class Decoder(nn.Module):
    """
    Decoder network for reconstructing gene expression from latent representation.
    
    Used for autoencoder-style training where reconstruction loss
    ensures the latent representation preserves expression information.
    """
    
    def __init__(self, config: Config):
        super(Decoder, self).__init__()
        self.config = config
        
        # Build decoder layers (reverse of encoder)
        layers = []
        input_dim = config.latent_dim
        
        # Reverse the hidden dimensions
        hidden_dims = list(reversed(config.hidden_dims))
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(config.dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # Final reconstruction layer
        layers.append(nn.Linear(input_dim, config.input_dim))
        
        self.decoder = nn.Sequential(*layers)
        self._init_weights()
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config."""
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Parameters:
        -----------
        z : torch.Tensor, shape (batch_size, latent_dim)
            Latent representation
            
        Returns:
        --------
        x_recon : torch.Tensor, shape (batch_size, n_genes)
            Reconstructed gene expression
        """
        return self.decoder(z)


def create_encoder(config: Config, encoder_type: str = "mlp") -> nn.Module:
    """
    Factory function to create encoder of specified type.
    
    Parameters:
    -----------
    config : Config
        Model configuration
    encoder_type : str
        Type of encoder ("mlp", "variational", "attention")
        
    Returns:
    --------
    encoder : nn.Module
        Encoder network
    """
    
    if encoder_type == "mlp":
        return MLPEncoder(config)
    elif encoder_type == "variational":
        return VariationalEncoder(config)
    elif encoder_type == "attention":
        return AttentionEncoder(config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")