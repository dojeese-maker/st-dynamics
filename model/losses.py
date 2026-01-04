"""
Loss functions for spatial transcriptomics temporal dynamics analysis.

This module implements the core loss functions for ST-Dynamics:
1. Temporal consistency loss (between consecutive timepoints)
2. Spatial smoothness loss (based on spatial neighborhood)
3. Reconstruction loss (autoencoder objective)

Critical: No direct supervision from true time labels is allowed.
Time information is used ONLY for temporal consistency constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

from ..config import Config


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for autoencoder-style training.
    
    Ensures that latent representations preserve gene expression information
    without relying on temporal supervision.
    """
    
    def __init__(self, loss_type: str = "mse"):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == "mae":
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown reconstruction loss type: {loss_type}")
    
    def forward(self, x_recon: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Parameters:
        -----------
        x_recon : torch.Tensor, shape (batch_size, n_genes)
            Reconstructed gene expression
        x_true : torch.Tensor, shape (batch_size, n_genes)
            Original gene expression
            
        Returns:
        --------
        loss : torch.Tensor, scalar
            Reconstruction loss
        """
        loss_per_sample = self.criterion(x_recon, x_true).mean(dim=1)
        return loss_per_sample.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for consecutive timepoints.
    
    Enforces smooth transitions between latent representations
    of consecutive time points. This is NOT direct supervision
    but rather a consistency constraint.
    
    Key principle: We encourage similar cellular states at consecutive
    timepoints to have smooth latent transitions, without directly
    predicting time from latent representations.
    """
    
    def __init__(self, consistency_type: str = "contrastive", temperature: float = 0.1):
        super(TemporalConsistencyLoss, self).__init__()
        self.consistency_type = consistency_type
        self.temperature = temperature
    
    def contrastive_temporal_loss(
        self, 
        z_t: torch.Tensor, 
        z_t_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss between consecutive timepoints.
        
        Encourages representations from consecutive timepoints to be
        more similar to each other than to representations from
        non-consecutive timepoints.
        """
        
        batch_size_t = z_t.size(0)
        batch_size_t1 = z_t_plus_1.size(0)
        
        # Normalize embeddings
        z_t_norm = F.normalize(z_t, dim=1)
        z_t1_norm = F.normalize(z_t_plus_1, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(z_t_norm, z_t1_norm.t()) / self.temperature
        
        # Create positive pairs (assuming aligned indices represent similar spatial locations)
        # This is a simplification - in practice, spatial alignment would be more sophisticated
        min_size = min(batch_size_t, batch_size_t1)
        positive_indices = torch.arange(min_size, device=z_t.device)
        
        # Positive similarities (diagonal elements)
        positive_sim = similarity[positive_indices, positive_indices]
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity)
        sum_exp_sim = exp_sim.sum(dim=1)
        
        loss = -positive_sim + torch.log(sum_exp_sim[positive_indices])
        
        return loss.mean()
    
    def smooth_transition_loss(
        self, 
        z_t: torch.Tensor, 
        z_t_plus_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Smooth transition loss between consecutive timepoints.
        
        Encourages gradual changes in latent space between timepoints.
        """
        
        # Align representations (simplified - assumes same spatial ordering)
        min_size = min(z_t.size(0), z_t_plus_1.size(0))
        z_t_aligned = z_t[:min_size]
        z_t1_aligned = z_t_plus_1[:min_size]
        
        # Compute transition smoothness
        transition = z_t1_aligned - z_t_aligned
        smoothness_loss = torch.norm(transition, dim=1, p=2).mean()
        
        return smoothness_loss
    
    def cycle_consistency_loss(
        self,
        z_t: torch.Tensor,
        z_t_plus_1: torch.Tensor, 
        z_t_plus_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Cycle consistency for temporal progression.
        
        If we have three consecutive timepoints, the transition
        from t to t+2 should be similar to the sum of transitions
        t->t+1 and t+1->t+2.
        """
        
        min_size = min(z_t.size(0), z_t_plus_1.size(0), z_t_plus_2.size(0))
        
        # Align all representations
        z_t = z_t[:min_size]
        z_t1 = z_t_plus_1[:min_size]
        z_t2 = z_t_plus_2[:min_size]
        
        # Direct transition t -> t+2
        direct_transition = z_t2 - z_t
        
        # Indirect transition t -> t+1 -> t+2
        indirect_transition = (z_t1 - z_t) + (z_t2 - z_t1)
        
        # Cycle consistency loss
        cycle_loss = F.mse_loss(direct_transition, indirect_transition)
        
        return cycle_loss
    
    def forward(
        self, 
        temporal_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Parameters:
        -----------
        temporal_pairs : list of tuples
            Each tuple contains (z_t, z_t+1) embeddings for consecutive timepoints
            
        Returns:
        --------
        loss : torch.Tensor, scalar
            Temporal consistency loss
        """
        
        if len(temporal_pairs) == 0:
            return torch.tensor(0.0, device=temporal_pairs[0][0].device if temporal_pairs else None)
        
        total_loss = 0.0
        count = 0
        
        for z_t, z_t_plus_1 in temporal_pairs:
            if z_t.size(0) == 0 or z_t_plus_1.size(0) == 0:
                continue
                
            if self.consistency_type == "contrastive":
                loss = self.contrastive_temporal_loss(z_t, z_t_plus_1)
            elif self.consistency_type == "smooth":
                loss = self.smooth_transition_loss(z_t, z_t_plus_1)
            else:
                raise ValueError(f"Unknown consistency type: {self.consistency_type}")
            
            total_loss += loss
            count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0)


class SpatialSmoothnessLoss(nn.Module):
    """
    Spatial smoothness loss based on spatial neighborhood structure.
    
    Encourages nearby spots in physical space to have similar
    latent representations, reflecting the spatial organization
    of biological processes.
    
    Uses ONLY spatial coordinates, never biological labels or time information.
    """
    
    def __init__(self, smoothness_type: str = "laplacian"):
        super(SpatialSmoothnessLoss, self).__init__()
        self.smoothness_type = smoothness_type
    
    def build_spatial_graph(
        self, 
        coords: torch.Tensor, 
        k: int = 6,
        method: str = "knn"
    ) -> torch.Tensor:
        """
        Build spatial adjacency matrix from coordinates.
        
        Parameters:
        -----------
        coords : torch.Tensor, shape (n_spots, 2)
            Spatial coordinates
        k : int
            Number of nearest neighbors
        method : str
            Method for building graph ("knn" or "radius")
            
        Returns:
        --------
        adj_matrix : torch.Tensor, shape (n_spots, n_spots)
            Spatial adjacency matrix
        """
        
        n_spots = coords.size(0)
        
        if method == "knn":
            # Compute pairwise distances
            coords_np = coords.cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coords_np)
            distances, indices = nbrs.kneighbors(coords_np)
            
            # Build adjacency matrix (exclude self-connections)
            adj_matrix = torch.zeros((n_spots, n_spots), device=coords.device)
            for i in range(n_spots):
                neighbors = indices[i, 1:]  # Exclude self (first index)
                neighbor_distances = distances[i, 1:]
                
                # Weight by inverse distance (with small epsilon to avoid division by zero)
                weights = 1.0 / (neighbor_distances + 1e-8)
                
                for j, weight in zip(neighbors, weights):
                    adj_matrix[i, j] = weight
                    adj_matrix[j, i] = weight  # Make symmetric
        
        else:
            raise ValueError(f"Unknown graph building method: {method}")
        
        # Normalize adjacency matrix
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adj_matrix = adj_matrix / degree
        
        return adj_matrix
    
    def laplacian_smoothness_loss(
        self,
        z: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Laplacian smoothness loss.
        
        Encourages neighboring spots to have similar embeddings
        using graph Laplacian regularization.
        """
        
        # Compute degree matrix
        degree = adj_matrix.sum(dim=1)
        degree_matrix = torch.diag(degree)
        
        # Compute graph Laplacian
        laplacian = degree_matrix - adj_matrix
        
        # Compute smoothness loss: z^T L z
        smoothness = torch.trace(torch.mm(torch.mm(z.t(), laplacian), z))
        
        return smoothness / z.size(0)  # Normalize by number of spots
    
    def neighbor_consistency_loss(
        self,
        z: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Neighbor consistency loss.
        
        Direct penalty for differences between neighboring embeddings.
        """
        
        n_spots, latent_dim = z.shape
        total_loss = 0.0
        total_weight = 0.0
        
        for i in range(n_spots):
            neighbors = torch.nonzero(adj_matrix[i] > 0, as_tuple=True)[0]
            if len(neighbors) == 0:
                continue
            
            # Compute weighted difference with neighbors
            z_i = z[i].unsqueeze(0).expand(len(neighbors), -1)
            z_neighbors = z[neighbors]
            weights = adj_matrix[i, neighbors].unsqueeze(1)
            
            # Weighted L2 difference
            diff = (z_i - z_neighbors) ** 2
            weighted_diff = (diff * weights).sum()
            
            total_loss += weighted_diff
            total_weight += weights.sum()
        
        return total_loss / (total_weight + 1e-8)
    
    def forward(
        self,
        z: torch.Tensor,
        coords: torch.Tensor,
        precomputed_adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute spatial smoothness loss.
        
        Parameters:
        -----------
        z : torch.Tensor, shape (batch_size, latent_dim)
            Latent embeddings
        coords : torch.Tensor, shape (batch_size, 2)
            Spatial coordinates
        precomputed_adj : torch.Tensor, optional
            Precomputed adjacency matrix
            
        Returns:
        --------
        loss : torch.Tensor, scalar
            Spatial smoothness loss
        """
        
        if z.size(0) != coords.size(0):
            raise ValueError("Number of embeddings must match number of coordinates")
        
        # Build or use precomputed adjacency matrix
        if precomputed_adj is None:
            adj_matrix = self.build_spatial_graph(coords)
        else:
            adj_matrix = precomputed_adj
        
        # Compute smoothness loss
        if self.smoothness_type == "laplacian":
            return self.laplacian_smoothness_loss(z, adj_matrix)
        elif self.smoothness_type == "neighbor":
            return self.neighbor_consistency_loss(z, adj_matrix)
        else:
            raise ValueError(f"Unknown smoothness type: {self.smoothness_type}")


class VAEKLDivergenceLoss(nn.Module):
    """
    KL divergence loss for variational autoencoders.
    
    Encourages latent representations to follow a prior distribution
    while maintaining expressiveness for downstream tasks.
    """
    
    def __init__(self, beta: float = 1.0):
        super(VAEKLDivergenceLoss, self).__init__()
        self.beta = beta  # Beta-VAE weight for KL term
    
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Parameters:
        -----------
        mu : torch.Tensor, shape (batch_size, latent_dim)
            Mean of latent distribution
        logvar : torch.Tensor, shape (batch_size, latent_dim)
            Log-variance of latent distribution
            
        Returns:
        --------
        loss : torch.Tensor, scalar
            KL divergence loss
        """
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return self.beta * kl_loss.mean()


class STDynamicsLoss(nn.Module):
    """
    Combined loss function for ST-Dynamics model.
    
    Integrates reconstruction, temporal consistency, and spatial smoothness
    losses with configurable weights.
    """
    
    def __init__(self, config: Config):
        super(STDynamicsLoss, self).__init__()
        self.config = config
        
        # Initialize individual loss components
        self.reconstruction_loss = ReconstructionLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.spatial_loss = SpatialSmoothnessLoss()
        self.kl_loss = VAEKLDivergenceLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Parameters:
        -----------
        outputs : dict
            Model outputs containing 'z', 'x_recon', and optionally 'mu', 'logvar'
        inputs : dict
            Input data containing 'expression', 'coords', 'temporal_pairs'
            
        Returns:
        --------
        total_loss : torch.Tensor, scalar
            Combined weighted loss
        loss_components : dict
            Individual loss components for monitoring
        """
        
        loss_components = {}
        total_loss = 0.0
        
        # Reconstruction loss
        if 'x_recon' in outputs:
            recon_loss = self.reconstruction_loss(outputs['x_recon'], inputs['expression'])
            loss_components['reconstruction'] = recon_loss
            total_loss += self.config.lambda_recon * recon_loss
        
        # Temporal consistency loss
        if 'temporal_pairs' in inputs and len(inputs['temporal_pairs']) > 0:
            temporal_loss = self.temporal_loss(inputs['temporal_pairs'])
            loss_components['temporal'] = temporal_loss
            total_loss += self.config.lambda_temporal * temporal_loss
        
        # Spatial smoothness loss
        if 'coords' in inputs:
            spatial_loss = self.spatial_loss(outputs['z'], inputs['coords'])
            loss_components['spatial'] = spatial_loss
            total_loss += self.config.lambda_spatial * spatial_loss
        
        # KL divergence loss (for VAE)
        if 'mu' in outputs and 'logvar' in outputs:
            kl_loss = self.kl_loss(outputs['mu'], outputs['logvar'])
            loss_components['kl_divergence'] = kl_loss
            total_loss += kl_loss
        
        loss_components['total'] = total_loss
        
        return total_loss, loss_components