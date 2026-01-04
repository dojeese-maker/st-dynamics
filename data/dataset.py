"""
Spatial transcriptomics dataset interface for ST-Dynamics.

This module defines the standard data interface for spatial transcriptomics
with temporal information, ensuring reproducible data handling.

Key principles:
- Explicit input/output interfaces
- No implicit time supervision during training
- Support for batch effects and multiple patients
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, List, Union
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, issparse

from ..config import Config


class SpatialTranscriptomicsDataset(Dataset):
    """
    Standard dataset interface for spatial transcriptomics with temporal dynamics.
    
    This class handles:
    - Gene expression data (X): shape (N, G)  
    - Spatial coordinates (coords): shape (N, 2)
    - Discrete time labels (time): shape (N,) - used only for consistency, NOT supervision
    - Optional batch information (batch): shape (N,)
    
    Critical: Time labels are used ONLY for temporal consistency constraints,
    never as direct supervision for latent time inference.
    """
    
    def __init__(
        self,
        X: Union[np.ndarray, csr_matrix, pd.DataFrame],
        coords: Union[np.ndarray, pd.DataFrame],
        time: Union[np.ndarray, pd.Series],
        batch: Optional[Union[np.ndarray, pd.Series]] = None,
        gene_names: Optional[List[str]] = None,
        sample_names: Optional[List[str]] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize spatial transcriptomics dataset.
        
        Parameters:
        -----------
        X : array-like, shape (N, G)
            Gene expression matrix. N = number of spots, G = number of genes
        coords : array-like, shape (N, 2)
            Spatial coordinates (x, y) for each spot
        time : array-like, shape (N,)
            Discrete time labels (e.g., 0, 1, 2, ... for different timepoints)
            Used ONLY for temporal consistency, NOT as supervision signal
        batch : array-like, shape (N,), optional
            Batch identifiers (e.g., patient ID, slide ID)
        gene_names : list of str, optional
            Gene names corresponding to columns in X
        sample_names : list of str, optional
            Sample/spot identifiers
        config : Config, optional
            Configuration object with preprocessing parameters
        """
        
        self.config = config or Config()
        
        # Convert inputs to numpy arrays
        self.X = self._ensure_array(X)
        self.coords = self._ensure_array(coords)
        self.time = self._ensure_array(time).astype(int)
        self.batch = self._ensure_array(batch) if batch is not None else None
        
        # Validate dimensions
        self._validate_inputs()
        
        # Store metadata
        self.gene_names = gene_names or [f"Gene_{i}" for i in range(self.X.shape[1])]
        self.sample_names = sample_names or [f"Spot_{i}" for i in range(self.X.shape[0])]
        
        # Initialize processed data
        self.X_processed = None
        self.spatial_graph = None
        self.hvg_mask = None
        
        # Process data
        self._preprocess_data()
        self._build_spatial_graph()
        
    def _ensure_array(self, data) -> np.ndarray:
        """Convert various data formats to numpy array."""
        if data is None:
            return None
        if issparse(data):
            return data.toarray()
        if isinstance(data, pd.DataFrame):
            return data.values
        if isinstance(data, pd.Series):
            return data.values
        return np.asarray(data)
    
    def _validate_inputs(self):
        """Validate input dimensions and consistency."""
        n_spots = self.X.shape[0]
        
        if self.coords.shape != (n_spots, 2):
            raise ValueError(f"Coordinates shape {self.coords.shape} doesn't match spots {n_spots}")
        
        if len(self.time) != n_spots:
            raise ValueError(f"Time labels length {len(self.time)} doesn't match spots {n_spots}")
        
        if self.batch is not None and len(self.batch) != n_spots:
            raise ValueError(f"Batch labels length {len(self.batch)} doesn't match spots {n_spots}")
            
        # Check for valid time points (should be sequential integers)
        unique_times = np.unique(self.time)
        if not np.array_equal(unique_times, np.arange(len(unique_times))):
            print(f"Warning: Time points {unique_times} are not sequential integers starting from 0")
    
    def _preprocess_data(self):
        """
        Preprocess gene expression data following standard scRNA-seq pipeline.
        
        Steps:
        1. Highly variable gene selection
        2. Normalization (if specified)
        3. Log transformation (if specified)  
        4. Feature scaling (if specified)
        """
        
        # Start with original data
        X_proc = self.X.copy()
        
        # Highly variable gene selection
        if self.config.n_hvgs < X_proc.shape[1]:
            self.hvg_mask = self._select_hvgs(X_proc, self.config.n_hvgs)
            X_proc = X_proc[:, self.hvg_mask]
            print(f"Selected {self.config.n_hvgs} highly variable genes")
        else:
            self.hvg_mask = np.ones(X_proc.shape[1], dtype=bool)
        
        # Normalization
        if self.config.normalize_counts:
            # Normalize to counts per 10,000 (standard)
            X_proc = X_proc / np.sum(X_proc, axis=1, keepdims=True) * 1e4
        
        # Log transformation
        if self.config.log_transform:
            X_proc = np.log1p(X_proc)
        
        # Feature scaling
        if self.config.scale_features:
            X_proc = (X_proc - np.mean(X_proc, axis=0)) / (np.std(X_proc, axis=0) + 1e-8)
        
        self.X_processed = X_proc.astype(np.float32)
        
    def _select_hvgs(self, X: np.ndarray, n_hvgs: int) -> np.ndarray:
        """Select highly variable genes using scanpy's method."""
        # Create temporary AnnData object
        import anndata as ad
        adata = ad.AnnData(X)
        
        # Calculate highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, flavor='seurat_v3')
        
        return adata.var['highly_variable'].values
    
    def _build_spatial_graph(self):
        """Build spatial neighborhood graph for spatial smoothness constraints."""
        
        if self.config.spatial_method == "knn":
            nbrs = NearestNeighbors(
                n_neighbors=self.config.n_spatial_neighbors + 1,  # +1 to exclude self
                metric='euclidean'
            ).fit(self.coords)
            
            distances, indices = nbrs.kneighbors(self.coords)
            
            # Remove self-connections (first column)
            indices = indices[:, 1:]
            distances = distances[:, 1:]
            
        elif self.config.spatial_method == "radius":
            if self.config.spatial_radius is None:
                raise ValueError("spatial_radius must be specified for radius method")
                
            nbrs = NearestNeighbors(
                radius=self.config.spatial_radius,
                metric='euclidean'
            ).fit(self.coords)
            
            distances, indices = nbrs.radius_neighbors(self.coords)
            
        else:
            raise ValueError(f"Unknown spatial method: {self.config.spatial_method}")
        
        self.spatial_graph = {
            'indices': indices,
            'distances': distances
        }
        
    def get_temporal_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get pairs of spot indices from consecutive time points.
        
        This is used ONLY for temporal consistency constraints,
        NOT for supervised learning of latent time.
        
        Returns:
        --------
        pairs : list of tuples
            Each tuple contains (indices_t, indices_t+1) for consecutive timepoints
        """
        pairs = []
        unique_times = np.unique(self.time)
        
        for t in range(len(unique_times) - 1):
            indices_t = np.where(self.time == unique_times[t])[0]
            indices_t_plus_1 = np.where(self.time == unique_times[t + 1])[0]
            pairs.append((indices_t, indices_t_plus_1))
            
        return pairs
    
    def get_spatial_neighbors(self, spot_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get spatial neighbors for a given spot."""
        neighbors = self.spatial_graph['indices'][spot_idx]
        distances = self.spatial_graph['distances'][spot_idx]
        return neighbors, distances
    
    def __len__(self) -> int:
        return self.X_processed.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample for PyTorch DataLoader."""
        sample = {
            'expression': torch.FloatTensor(self.X_processed[idx]),
            'coords': torch.FloatTensor(self.coords[idx]),
            'time': torch.LongTensor([self.time[idx]]),
            'spot_idx': torch.LongTensor([idx])
        }
        
        if self.batch is not None:
            sample['batch'] = torch.LongTensor([self.batch[idx]])
        
        return sample
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary statistics of the dataset."""
        return {
            'n_spots': self.X_processed.shape[0],
            'n_genes': self.X_processed.shape[1], 
            'n_timepoints': len(np.unique(self.time)),
            'timepoints': np.unique(self.time).tolist(),
            'n_batches': len(np.unique(self.batch)) if self.batch is not None else 1,
            'spatial_range_x': (self.coords[:, 0].min(), self.coords[:, 0].max()),
            'spatial_range_y': (self.coords[:, 1].min(), self.coords[:, 1].max()),
            'expression_range': (self.X_processed.min(), self.X_processed.max())
        }