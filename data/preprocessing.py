"""
Advanced preprocessing utilities for spatial transcriptomics data.

This module provides standardized preprocessing functions that ensure
reproducible and robust data preparation for ST-Dynamics analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse, csr_matrix
import warnings

from ..config import Config


class SpatialTranscriptomicsPreprocessor:
    """
    Comprehensive preprocessing pipeline for spatial transcriptomics data.
    
    Handles batch effects, normalization, feature selection, and quality control
    following best practices for spatial omics data analysis.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.hvg_genes = None
        self.scaler = None
        self.normalization_factors = None
        
    def quality_control(
        self, 
        adata: ad.AnnData,
        min_genes: int = 200,
        min_cells: int = 3,
        max_genes: Optional[int] = None,
        mt_gene_pattern: str = "^MT-|^mt-"
    ) -> ad.AnnData:
        """
        Perform quality control filtering on spatial transcriptomics data.
        
        Parameters:
        -----------
        adata : anndata.AnnData
            Annotated data matrix
        min_genes : int
            Minimum number of genes expressed per spot
        min_cells : int  
            Minimum number of spots expressing each gene
        max_genes : int, optional
            Maximum number of genes per spot (for outlier removal)
        mt_gene_pattern : str
            Regex pattern to identify mitochondrial genes
            
        Returns:
        --------
        adata_filtered : anndata.AnnData
            Quality-controlled data
        """
        
        # Calculate QC metrics
        adata.var['mt'] = adata.var_names.str.match(mt_gene_pattern)
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        
        # Store original dimensions
        n_spots_orig, n_genes_orig = adata.shape
        
        # Filter spots (cells)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        if max_genes is not None:
            adata = adata[adata.obs.n_genes_by_counts < max_genes, :].copy()
            
        # Filter genes
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
        print(f"QC filtering: {n_spots_orig} → {adata.n_obs} spots, "
              f"{n_genes_orig} → {adata.n_vars} genes")
              
        return adata
    
    def normalize_counts(
        self, 
        adata: ad.AnnData,
        target_sum: float = 1e4,
        exclude_highly_expressed: bool = True
    ) -> ad.AnnData:
        """
        Normalize count data to account for sequencing depth differences.
        
        Parameters:
        -----------
        adata : anndata.AnnData
            Raw count data
        target_sum : float
            Target sum for normalization (typically 10,000)
        exclude_highly_expressed : bool
            Whether to exclude highly expressed genes from size factor calculation
            
        Returns:
        --------
        adata_norm : anndata.AnnData
            Normalized data
        """
        
        # Save raw counts
        adata.raw = adata
        
        # Calculate normalization factors
        if exclude_highly_expressed:
            sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=True)
        else:
            sc.pp.normalize_total(adata, target_sum=target_sum)
            
        # Store normalization factors
        self.normalization_factors = adata.obs['total_counts'].copy()
        
        return adata
    
    def select_highly_variable_genes(
        self,
        adata: ad.AnnData, 
        n_top_genes: Optional[int] = None,
        flavor: str = 'seurat_v3',
        batch_key: Optional[str] = None
    ) -> ad.AnnData:
        """
        Select highly variable genes for downstream analysis.
        
        Parameters:
        -----------
        adata : anndata.AnnData
            Normalized expression data
        n_top_genes : int, optional
            Number of top variable genes to select
        flavor : str
            Method for HVG selection ('seurat_v3', 'cell_ranger', 'seurat')
        batch_key : str, optional
            Key in adata.obs for batch correction during HVG selection
            
        Returns:
        --------
        adata_hvg : anndata.AnnData
            Data with HVG selection
        """
        
        n_top_genes = n_top_genes or self.config.n_hvgs
        
        # HVG selection
        if batch_key is not None:
            # Batch-aware HVG selection
            sc.pp.highly_variable_genes(
                adata, 
                n_top_genes=n_top_genes,
                flavor=flavor,
                batch_key=batch_key,
                subset=True
            )
        else:
            # Standard HVG selection
            sc.pp.highly_variable_genes(
                adata, 
                n_top_genes=n_top_genes,
                flavor=flavor,
                subset=True
            )
        
        # Store selected genes
        self.hvg_genes = adata.var_names.tolist()
        
        print(f"Selected {len(self.hvg_genes)} highly variable genes")
        
        return adata
    
    def log_transform(self, adata: ad.AnnData) -> ad.AnnData:
        """Apply log transformation to normalized counts."""
        sc.pp.log1p(adata)
        return adata
    
    def scale_features(
        self, 
        adata: ad.AnnData,
        zero_center: bool = True,
        max_value: Optional[float] = 10.0
    ) -> ad.AnnData:
        """
        Scale features to unit variance and optionally zero center.
        
        Parameters:
        -----------
        adata : anndata.AnnData
            Log-transformed expression data
        zero_center : bool
            Whether to center features to zero mean
        max_value : float, optional
            Clip scaled values to this maximum (prevents extreme outliers)
            
        Returns:
        --------
        adata_scaled : anndata.AnnData
            Scaled expression data
        """
        
        # Scale using scanpy
        sc.pp.scale(adata, zero_center=zero_center, max_value=max_value)
        
        # Store scaler for inverse transformation if needed
        self.scaler = StandardScaler()
        self.scaler.fit(adata.X if not issparse(adata.X) else adata.X.toarray())
        
        return adata
    
    def correct_batch_effects(
        self,
        adata: ad.AnnData,
        batch_key: str,
        method: str = "combat"
    ) -> ad.AnnData:
        """
        Correct for batch effects in spatial transcriptomics data.
        
        Parameters:
        -----------
        adata : anndata.AnnData
            Expression data with batch information
        batch_key : str
            Key in adata.obs containing batch labels
        method : str
            Batch correction method ('combat', 'harmony', 'scanorama')
            
        Returns:
        --------
        adata_corrected : anndata.AnnData
            Batch-corrected expression data
        """
        
        if method == "combat":
            # ComBat batch correction
            sc.pp.combat(adata, key=batch_key)
            
        elif method == "harmony":
            # Harmony integration (requires harmonypy)
            try:
                import harmonypy as hm
                # Apply Harmony on PC space
                sc.pp.pca(adata, n_comps=50)
                harmony_out = hm.run_harmony(
                    adata.obsm['X_pca'], 
                    adata.obs, 
                    batch_key,
                    random_state=self.config.random_seed
                )
                adata.obsm['X_pca'] = harmony_out.Z_corr.T
                
            except ImportError:
                warnings.warn("harmonypy not installed, falling back to ComBat")
                sc.pp.combat(adata, key=batch_key)
                
        elif method == "scanorama":
            # Scanorama integration
            try:
                import scanorama
                # Split by batch
                batches = []
                batch_names = []
                for batch in adata.obs[batch_key].unique():
                    batch_data = adata[adata.obs[batch_key] == batch].copy()
                    batches.append(batch_data.X)
                    batch_names.append(str(batch))
                
                # Integrate
                integrated, genes = scanorama.integrate_scanpy(
                    batches, 
                    batch_names,
                    return_dimred=True
                )
                
                # Reconstruct integrated data
                integrated_data = np.vstack(integrated)
                adata.X = integrated_data
                
            except ImportError:
                warnings.warn("scanorama not installed, falling back to ComBat")
                sc.pp.combat(adata, key=batch_key)
        
        else:
            raise ValueError(f"Unknown batch correction method: {method}")
            
        return adata
    
    def full_preprocessing_pipeline(
        self,
        X: np.ndarray,
        coords: np.ndarray, 
        time: np.ndarray,
        batch: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        sample_names: Optional[List[str]] = None
    ) -> Tuple[ad.AnnData, Dict[str, any]]:
        """
        Run complete preprocessing pipeline.
        
        Parameters:
        -----------
        X : np.ndarray
            Raw gene expression matrix
        coords : np.ndarray
            Spatial coordinates
        time : np.ndarray
            Time labels
        batch : np.ndarray, optional
            Batch labels
        gene_names : list, optional
            Gene names
        sample_names : list, optional
            Sample names
            
        Returns:
        --------
        adata_processed : anndata.AnnData
            Fully processed data
        processing_info : dict
            Information about preprocessing steps
        """
        
        # Create AnnData object
        adata = ad.AnnData(X)
        adata.obs_names = sample_names or [f"spot_{i}" for i in range(X.shape[0])]
        adata.var_names = gene_names or [f"gene_{i}" for i in range(X.shape[1])]
        
        # Add metadata
        adata.obs['x'] = coords[:, 0]
        adata.obs['y'] = coords[:, 1]
        adata.obs['time'] = time
        if batch is not None:
            adata.obs['batch'] = batch
        
        processing_info = {}
        
        # 1. Quality control
        print("Step 1: Quality control")
        adata = self.quality_control(adata)
        processing_info['qc'] = {'final_shape': adata.shape}
        
        # 2. Normalization
        print("Step 2: Normalization")
        adata = self.normalize_counts(adata)
        processing_info['normalization'] = {'target_sum': 1e4}
        
        # 3. Log transformation
        if self.config.log_transform:
            print("Step 3: Log transformation")
            adata = self.log_transform(adata)
            processing_info['log_transform'] = True
        
        # 4. Highly variable genes
        print("Step 4: Highly variable gene selection")
        batch_key = 'batch' if batch is not None else None
        adata = self.select_highly_variable_genes(adata, batch_key=batch_key)
        processing_info['hvg'] = {
            'n_genes': len(self.hvg_genes),
            'method': 'seurat_v3'
        }
        
        # 5. Batch correction (if needed)
        if batch is not None and len(np.unique(batch)) > 1:
            print("Step 5: Batch effect correction")
            adata = self.correct_batch_effects(adata, batch_key='batch')
            processing_info['batch_correction'] = {'method': 'combat'}
        
        # 6. Feature scaling
        if self.config.scale_features:
            print("Step 6: Feature scaling")
            adata = self.scale_features(adata)
            processing_info['scaling'] = {'zero_center': True, 'max_value': 10.0}
        
        print(f"Preprocessing complete. Final shape: {adata.shape}")
        
        return adata, processing_info