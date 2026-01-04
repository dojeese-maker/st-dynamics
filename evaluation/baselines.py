"""
Baseline and SOTA methods for spatial transcriptomics temporal dynamics analysis.

This module implements unified interfaces for comparing ST-Dynamics against
established methods in the field. All methods return standardized outputs
for fair comparison.

Baseline methods:
1. Diffusion Pseudotime (DPT) - from scanpy
2. Optimal Transport (OT) based alignment 
3. PASTE - spatial transcriptomics alignment
4. DeST-OT - spatiotemporal optimal transport (simplified implementation)
5. Standard dimensionality reduction (PCA, UMAP) + temporal ordering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad

from ..config import Config
from ..data.dataset import SpatialTranscriptomicsDataset


class BaselineModel(ABC):
    """
    Abstract base class for baseline and SOTA methods.
    
    All baseline implementations must inherit from this class and
    implement the required methods for standardized comparison.
    """
    
    def __init__(self, name: str, config: Optional[Config] = None):
        self.name = name
        self.config = config or Config()
        self.is_fitted = False
        
    @abstractmethod
    def fit_transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        time: np.ndarray,
        batch: Optional[np.ndarray] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Fit the baseline method and transform data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Gene expression data
        coords : np.ndarray, shape (n_samples, 2)
            Spatial coordinates
        time : np.ndarray, shape (n_samples,)
            Time labels (for methods that can use them)
        batch : np.ndarray, optional
            Batch labels
            
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'embedding': latent embedding (n_samples, embed_dim) or None
            - 'inferred_time': continuous time (n_samples,) or None
            - 'method_specific': any method-specific outputs
        """
        pass
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return capabilities of the baseline method."""
        return {
            'produces_embedding': False,
            'produces_time': False,
            'uses_spatial_info': False,
            'uses_temporal_info': False,
            'handles_batches': False
        }


class DiffusionPseudotimeBaseline(BaselineModel):
    """
    Diffusion Pseudotime (DPT) baseline using scanpy implementation.
    
    This is the gold standard pseudotime method from single-cell analysis,
    adapted for spatial transcriptomics.
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__("Diffusion_Pseudotime", config)
        self.adata = None
        
    def fit_transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        time: np.ndarray,
        batch: Optional[np.ndarray] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Apply diffusion pseudotime analysis.
        
        Note: This method does NOT use true time labels for supervision.
        Time labels are only used post-hoc for root selection heuristics.
        """
        
        try:
            # Create AnnData object
            adata = ad.AnnData(X)
            adata.obs['x_coord'] = coords[:, 0]
            adata.obs['y_coord'] = coords[:, 1] 
            adata.obs['time_label'] = time.astype(str)
            
            if batch is not None:
                adata.obs['batch'] = batch.astype(str)
            
            # Standard scanpy preprocessing
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=self.config.n_hvgs)
            adata.raw = adata
            adata = adata[:, adata.var.highly_variable].copy()
            sc.pp.scale(adata, max_value=10)
            
            # Principal component analysis
            sc.pp.pca(adata, n_comps=50, random_state=self.config.random_seed)
            
            # Compute neighborhood graph
            sc.pp.neighbors(adata, n_neighbors=15, random_state=self.config.random_seed)
            
            # Diffusion map
            sc.tl.diffmap(adata, n_comps=self.config.n_diffusion_components)
            
            # Root selection: use earliest timepoint centroid as heuristic
            # This is NOT supervision but a reasonable initialization
            earliest_time = np.min(time)
            earliest_mask = (time == earliest_time)
            
            if np.sum(earliest_mask) > 0:
                # Find spot from earliest timepoint closest to PCA center
                earliest_pca = adata.obsm['X_pca'][earliest_mask]
                pca_center = np.mean(adata.obsm['X_pca'], axis=0)
                distances = np.linalg.norm(earliest_pca - pca_center, axis=1)
                root_in_earliest = np.argmin(distances)
                root_idx = np.where(earliest_mask)[0][root_in_earliest]
            else:
                # Fallback: use global centroid
                pca_center = np.mean(adata.obsm['X_pca'], axis=0)
                distances = np.linalg.norm(adata.obsm['X_pca'] - pca_center, axis=1)
                root_idx = np.argmin(distances)
            
            # Set root and compute DPT
            adata.uns['iroot'] = root_idx
            sc.tl.dpt(adata)
            
            # Extract results
            embedding = adata.obsm['X_diffmap']
            inferred_time = adata.obs['dpt_pseudotime'].values
            
            self.adata = adata
            self.is_fitted = True
            
            return {
                'embedding': embedding,
                'inferred_time': inferred_time,
                'method_specific': {
                    'X_pca': adata.obsm['X_pca'],
                    'root_idx': root_idx,
                    'dpt_groups': adata.obs.get('dpt_groups', None)
                }
            }
            
        except Exception as e:
            warnings.warn(f"DPT failed: {e}. Returning None results.")
            return {
                'embedding': None,
                'inferred_time': None,
                'method_specific': {'error': str(e)}
            }
    
    def get_capabilities(self) -> Dict[str, bool]:
        return {
            'produces_embedding': True,
            'produces_time': True,
            'uses_spatial_info': False,
            'uses_temporal_info': False,  # Only for root heuristic, not supervision
            'handles_batches': False
        }


class OptimalTransportBaseline(BaselineModel):
    """
    Optimal Transport (OT) based temporal alignment baseline.
    
    Uses Sinkhorn algorithm to align distributions across timepoints
    and infer temporal progression from transport maps.
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__("Optimal_Transport", config)
        self.transport_maps = None
        
    def sinkhorn_distance(
        self, 
        X1: np.ndarray, 
        X2: np.ndarray,
        reg: float = 0.1,
        max_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Sinkhorn distance and transport plan between two distributions.
        
        Parameters:
        -----------
        X1, X2 : np.ndarray
            Source and target distributions
        reg : float
            Regularization parameter
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        transport_plan : np.ndarray
            Optimal transport plan
        cost : float
            Transport cost
        """
        
        # Compute cost matrix (Euclidean distance)
        C = cdist(X1, X2, metric='euclidean')
        C = C / np.max(C)  # Normalize
        
        n1, n2 = C.shape
        
        # Initialize uniform distributions
        a = np.ones(n1) / n1
        b = np.ones(n2) / n2
        
        # Sinkhorn algorithm
        K = np.exp(-C / reg)
        u = np.ones(n1)
        v = np.ones(n2)
        
        for _ in range(max_iter):
            u_new = a / (K @ v)
            v_new = b / (K.T @ u_new)
            
            # Check convergence
            if np.allclose(u, u_new) and np.allclose(v, v_new):
                break
                
            u, v = u_new, v_new
        
        # Compute transport plan
        transport_plan = np.diag(u) @ K @ np.diag(v)
        cost = np.sum(transport_plan * C)
        
        return transport_plan, cost
    
    def fit_transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        time: np.ndarray,
        batch: Optional[np.ndarray] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Apply optimal transport based temporal alignment.
        """
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            unique_times = np.unique(time)
            n_times = len(unique_times)
            
            if n_times < 2:
                warnings.warn("Need at least 2 timepoints for OT analysis")
                return {
                    'embedding': None,
                    'inferred_time': None,
                    'method_specific': {'error': 'Insufficient timepoints'}
                }
            
            # Compute transport maps between consecutive timepoints
            transport_maps = {}
            transport_costs = []
            
            for i in range(n_times - 1):
                t1, t2 = unique_times[i], unique_times[i + 1]
                
                mask1 = (time == t1)
                mask2 = (time == t2)
                
                X1 = X_scaled[mask1]
                X2 = X_scaled[mask2]
                
                if len(X1) > 0 and len(X2) > 0:
                    transport_plan, cost = self.sinkhorn_distance(X1, X2)
                    transport_maps[f"{t1}_to_{t2}"] = transport_plan
                    transport_costs.append(cost)
            
            # Infer latent time from transport costs
            # Simple heuristic: cumulative transport cost from first timepoint
            inferred_time = np.zeros(len(time))
            cumulative_cost = 0
            
            for i, t in enumerate(unique_times):
                mask = (time == t)
                inferred_time[mask] = cumulative_cost
                
                if i < len(transport_costs):
                    cumulative_cost += transport_costs[i]
            
            # Normalize to [0, 1]
            if np.max(inferred_time) > np.min(inferred_time):
                inferred_time = (inferred_time - np.min(inferred_time)) / (np.max(inferred_time) - np.min(inferred_time))
            
            # No embedding produced by this method
            self.transport_maps = transport_maps
            self.is_fitted = True
            
            return {
                'embedding': None,
                'inferred_time': inferred_time,
                'method_specific': {
                    'transport_maps': transport_maps,
                    'transport_costs': transport_costs
                }
            }
            
        except Exception as e:
            warnings.warn(f"Optimal Transport failed: {e}")
            return {
                'embedding': None,
                'inferred_time': None,
                'method_specific': {'error': str(e)}
            }
    
    def get_capabilities(self) -> Dict[str, bool]:
        return {
            'produces_embedding': False,
            'produces_time': True,
            'uses_spatial_info': False,
            'uses_temporal_info': True,  # Uses timepoint labels for alignment
            'handles_batches': False
        }


class PASTEBaseline(BaselineModel):
    """
    PASTE (Pairwise Alignment of Spatial Transcriptomics Experiments) baseline.
    
    Simplified implementation of spatial alignment between timepoints.
    Focus on spatial registration and temporal progression inference.
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__("PASTE", config)
        self.alignments = None
        
    def spatial_alignment_score(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        expr1: np.ndarray,
        expr2: np.ndarray,
        alpha: float = 0.1
    ) -> np.ndarray:
        """
        Compute spatial alignment score between two timepoints.
        
        Combines spatial distance and expression similarity.
        """
        
        # Spatial cost matrix
        spatial_cost = cdist(coords1, coords2, metric='euclidean')
        spatial_cost = spatial_cost / np.max(spatial_cost)  # Normalize
        
        # Expression cost matrix
        expr_cost = cdist(expr1, expr2, metric='cosine')
        
        # Combined cost
        combined_cost = alpha * spatial_cost + (1 - alpha) * expr_cost
        
        return combined_cost
    
    def fit_transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        time: np.ndarray,
        batch: Optional[np.ndarray] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Apply PASTE-style spatial alignment.
        """
        
        try:
            # Standardize expression data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            unique_times = np.unique(time)
            n_times = len(unique_times)
            
            if n_times < 2:
                return {
                    'embedding': None,
                    'inferred_time': None,
                    'method_specific': {'error': 'Insufficient timepoints'}
                }
            
            # Perform pairwise alignments
            alignments = {}
            alignment_costs = []
            
            for i in range(n_times - 1):
                t1, t2 = unique_times[i], unique_times[i + 1]
                
                mask1 = (time == t1)
                mask2 = (time == t2)
                
                coords1, coords2 = coords[mask1], coords[mask2]
                expr1, expr2 = X_scaled[mask1], X_scaled[mask2]
                
                if len(coords1) > 0 and len(coords2) > 0:
                    # Compute alignment cost
                    cost_matrix = self.spatial_alignment_score(coords1, coords2, expr1, expr2)
                    
                    # Solve assignment problem
                    if cost_matrix.shape[0] <= cost_matrix.shape[1]:
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        alignment_cost = cost_matrix[row_ind, col_ind].sum()
                    else:
                        # Transpose if more spots in t1 than t2
                        row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
                        alignment_cost = cost_matrix.T[row_ind, col_ind].sum()
                    
                    alignments[f"{t1}_to_{t2}"] = (row_ind, col_ind)
                    alignment_costs.append(alignment_cost)
            
            # Infer temporal progression from alignment costs
            inferred_time = np.zeros(len(time))
            cumulative_cost = 0
            
            for i, t in enumerate(unique_times):
                mask = (time == t)
                inferred_time[mask] = cumulative_cost
                
                if i < len(alignment_costs):
                    cumulative_cost += alignment_costs[i]
            
            # Normalize
            if np.max(inferred_time) > np.min(inferred_time):
                inferred_time = (inferred_time - np.min(inferred_time)) / (np.max(inferred_time) - np.min(inferred_time))
            
            self.alignments = alignments
            self.is_fitted = True
            
            return {
                'embedding': None,
                'inferred_time': inferred_time,
                'method_specific': {
                    'alignments': alignments,
                    'alignment_costs': alignment_costs
                }
            }
            
        except Exception as e:
            warnings.warn(f"PASTE failed: {e}")
            return {
                'embedding': None,
                'inferred_time': None,
                'method_specific': {'error': str(e)}
            }
    
    def get_capabilities(self) -> Dict[str, bool]:
        return {
            'produces_embedding': False,
            'produces_time': True,
            'uses_spatial_info': True,
            'uses_temporal_info': True,
            'handles_batches': False
        }


class DestOTBaseline(BaselineModel):
    """
    DeST-OT inspired baseline: spatiotemporal optimal transport.
    
    Simplified implementation that combines spatial and temporal information
    for optimal transport based alignment and pseudotime inference.
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__("DeST_OT", config)
        self.spatiotemporal_maps = None
        
    def spatiotemporal_cost(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        expr1: np.ndarray,
        expr2: np.ndarray,
        spatial_weight: float = 0.5
    ) -> np.ndarray:
        """
        Compute spatiotemporal cost matrix.
        
        Combines spatial distance, expression distance, and temporal prior.
        """
        
        # Spatial cost
        spatial_cost = cdist(coords1, coords2, metric='euclidean')
        spatial_cost = spatial_cost / (np.max(spatial_cost) + 1e-8)
        
        # Expression cost
        expr_cost = cdist(expr1, expr2, metric='cosine')
        
        # Combined cost
        cost_matrix = spatial_weight * spatial_cost + (1 - spatial_weight) * expr_cost
        
        return cost_matrix
    
    def fit_transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        time: np.ndarray,
        batch: Optional[np.ndarray] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Apply DeST-OT style spatiotemporal optimal transport.
        """
        
        try:
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Normalize coordinates
            coords_scaled = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)
            
            unique_times = np.unique(time)
            n_times = len(unique_times)
            
            if n_times < 2:
                return {
                    'embedding': None,
                    'inferred_time': None,
                    'method_specific': {'error': 'Insufficient timepoints'}
                }
            
            # Compute spatiotemporal transport maps
            transport_maps = {}
            transport_costs = []
            
            for i in range(n_times - 1):
                t1, t2 = unique_times[i], unique_times[i + 1]
                
                mask1 = (time == t1)
                mask2 = (time == t2)
                
                coords1, coords2 = coords_scaled[mask1], coords_scaled[mask2]
                expr1, expr2 = X_scaled[mask1], X_scaled[mask2]
                
                if len(coords1) > 0 and len(coords2) > 0:
                    # Compute spatiotemporal cost
                    cost_matrix = self.spatiotemporal_cost(coords1, coords2, expr1, expr2)
                    
                    # Simple transport plan (can be improved with proper OT solver)
                    if cost_matrix.shape[0] == cost_matrix.shape[1]:
                        # Equal sizes: use Hungarian algorithm
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        transport_cost = cost_matrix[row_ind, col_ind].sum()
                    else:
                        # Unequal sizes: use minimum cost matching
                        min_dim = min(cost_matrix.shape)
                        if cost_matrix.shape[0] < cost_matrix.shape[1]:
                            row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        else:
                            row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
                            row_ind, col_ind = col_ind, row_ind
                        transport_cost = cost_matrix[row_ind[:min_dim], col_ind[:min_dim]].sum()
                    
                    transport_maps[f"{t1}_to_{t2}"] = cost_matrix
                    transport_costs.append(transport_cost)
            
            # Infer latent time from spatiotemporal progression
            inferred_time = np.zeros(len(time))
            
            # Use PCA embedding as latent representation
            pca = PCA(n_components=min(50, X_scaled.shape[1]))
            embedding = pca.fit_transform(X_scaled)
            
            # Project temporal progression onto first PC
            time_progression = {}
            for t in unique_times:
                mask = (time == t)
                time_progression[t] = np.mean(embedding[mask, 0])  # Use first PC
            
            # Assign latent time based on PC progression
            for t in unique_times:
                mask = (time == t)
                inferred_time[mask] = time_progression[t]
            
            # Normalize
            if np.max(inferred_time) > np.min(inferred_time):
                inferred_time = (inferred_time - np.min(inferred_time)) / (np.max(inferred_time) - np.min(inferred_time))
            
            self.spatiotemporal_maps = transport_maps
            self.is_fitted = True
            
            return {
                'embedding': embedding,
                'inferred_time': inferred_time,
                'method_specific': {
                    'transport_maps': transport_maps,
                    'transport_costs': transport_costs,
                    'pca_components': pca.components_
                }
            }
            
        except Exception as e:
            warnings.warn(f"DeST-OT failed: {e}")
            return {
                'embedding': None,
                'inferred_time': None,
                'method_specific': {'error': str(e)}
            }
    
    def get_capabilities(self) -> Dict[str, bool]:
        return {
            'produces_embedding': True,
            'produces_time': True,
            'uses_spatial_info': True,
            'uses_temporal_info': True,
            'handles_batches': False
        }


class DimensionalityReductionBaseline(BaselineModel):
    """
    Standard dimensionality reduction + temporal ordering baseline.
    
    Uses PCA/UMAP for embedding and simple temporal ordering for time inference.
    Serves as a simple baseline to demonstrate the value of sophisticated methods.
    """
    
    def __init__(self, method: str = "pca", config: Optional[Config] = None):
        super().__init__(f"DimReduction_{method.upper()}", config)
        self.method = method
        self.reducer = None
        
    def fit_transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        time: np.ndarray,
        batch: Optional[np.ndarray] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Apply dimensionality reduction and simple temporal ordering.
        """
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply dimensionality reduction
            if self.method == "pca":
                from sklearn.decomposition import PCA
                self.reducer = PCA(n_components=self.config.latent_dim, random_state=self.config.random_seed)
                embedding = self.reducer.fit_transform(X_scaled)
                
            elif self.method == "umap":
                try:
                    import umap
                    self.reducer = umap.UMAP(
                        n_components=self.config.latent_dim,
                        random_state=self.config.random_seed
                    )
                    embedding = self.reducer.fit_transform(X_scaled)
                except ImportError:
                    warnings.warn("UMAP not available, falling back to PCA")
                    from sklearn.decomposition import PCA
                    self.reducer = PCA(n_components=self.config.latent_dim, random_state=self.config.random_seed)
                    embedding = self.reducer.fit_transform(X_scaled)
                    
            elif self.method == "tsne":
                from sklearn.manifold import TSNE
                self.reducer = TSNE(
                    n_components=min(self.config.latent_dim, 3),  # t-SNE limited to 3D
                    random_state=self.config.random_seed
                )
                embedding = self.reducer.fit_transform(X_scaled)
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Simple temporal ordering: use first principal component
            inferred_time = embedding[:, 0]
            
            # Normalize to [0, 1]
            if np.max(inferred_time) > np.min(inferred_time):
                inferred_time = (inferred_time - np.min(inferred_time)) / (np.max(inferred_time) - np.min(inferred_time))
            
            self.is_fitted = True
            
            return {
                'embedding': embedding,
                'inferred_time': inferred_time,
                'method_specific': {
                    'explained_variance_ratio': getattr(self.reducer, 'explained_variance_ratio_', None)
                }
            }
            
        except Exception as e:
            warnings.warn(f"Dimensionality reduction failed: {e}")
            return {
                'embedding': None,
                'inferred_time': None,
                'method_specific': {'error': str(e)}
            }
    
    def get_capabilities(self) -> Dict[str, bool]:
        return {
            'produces_embedding': True,
            'produces_time': True,
            'uses_spatial_info': False,
            'uses_temporal_info': False,
            'handles_batches': False
        }


class BaselineComparison:
    """
    Comprehensive baseline comparison framework.
    
    Manages multiple baseline methods and provides unified evaluation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.baselines = {}
        
        # Initialize default baselines
        self._initialize_baselines()
    
    def _initialize_baselines(self):
        """Initialize all available baseline methods."""
        
        self.baselines = {
            'dpt': DiffusionPseudotimeBaseline(self.config),
            'optimal_transport': OptimalTransportBaseline(self.config),
            'paste': PASTEBaseline(self.config),
            'dest_ot': DestOTBaseline(self.config),
            'pca': DimensionalityReductionBaseline('pca', self.config),
            'umap': DimensionalityReductionBaseline('umap', self.config)
        }
    
    def add_baseline(self, name: str, baseline: BaselineModel):
        """Add a custom baseline method."""
        self.baselines[name] = baseline
    
    def run_comparison(
        self,
        dataset: SpatialTranscriptomicsDataset,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run comparison across specified baseline methods.
        
        Parameters:
        -----------
        dataset : SpatialTranscriptomicsDataset
            Dataset to analyze
        methods : list, optional
            List of method names to compare (all if None)
            
        Returns:
        --------
        results : dict
            Results from all baseline methods
        """
        
        if methods is None:
            methods = list(self.baselines.keys())
        
        # Extract data from dataset
        X = dataset.X_processed
        coords = dataset.coords
        time = dataset.time
        batch = dataset.batch
        
        results = {}
        
        for method_name in methods:
            if method_name not in self.baselines:
                warnings.warn(f"Unknown baseline method: {method_name}")
                continue
            
            print(f"Running {method_name}...")
            
            try:
                baseline = self.baselines[method_name]
                result = baseline.fit_transform(X, coords, time, batch)
                
                # Add method metadata
                result['method_name'] = method_name
                result['capabilities'] = baseline.get_capabilities()
                result['success'] = True
                
                results[method_name] = result
                
            except Exception as e:
                print(f"  Failed: {e}")
                results[method_name] = {
                    'embedding': None,
                    'inferred_time': None,
                    'method_name': method_name,
                    'capabilities': self.baselines[method_name].get_capabilities(),
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def get_capability_summary(self) -> pd.DataFrame:
        """Get summary of baseline method capabilities."""
        
        capability_data = []
        
        for name, baseline in self.baselines.items():
            capabilities = baseline.get_capabilities()
            capabilities['method'] = name
            capability_data.append(capabilities)
        
        return pd.DataFrame(capability_data).set_index('method')
    
    def filter_by_capability(
        self,
        produces_embedding: Optional[bool] = None,
        produces_time: Optional[bool] = None,
        uses_spatial_info: Optional[bool] = None,
        uses_temporal_info: Optional[bool] = None
    ) -> List[str]:
        """
        Filter baseline methods by capabilities.
        
        Returns list of method names matching the criteria.
        """
        
        filtered_methods = []
        
        for name, baseline in self.baselines.items():
            capabilities = baseline.get_capabilities()
            
            # Check each filter criterion
            if produces_embedding is not None and capabilities['produces_embedding'] != produces_embedding:
                continue
            if produces_time is not None and capabilities['produces_time'] != produces_time:
                continue
            if uses_spatial_info is not None and capabilities['uses_spatial_info'] != uses_spatial_info:
                continue
            if uses_temporal_info is not None and capabilities['uses_temporal_info'] != uses_temporal_info:
                continue
            
            filtered_methods.append(name)
        
        return filtered_methods


def compare_with_baselines(
    st_dynamics_results: Dict[str, np.ndarray],
    dataset: SpatialTranscriptomicsDataset,
    config: Optional[Config] = None,
    baseline_methods: Optional[List[str]] = None
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Quick function to compare ST-Dynamics results with baseline methods.
    
    Parameters:
    -----------
    st_dynamics_results : dict
        ST-Dynamics results containing 'embedding' and 'inferred_time'
    dataset : SpatialTranscriptomicsDataset
        Dataset used for analysis
    config : Config, optional
        Configuration object
    baseline_methods : list, optional
        Specific baseline methods to compare
        
    Returns:
    --------
    all_results : dict
        Results from all methods including ST-Dynamics
    capabilities_summary : pd.DataFrame
        Summary of method capabilities
    """
    
    comparison = BaselineComparison(config)
    
    # Run baseline comparison
    baseline_results = comparison.run_comparison(dataset, baseline_methods)
    
    # Add ST-Dynamics results
    all_results = baseline_results.copy()
    all_results['ST_Dynamics'] = {
        'embedding': st_dynamics_results['embedding'],
        'inferred_time': st_dynamics_results['inferred_time'],
        'method_name': 'ST_Dynamics',
        'capabilities': {
            'produces_embedding': True,
            'produces_time': True,
            'uses_spatial_info': True,
            'uses_temporal_info': False,  # No direct supervision
            'handles_batches': True
        },
        'success': True
    }
    
    # Get capabilities summary
    capabilities_summary = comparison.get_capability_summary()
    
    # Add ST-Dynamics to summary
    st_dynamics_capabilities = pd.DataFrame([{
        'produces_embedding': True,
        'produces_time': True,
        'uses_spatial_info': True,
        'uses_temporal_info': False,
        'handles_batches': True
    }], index=['ST_Dynamics'])
    
    capabilities_summary = pd.concat([capabilities_summary, st_dynamics_capabilities])
    
    return all_results, capabilities_summary