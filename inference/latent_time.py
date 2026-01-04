"""
Post-hoc latent time inference for spatial transcriptomics dynamics.

This module implements methods for inferring continuous latent time
from learned latent representations WITHOUT any direct supervision
from true time labels during the inference process.

Critical principle: This is completely separate from model training.
The latent time is inferred ONLY from the spatial-temporal structure
of the learned embeddings, not from any time supervision.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, UMAP
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
import warnings

from ..config import Config


class LatentTimeInference:
    """
    Post-hoc inference of continuous latent time from learned embeddings.
    
    This class provides multiple methods for inferring temporal progression
    from latent representations without using true time labels as supervision.
    
    Methods available:
    1. Diffusion maps - spectral embedding of temporal manifold
    2. Trajectory projection - projection onto inferred temporal axis  
    3. Graph-based progression - temporal ordering via graph traversal
    4. Pseudotime inference - adaptation of single-cell pseudotime methods
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.is_fitted = False
        self.method = None
        self.latent_time_model = None
        self.embedding_scaler = None
        
    def fit_diffusion_map(
        self,
        embeddings: np.ndarray,
        n_components: Optional[int] = None,
        alpha: Optional[float] = None,
        epsilon: str = "auto"
    ) -> np.ndarray:
        """
        Fit diffusion map for temporal progression inference.
        
        Diffusion maps capture the intrinsic temporal manifold structure
        by computing the dominant eigenvectors of a diffusion operator
        constructed from the embedding similarity graph.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        n_components : int, optional
            Number of diffusion components (default from config)
        alpha : float, optional
            Diffusion alpha parameter (default from config)
        epsilon : str or float
            Bandwidth parameter for Gaussian kernel ("auto" or float)
            
        Returns:
        --------
        diffusion_coords : np.ndarray, shape (n_samples, n_components)
            Diffusion map coordinates
        """
        
        n_components = n_components or self.config.n_diffusion_components
        alpha = alpha or self.config.diffusion_alpha
        
        n_samples = embeddings.shape[0]
        
        # Standardize embeddings
        self.embedding_scaler = StandardScaler()
        embeddings_scaled = self.embedding_scaler.fit_transform(embeddings)
        
        # Compute pairwise distances
        distances = squareform(pdist(embeddings_scaled, metric='euclidean'))
        
        # Determine epsilon (bandwidth) automatically if needed
        if epsilon == "auto":
            # Use median of k-nearest neighbor distances
            k = min(10, n_samples // 10)
            knn_distances = []
            for i in range(n_samples):
                row_distances = distances[i]
                kth_distance = np.partition(row_distances, k)[k]
                knn_distances.append(kth_distance)
            epsilon = np.median(knn_distances)
        
        # Build affinity matrix using Gaussian kernel
        affinity_matrix = np.exp(-distances**2 / (2 * epsilon**2))
        
        # Normalize for diffusion operator
        # P = D^(-alpha) * A * D^(-alpha) where D is degree matrix
        row_sums = affinity_matrix.sum(axis=1)
        degree_matrix = np.diag(row_sums**(-alpha))
        
        # Construct diffusion matrix
        diffusion_matrix = degree_matrix @ affinity_matrix @ degree_matrix
        
        # Renormalize rows to sum to 1 (transition matrix)
        row_sums = diffusion_matrix.sum(axis=1)
        diffusion_matrix = diffusion_matrix / row_sums[:, np.newaxis]
        
        # Compute dominant eigenvectors
        # We want the largest eigenvalues (closest to 1)
        try:
            eigenvalues, eigenvectors = eigsh(
                diffusion_matrix, 
                k=n_components + 1,  # +1 because first eigenvector is constant
                which='LA',  # Largest algebraic eigenvalues
                return_eigenvectors=True
            )
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Skip the first eigenvector (constant) and take the rest
            diffusion_coords = eigenvectors[:, 1:n_components+1]
            
            # Scale by eigenvalues for proper diffusion coordinates
            eigenvalue_scaling = eigenvalues[1:n_components+1]
            diffusion_coords = diffusion_coords * eigenvalue_scaling[np.newaxis, :]
            
        except Exception as e:
            warnings.warn(f"Sparse eigenvalue computation failed: {e}. Using dense computation.")
            # Fallback to dense eigenvalue computation
            eigenvalues, eigenvectors = np.linalg.eigh(diffusion_matrix)
            
            # Sort by eigenvalue (descending) 
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Take diffusion coordinates
            diffusion_coords = eigenvectors[:, 1:n_components+1]
            eigenvalue_scaling = eigenvalues[1:n_components+1]
            diffusion_coords = diffusion_coords * eigenvalue_scaling[np.newaxis, :]
        
        # Store model parameters
        self.method = "diffusion_map"
        self.latent_time_model = {
            'diffusion_coords': diffusion_coords,
            'eigenvalues': eigenvalues[1:n_components+1],
            'epsilon': epsilon,
            'alpha': alpha,
            'affinity_matrix': affinity_matrix
        }
        self.is_fitted = True
        
        return diffusion_coords
    
    def fit_trajectory_projection(
        self,
        embeddings: np.ndarray,
        coords: np.ndarray,
        method: str = "pca"
    ) -> np.ndarray:
        """
        Fit trajectory projection method for temporal inference.
        
        Projects embeddings onto a one-dimensional trajectory that
        best captures temporal progression based on spatial-temporal structure.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        coords : np.ndarray, shape (n_samples, 2)
            Spatial coordinates
        method : str
            Projection method ("pca", "tsne", "umap")
            
        Returns:
        --------
        trajectory_time : np.ndarray, shape (n_samples,)
            Inferred trajectory time
        """
        
        # Standardize embeddings
        self.embedding_scaler = StandardScaler()
        embeddings_scaled = self.embedding_scaler.fit_transform(embeddings)
        
        if method == "pca":
            # Use first principal component as temporal axis
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            trajectory_time = pca.fit_transform(embeddings_scaled).flatten()
            
            projection_model = pca
            
        elif method == "tsne":
            # Use t-SNE with 1D output
            tsne = TSNE(n_components=1, random_state=self.config.random_seed)
            trajectory_time = tsne.fit_transform(embeddings_scaled).flatten()
            
            projection_model = tsne
            
        elif method == "umap":
            # Use UMAP with 1D output  
            try:
                import umap
                umap_model = umap.UMAP(
                    n_components=1, 
                    random_state=self.config.random_seed
                )
                trajectory_time = umap_model.fit_transform(embeddings_scaled).flatten()
                
                projection_model = umap_model
                
            except ImportError:
                warnings.warn("UMAP not available, falling back to PCA")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                trajectory_time = pca.fit_transform(embeddings_scaled).flatten()
                projection_model = pca
        
        else:
            raise ValueError(f"Unknown projection method: {method}")
        
        # Store model
        self.method = "trajectory_projection"
        self.latent_time_model = {
            'projection_model': projection_model,
            'method': method
        }
        self.is_fitted = True
        
        return trajectory_time
    
    def fit_graph_progression(
        self,
        embeddings: np.ndarray,
        coords: np.ndarray,
        start_point: Optional[int] = None,
        k_neighbors: int = 10
    ) -> np.ndarray:
        """
        Fit graph-based progression method.
        
        Builds a k-nearest neighbor graph in embedding space and
        computes shortest path distances from a starting point
        to infer temporal progression.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        coords : np.ndarray, shape (n_samples, 2)  
            Spatial coordinates
        start_point : int, optional
            Index of starting point (earliest timepoint)
        k_neighbors : int
            Number of nearest neighbors for graph construction
            
        Returns:
        --------
        progression_time : np.ndarray, shape (n_samples,)
            Graph-based progression time
        """
        
        # Standardize embeddings
        self.embedding_scaler = StandardScaler()
        embeddings_scaled = self.embedding_scaler.fit_transform(embeddings)
        
        n_samples = embeddings_scaled.shape[0]
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nbrs.fit(embeddings_scaled)
        distances, indices = nbrs.kneighbors(embeddings_scaled)
        
        # Create adjacency matrix
        from scipy.sparse import csr_matrix
        row_indices = np.repeat(np.arange(n_samples), k_neighbors)
        col_indices = indices.flatten()
        edge_weights = distances.flatten()
        
        adjacency_matrix = csr_matrix(
            (edge_weights, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        
        # Make symmetric by taking minimum distances
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
        
        # Determine start point if not provided
        if start_point is None:
            # Use point with minimum sum of distances to all other points
            # as proxy for "earliest" timepoint
            total_distances = np.array(adjacency_matrix.sum(axis=1)).flatten()
            start_point = np.argmin(total_distances)
        
        # Compute shortest path distances using Dijkstra's algorithm
        from scipy.sparse.csgraph import dijkstra
        
        progression_distances = dijkstra(
            adjacency_matrix,
            indices=[start_point],
            return_predecessors=False
        ).flatten()
        
        # Handle disconnected components
        if np.any(np.isinf(progression_distances)):
            warnings.warn("Graph has disconnected components. Using connected component only.")
            connected_mask = np.isfinite(progression_distances)
            progression_distances[~connected_mask] = np.max(progression_distances[connected_mask]) + 1
        
        # Store model
        self.method = "graph_progression" 
        self.latent_time_model = {
            'adjacency_matrix': adjacency_matrix,
            'start_point': start_point,
            'k_neighbors': k_neighbors
        }
        self.is_fitted = True
        
        return progression_distances
    
    def fit_pseudotime(
        self,
        embeddings: np.ndarray,
        method: str = "dpt"
    ) -> np.ndarray:
        """
        Fit pseudotime method adapted for spatial transcriptomics.
        
        Adapts single-cell pseudotime inference methods for spatial
        transcriptomics temporal dynamics.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        method : str
            Pseudotime method ("dpt" for diffusion pseudotime)
            
        Returns:
        --------
        pseudotime : np.ndarray, shape (n_samples,)
            Inferred pseudotime
        """
        
        if method == "dpt":
            # Use diffusion pseudotime
            try:
                import scanpy as sc
                import anndata as ad
                
                # Create AnnData object with embeddings
                adata = ad.AnnData(embeddings)
                
                # Compute neighborhood graph
                sc.pp.neighbors(adata, use_rep='X', random_state=self.config.random_seed)
                
                # Compute diffusion map
                sc.tl.diffmap(adata, n_comps=self.config.n_diffusion_components)
                
                # Infer diffusion pseudotime
                # Use the point with minimum total distance as root
                distances = adata.obsp['distances']
                total_distances = np.array(distances.sum(axis=1)).flatten()
                root_idx = np.argmin(total_distances)
                
                adata.uns['iroot'] = root_idx
                sc.tl.dpt(adata)
                
                pseudotime = adata.obs['dpt_pseudotime'].values
                
                # Store model components
                self.method = "pseudotime_dpt"
                self.latent_time_model = {
                    'adata': adata,
                    'root_idx': root_idx
                }
                
            except ImportError:
                warnings.warn("scanpy not available, falling back to diffusion map")
                pseudotime = self.fit_diffusion_map(embeddings)[:, 0]  # Use first component
        
        else:
            raise ValueError(f"Unknown pseudotime method: {method}")
        
        self.is_fitted = True
        return pseudotime
    
    def infer_latent_time(
        self,
        embeddings: np.ndarray,
        coords: Optional[np.ndarray] = None,
        method: Optional[str] = None,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Infer latent time from embeddings using specified method.
        
        This is the main interface for latent time inference.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        coords : np.ndarray, shape (n_samples, 2), optional
            Spatial coordinates (required for some methods)
        method : str, optional
            Inference method (uses config default if None)
        normalize : bool
            Whether to normalize latent time to [0, 1] range
        **kwargs
            Additional parameters for specific methods
            
        Returns:
        --------
        latent_time : np.ndarray, shape (n_samples,)
            Inferred continuous latent time
        """
        
        method = method or self.config.latent_time_method
        
        # Apply the specified method
        if method == "diffusion_map":
            diffusion_coords = self.fit_diffusion_map(embeddings, **kwargs)
            # Use first diffusion coordinate as latent time
            latent_time = diffusion_coords[:, 0]
            
        elif method == "trajectory_projection":
            if coords is None:
                raise ValueError("Spatial coordinates required for trajectory projection")
            latent_time = self.fit_trajectory_projection(embeddings, coords, **kwargs)
            
        elif method == "graph_progression":
            if coords is None:
                raise ValueError("Spatial coordinates required for graph progression")
            latent_time = self.fit_graph_progression(embeddings, coords, **kwargs)
            
        elif method == "pseudotime":
            latent_time = self.fit_pseudotime(embeddings, **kwargs)
            
        else:
            raise ValueError(f"Unknown latent time inference method: {method}")
        
        # Normalize to [0, 1] if requested
        if normalize:
            latent_time = (latent_time - latent_time.min()) / (latent_time.max() - latent_time.min())
        
        return latent_time
    
    def transform(self, new_embeddings: np.ndarray) -> np.ndarray:
        """
        Transform new embeddings to latent time using fitted model.
        
        Parameters:
        -----------
        new_embeddings : np.ndarray, shape (n_new_samples, latent_dim)
            New embeddings to transform
            
        Returns:
        --------
        latent_time : np.ndarray, shape (n_new_samples,)
            Latent time for new embeddings
        """
        
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transforming new data")
        
        # Standardize using fitted scaler
        if self.embedding_scaler is not None:
            new_embeddings_scaled = self.embedding_scaler.transform(new_embeddings)
        else:
            new_embeddings_scaled = new_embeddings
        
        if self.method == "trajectory_projection":
            # Transform using fitted projection model
            projection_model = self.latent_time_model['projection_model']
            latent_time = projection_model.transform(new_embeddings_scaled).flatten()
            
        elif self.method == "diffusion_map":
            # For diffusion map, we would need to extend to new points
            # This is complex and requires kernel extension
            warnings.warn("Transform not implemented for diffusion map method")
            raise NotImplementedError("Transform not available for diffusion map")
            
        elif self.method == "graph_progression":
            # For graph methods, would need to add new points to existing graph
            warnings.warn("Transform not implemented for graph progression method")
            raise NotImplementedError("Transform not available for graph progression")
            
        else:
            raise NotImplementedError(f"Transform not implemented for method: {self.method}")
        
        return latent_time
    
    def get_inference_summary(self) -> Dict[str, any]:
        """Get summary of the fitted latent time inference model."""
        
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        summary = {
            'method': self.method,
            'is_fitted': self.is_fitted
        }
        
        if self.method == "diffusion_map":
            summary.update({
                'n_components': len(self.latent_time_model['eigenvalues']),
                'eigenvalues': self.latent_time_model['eigenvalues'].tolist(),
                'epsilon': self.latent_time_model['epsilon'],
                'alpha': self.latent_time_model['alpha']
            })
        elif self.method == "trajectory_projection":
            summary['projection_method'] = self.latent_time_model['method']
        elif self.method == "graph_progression":
            summary.update({
                'start_point': self.latent_time_model['start_point'],
                'k_neighbors': self.latent_time_model['k_neighbors']
            })
        
        return summary


def infer_latent_time_simple(
    embeddings: np.ndarray,
    method: str = "diffusion_map",
    coords: Optional[np.ndarray] = None,
    config: Optional[Config] = None,
    **kwargs
) -> np.ndarray:
    """
    Simple interface for latent time inference.
    
    Parameters:
    -----------
    embeddings : np.ndarray, shape (n_samples, latent_dim)
        Learned latent embeddings
    method : str
        Inference method
    coords : np.ndarray, optional
        Spatial coordinates
    config : Config, optional
        Configuration object
    **kwargs
        Additional method parameters
        
    Returns:
    --------
    latent_time : np.ndarray, shape (n_samples,)
        Inferred latent time
    """
    
    inference = LatentTimeInference(config)
    return inference.infer_latent_time(embeddings, coords, method, **kwargs)