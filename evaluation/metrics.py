"""
Evaluation metrics for spatial transcriptomics temporal dynamics analysis.

This module implements comprehensive evaluation metrics for assessing
the quality of latent time inference and spatial-temporal modeling,
following best practices for computational biology research.

Metrics include:
1. Monotonicity measures - temporal progression quality
2. Spatial smoothness measures - spatial coherence
3. Temporal consistency measures - cross-timepoint alignment  
4. Reconstruction quality - expression preservation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import warnings


class TemporalMonotonicityMetrics:
    """
    Metrics for evaluating temporal monotonicity of latent time inference.
    
    These metrics assess how well the inferred latent time captures
    the true temporal progression without using true time as supervision.
    """
    
    @staticmethod
    def kendall_tau_correlation(
        inferred_time: np.ndarray,
        true_time: np.ndarray
    ) -> float:
        """
        Compute Kendall's tau correlation between inferred and true time.
        
        Measures rank correlation, which is more robust to nonlinear
        monotonic relationships than Pearson correlation.
        
        Parameters:
        -----------
        inferred_time : np.ndarray
            Inferred continuous latent time
        true_time : np.ndarray
            True discrete time labels
            
        Returns:
        --------
        tau : float
            Kendall's tau correlation coefficient (-1 to 1)
        """
        
        tau, p_value = stats.kendalltau(inferred_time, true_time)
        return tau
    
    @staticmethod
    def spearman_correlation(
        inferred_time: np.ndarray,
        true_time: np.ndarray
    ) -> float:
        """
        Compute Spearman rank correlation between inferred and true time.
        
        Parameters:
        -----------
        inferred_time : np.ndarray
            Inferred continuous latent time
        true_time : np.ndarray  
            True discrete time labels
            
        Returns:
        --------
        rho : float
            Spearman correlation coefficient (-1 to 1)
        """
        
        rho, p_value = stats.spearmanr(inferred_time, true_time)
        return rho
    
    @staticmethod
    def temporal_ordering_accuracy(
        inferred_time: np.ndarray,
        true_time: np.ndarray,
        n_pairs: int = 1000
    ) -> float:
        """
        Compute temporal ordering accuracy for random pairs.
        
        Samples random pairs of spots and checks if their relative
        ordering in inferred time matches their true time ordering.
        
        Parameters:
        -----------
        inferred_time : np.ndarray
            Inferred continuous latent time
        true_time : np.ndarray
            True discrete time labels
        n_pairs : int
            Number of random pairs to sample
            
        Returns:
        --------
        accuracy : float
            Fraction of correctly ordered pairs (0 to 1)
        """
        
        n_samples = len(inferred_time)
        
        if n_pairs > n_samples * (n_samples - 1) // 2:
            n_pairs = n_samples * (n_samples - 1) // 2
        
        # Sample random pairs
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(n_samples, size=(n_pairs, 2), replace=True)
        
        correct_orders = 0
        valid_pairs = 0
        
        for i, j in indices:
            if i == j:
                continue
                
            true_order = np.sign(true_time[j] - true_time[i])
            inferred_order = np.sign(inferred_time[j] - inferred_time[i])
            
            # Skip pairs with same true time (undefined order)
            if true_order == 0:
                continue
                
            valid_pairs += 1
            if true_order == inferred_order:
                correct_orders += 1
        
        return correct_orders / valid_pairs if valid_pairs > 0 else 0.0
    
    @staticmethod
    def temporal_coherence_score(
        inferred_time: np.ndarray,
        true_time: np.ndarray
    ) -> float:
        """
        Compute temporal coherence score.
        
        Measures how well spots from the same timepoint are grouped
        together in the inferred latent time space.
        
        Parameters:
        -----------
        inferred_time : np.ndarray
            Inferred continuous latent time
        true_time : np.ndarray
            True discrete time labels
            
        Returns:
        --------
        coherence : float
            Temporal coherence score (higher is better)
        """
        
        # Compute within-timepoint variance vs between-timepoint variance
        unique_times = np.unique(true_time)
        
        # Overall variance
        total_variance = np.var(inferred_time)
        
        # Within-timepoint variance
        within_variance = 0.0
        total_count = 0
        
        for t in unique_times:
            mask = (true_time == t)
            if np.sum(mask) > 1:  # Need at least 2 samples
                within_variance += np.var(inferred_time[mask]) * np.sum(mask)
                total_count += np.sum(mask)
        
        within_variance /= total_count if total_count > 0 else 1
        
        # Coherence score (higher when within-variance is low relative to total)
        coherence = 1.0 - (within_variance / total_variance) if total_variance > 0 else 1.0
        
        return max(0.0, coherence)  # Ensure non-negative


class SpatialSmoothnessMetrics:
    """
    Metrics for evaluating spatial smoothness of learned representations.
    
    These metrics assess how well the learned embeddings respect
    the spatial organization of the tissue.
    """
    
    @staticmethod
    def spatial_autocorrelation(
        embeddings: np.ndarray,
        coords: np.ndarray,
        k_neighbors: int = 6
    ) -> float:
        """
        Compute spatial autocorrelation (Moran's I) for embeddings.
        
        Measures the degree of spatial clustering in the embedding space.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        coords : np.ndarray, shape (n_samples, 2)
            Spatial coordinates
        k_neighbors : int
            Number of spatial neighbors
            
        Returns:
        --------
        morans_i : float
            Moran's I statistic (higher indicates more spatial clustering)
        """
        
        n_samples, latent_dim = embeddings.shape
        
        # Build spatial weights matrix
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Create spatial weights matrix (inverse distance weighting)
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            neighbors = indices[i, 1:]  # Exclude self
            neighbor_distances = distances[i, 1:]
            weights = 1.0 / (neighbor_distances + 1e-8)
            W[i, neighbors] = weights
        
        # Normalize weights
        row_sums = W.sum(axis=1)
        W = W / row_sums[:, np.newaxis]
        W[np.isnan(W)] = 0  # Handle division by zero
        
        # Compute Moran's I for each embedding dimension
        morans_i_values = []
        
        for dim in range(latent_dim):
            x = embeddings[:, dim]
            x_mean = np.mean(x)
            
            # Compute Moran's I
            numerator = 0.0
            denominator = 0.0
            
            for i in range(n_samples):
                for j in range(n_samples):
                    if W[i, j] > 0:
                        numerator += W[i, j] * (x[i] - x_mean) * (x[j] - x_mean)
                
                denominator += (x[i] - x_mean) ** 2
            
            if denominator > 0:
                morans_i = (n_samples / np.sum(W)) * (numerator / denominator)
                morans_i_values.append(morans_i)
        
        # Return average Moran's I across dimensions
        return np.mean(morans_i_values) if morans_i_values else 0.0
    
    @staticmethod
    def spatial_neighbor_preservation(
        embeddings: np.ndarray,
        coords: np.ndarray,
        k_neighbors: int = 10
    ) -> float:
        """
        Compute spatial neighbor preservation score.
        
        Measures how well spatial neighbors in physical space
        remain neighbors in the learned embedding space.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        coords : np.ndarray, shape (n_samples, 2)
            Spatial coordinates
        k_neighbors : int
            Number of neighbors to consider
            
        Returns:
        --------
        preservation : float
            Neighbor preservation score (0 to 1, higher is better)
        """
        
        # Find k-nearest neighbors in spatial coordinates
        spatial_nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
        spatial_nbrs.fit(coords)
        _, spatial_indices = spatial_nbrs.kneighbors(coords)
        spatial_indices = spatial_indices[:, 1:]  # Exclude self
        
        # Find k-nearest neighbors in embedding space
        embedding_nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
        embedding_nbrs.fit(embeddings)
        _, embedding_indices = embedding_nbrs.kneighbors(embeddings)
        embedding_indices = embedding_indices[:, 1:]  # Exclude self
        
        # Compute preservation score
        total_preservation = 0.0
        n_samples = embeddings.shape[0]
        
        for i in range(n_samples):
            spatial_neighbors = set(spatial_indices[i])
            embedding_neighbors = set(embedding_indices[i])
            
            # Compute Jaccard similarity (intersection over union)
            intersection = spatial_neighbors.intersection(embedding_neighbors)
            preservation = len(intersection) / k_neighbors
            
            total_preservation += preservation
        
        return total_preservation / n_samples
    
    @staticmethod
    def local_spatial_variance(
        embeddings: np.ndarray,
        coords: np.ndarray,
        radius: Optional[float] = None
    ) -> float:
        """
        Compute local spatial variance in embeddings.
        
        Lower variance indicates better spatial smoothness.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        coords : np.ndarray, shape (n_samples, 2)
            Spatial coordinates
        radius : float, optional
            Spatial radius for local neighborhood (auto if None)
            
        Returns:
        --------
        avg_variance : float
            Average local spatial variance (lower is better)
        """
        
        n_samples = embeddings.shape[0]
        
        # Determine radius automatically if not provided
        if radius is None:
            # Use median of pairwise distances
            spatial_distances = pdist(coords)
            radius = np.median(spatial_distances)
        
        # Find neighbors within radius for each point
        spatial_nbrs = NearestNeighbors(radius=radius, metric='euclidean')
        spatial_nbrs.fit(coords)
        neighbor_lists = spatial_nbrs.radius_neighbors(coords, return_distance=False)
        
        # Compute local variance for each point
        local_variances = []
        
        for i in range(n_samples):
            neighbors = neighbor_lists[i]
            
            if len(neighbors) > 1:  # Need at least 2 points for variance
                neighbor_embeddings = embeddings[neighbors]
                local_var = np.var(neighbor_embeddings, axis=0).mean()
                local_variances.append(local_var)
        
        return np.mean(local_variances) if local_variances else 0.0


class ReconstructionMetrics:
    """
    Metrics for evaluating reconstruction quality.
    
    These metrics assess how well the learned latent representations
    preserve the original gene expression information.
    """
    
    @staticmethod
    def reconstruction_mse(
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> float:
        """
        Compute mean squared error between original and reconstructed data.
        
        Parameters:
        -----------
        original : np.ndarray, shape (n_samples, n_features)
            Original data
        reconstructed : np.ndarray, shape (n_samples, n_features)
            Reconstructed data
            
        Returns:
        --------
        mse : float
            Mean squared error
        """
        
        return np.mean((original - reconstructed) ** 2)
    
    @staticmethod
    def reconstruction_correlation(
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> float:
        """
        Compute average correlation between original and reconstructed features.
        
        Parameters:
        -----------
        original : np.ndarray, shape (n_samples, n_features)
            Original data
        reconstructed : np.ndarray, shape (n_samples, n_features)
            Reconstructed data
            
        Returns:
        --------
        avg_correlation : float
            Average Pearson correlation across features
        """
        
        correlations = []
        n_features = original.shape[1]
        
        for i in range(n_features):
            corr, _ = stats.pearsonr(original[:, i], reconstructed[:, i])
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    @staticmethod
    def gene_expression_preservation(
        original: np.ndarray,
        reconstructed: np.ndarray,
        top_genes: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate preservation of highly variable gene expression patterns.
        
        Parameters:
        -----------
        original : np.ndarray, shape (n_samples, n_genes)
            Original gene expression
        reconstructed : np.ndarray, shape (n_samples, n_genes)
            Reconstructed gene expression
        top_genes : int
            Number of top variable genes to focus on
            
        Returns:
        --------
        preservation_metrics : dict
            Dictionary containing various preservation metrics
        """
        
        # Select top variable genes
        gene_variances = np.var(original, axis=0)
        top_gene_indices = np.argsort(gene_variances)[-top_genes:]
        
        # Focus on top variable genes
        orig_top = original[:, top_gene_indices]
        recon_top = reconstructed[:, top_gene_indices]
        
        # Compute metrics
        metrics = {
            'top_genes_mse': np.mean((orig_top - recon_top) ** 2),
            'top_genes_correlation': ReconstructionMetrics.reconstruction_correlation(orig_top, recon_top),
            'expression_range_preservation': np.corrcoef(
                orig_top.max(axis=0) - orig_top.min(axis=0),
                recon_top.max(axis=0) - recon_top.min(axis=0)
            )[0, 1]
        }
        
        return metrics


class ComprehensiveEvaluation:
    """
    Comprehensive evaluation suite for ST-Dynamics models.
    
    Combines all evaluation metrics into a single interface for
    systematic model assessment.
    """
    
    def __init__(self):
        self.monotonicity = TemporalMonotonicityMetrics()
        self.spatial = SpatialSmoothnessMetrics()
        self.reconstruction = ReconstructionMetrics()
    
    def evaluate_model(
        self,
        embeddings: np.ndarray,
        inferred_time: np.ndarray,
        true_time: np.ndarray,
        coords: np.ndarray,
        original_data: Optional[np.ndarray] = None,
        reconstructed_data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of ST-Dynamics model performance.
        
        Parameters:
        -----------
        embeddings : np.ndarray, shape (n_samples, latent_dim)
            Learned latent embeddings
        inferred_time : np.ndarray, shape (n_samples,)
            Inferred latent time
        true_time : np.ndarray, shape (n_samples,)
            True time labels
        coords : np.ndarray, shape (n_samples, 2)
            Spatial coordinates
        original_data : np.ndarray, optional
            Original gene expression data
        reconstructed_data : np.ndarray, optional
            Reconstructed gene expression data
            
        Returns:
        --------
        metrics : dict
            Comprehensive evaluation metrics
        """
        
        metrics = {}
        
        # Temporal monotonicity metrics
        metrics['kendall_tau'] = self.monotonicity.kendall_tau_correlation(inferred_time, true_time)
        metrics['spearman_rho'] = self.monotonicity.spearman_correlation(inferred_time, true_time)
        metrics['temporal_ordering_accuracy'] = self.monotonicity.temporal_ordering_accuracy(inferred_time, true_time)
        metrics['temporal_coherence'] = self.monotonicity.temporal_coherence_score(inferred_time, true_time)
        
        # Spatial smoothness metrics
        metrics['spatial_autocorrelation'] = self.spatial.spatial_autocorrelation(embeddings, coords)
        metrics['neighbor_preservation'] = self.spatial.spatial_neighbor_preservation(embeddings, coords)
        metrics['local_spatial_variance'] = self.spatial.local_spatial_variance(embeddings, coords)
        
        # Reconstruction metrics (if data available)
        if original_data is not None and reconstructed_data is not None:
            metrics['reconstruction_mse'] = self.reconstruction.reconstruction_mse(original_data, reconstructed_data)
            metrics['reconstruction_correlation'] = self.reconstruction.reconstruction_correlation(original_data, reconstructed_data)
            
            # Gene expression preservation
            preservation_metrics = self.reconstruction.gene_expression_preservation(original_data, reconstructed_data)
            metrics.update(preservation_metrics)
        
        # Compute composite scores
        metrics['temporal_performance'] = np.mean([
            metrics['kendall_tau'],
            metrics['spearman_rho'], 
            metrics['temporal_ordering_accuracy'],
            metrics['temporal_coherence']
        ])
        
        metrics['spatial_performance'] = np.mean([
            metrics['spatial_autocorrelation'],
            metrics['neighbor_preservation'],
            1.0 - metrics['local_spatial_variance']  # Lower variance is better
        ])
        
        # Overall performance (harmonic mean of temporal and spatial)
        if metrics['temporal_performance'] > 0 and metrics['spatial_performance'] > 0:
            metrics['overall_performance'] = 2 * (
                metrics['temporal_performance'] * metrics['spatial_performance']
            ) / (metrics['temporal_performance'] + metrics['spatial_performance'])
        else:
            metrics['overall_performance'] = 0.0
        
        return metrics
    
    def compare_methods(
        self,
        results_dict: Dict[str, Dict[str, np.ndarray]],
        true_time: np.ndarray,
        coords: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare multiple methods using comprehensive evaluation metrics.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with method names as keys and results as values.
            Each result should contain 'embeddings' and 'inferred_time'
        true_time : np.ndarray
            True time labels
        coords : np.ndarray
            Spatial coordinates
            
        Returns:
        --------
        comparison_df : pd.DataFrame
            DataFrame comparing all methods across all metrics
        """
        
        comparison_results = {}
        
        for method_name, results in results_dict.items():
            embeddings = results['embeddings']
            inferred_time = results['inferred_time']
            
            # Get optional reconstruction data
            original_data = results.get('original_data')
            reconstructed_data = results.get('reconstructed_data')
            
            # Evaluate method
            metrics = self.evaluate_model(
                embeddings=embeddings,
                inferred_time=inferred_time,
                true_time=true_time,
                coords=coords,
                original_data=original_data,
                reconstructed_data=reconstructed_data
            )
            
            comparison_results[method_name] = metrics
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        # Sort by overall performance
        comparison_df = comparison_df.sort_values('overall_performance', ascending=False)
        
        return comparison_df


def evaluate_latent_time_inference(
    inferred_time: np.ndarray,
    true_time: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
    detailed: bool = True
) -> Dict[str, float]:
    """
    Quick evaluation function for latent time inference quality.
    
    Parameters:
    -----------
    inferred_time : np.ndarray
        Inferred continuous latent time
    true_time : np.ndarray
        True discrete time labels
    embeddings : np.ndarray, optional
        Latent embeddings (for spatial analysis)
    coords : np.ndarray, optional
        Spatial coordinates (for spatial analysis)
    detailed : bool
        Whether to include detailed spatial metrics
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    
    evaluator = ComprehensiveEvaluation()
    
    if detailed and embeddings is not None and coords is not None:
        return evaluator.evaluate_model(
            embeddings=embeddings,
            inferred_time=inferred_time,
            true_time=true_time,
            coords=coords
        )
    else:
        # Basic temporal metrics only
        monotonicity = TemporalMonotonicityMetrics()
        return {
            'kendall_tau': monotonicity.kendall_tau_correlation(inferred_time, true_time),
            'spearman_rho': monotonicity.spearman_correlation(inferred_time, true_time),
            'temporal_ordering_accuracy': monotonicity.temporal_ordering_accuracy(inferred_time, true_time),
            'temporal_coherence': monotonicity.temporal_coherence_score(inferred_time, true_time)
        }