"""
Synthetic data experiments for ST-Dynamics validation.

This module implements synthetic spatial transcriptomics datasets
with known temporal dynamics for algorithm validation and benchmarking.

Synthetic data allows for:
1. Controlled temporal progression validation
2. Ground truth comparison for method development
3. Robustness testing under various noise conditions
4. Ablation studies on algorithm components
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Callable
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
import warnings

from ..config import Config
from ..data.dataset import SpatialTranscriptomicsDataset
from ..model.model import STDynamicsModel
from ..inference.latent_time import LatentTimeInference
from ..evaluation.metrics import ComprehensiveEvaluation
from ..evaluation.baselines import BaselineComparison


class SyntheticDataGenerator:
    """
    Generator for synthetic spatial transcriptomics data with temporal dynamics.
    
    Creates realistic synthetic datasets that mimic spatial transcriptomics
    experiments with controlled temporal progression patterns.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.random_state = np.random.RandomState(self.config.random_seed)
        
    def generate_spatial_grid(
        self,
        n_spots_per_time: int = 500,
        grid_type: str = "hexagonal",
        spatial_noise: float = 0.1
    ) -> np.ndarray:
        """
        Generate spatial coordinates in a tissue-like arrangement.
        
        Parameters:
        -----------
        n_spots_per_time : int
            Number of spots per timepoint
        grid_type : str
            Type of spatial arrangement ("hexagonal", "square", "random")
        spatial_noise : float
            Amount of noise to add to regular grid positions
            
        Returns:
        --------
        coords : np.ndarray, shape (n_spots, 2)
            Spatial coordinates (x, y)
        """
        
        if grid_type == "hexagonal":
            # Create hexagonal grid
            sqrt_n = int(np.sqrt(n_spots_per_time))
            x_coords, y_coords = [], []
            
            for i in range(sqrt_n):
                for j in range(sqrt_n):
                    x = j + (0.5 if i % 2 == 1 else 0.0)
                    y = i * np.sqrt(3) / 2
                    x_coords.append(x)
                    y_coords.append(y)
            
            # Trim to exact number needed
            coords = np.column_stack([x_coords, y_coords])[:n_spots_per_time]
            
        elif grid_type == "square":
            # Create square grid
            sqrt_n = int(np.sqrt(n_spots_per_time))
            x_vals = np.linspace(0, 10, sqrt_n)
            y_vals = np.linspace(0, 10, sqrt_n)
            x_coords, y_coords = np.meshgrid(x_vals, y_vals)
            coords = np.column_stack([x_coords.flatten(), y_coords.flatten()])[:n_spots_per_time]
            
        elif grid_type == "random":
            # Random spatial distribution
            coords = self.random_state.uniform(0, 10, size=(n_spots_per_time, 2))
            
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")
        
        # Add spatial noise
        if spatial_noise > 0:
            noise = self.random_state.normal(0, spatial_noise, coords.shape)
            coords += noise
        
        return coords
    
    def generate_temporal_trajectory(
        self,
        n_timepoints: int = 4,
        trajectory_type: str = "linear",
        trajectory_params: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Callable]:
        """
        Generate temporal trajectory function and timepoints.
        
        Parameters:
        -----------
        n_timepoints : int
            Number of discrete timepoints
        trajectory_type : str
            Type of temporal trajectory ("linear", "sigmoid", "oscillatory", "branching")
        trajectory_params : dict, optional
            Parameters specific to trajectory type
            
        Returns:
        --------
        timepoints : np.ndarray, shape (n_timepoints,)
            Discrete timepoints
        trajectory_func : callable
            Function that maps continuous time [0,1] to trajectory parameters
        """
        
        params = trajectory_params or {}
        timepoints = np.linspace(0, 1, n_timepoints)
        
        if trajectory_type == "linear":
            # Linear temporal progression
            def trajectory_func(t):
                return {'progression': t, 'branch_factor': 0.0}
                
        elif trajectory_type == "sigmoid":
            # Sigmoid temporal progression (slow-fast-slow)
            steepness = params.get('steepness', 5.0)
            midpoint = params.get('midpoint', 0.5)
            
            def trajectory_func(t):
                progression = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
                return {'progression': progression, 'branch_factor': 0.0}
                
        elif trajectory_type == "oscillatory":
            # Oscillatory progression with trend
            frequency = params.get('frequency', 2.0)
            amplitude = params.get('amplitude', 0.3)
            
            def trajectory_func(t):
                progression = t + amplitude * np.sin(2 * np.pi * frequency * t)
                progression = np.clip(progression, 0, 1)
                return {'progression': progression, 'branch_factor': 0.0}
                
        elif trajectory_type == "branching":
            # Branching trajectory (bifurcation)
            branch_time = params.get('branch_time', 0.5)
            branch_strength = params.get('branch_strength', 0.5)
            
            def trajectory_func(t):
                if t <= branch_time:
                    progression = t / branch_time * 0.5
                    branch_factor = 0.0
                else:
                    progression = 0.5 + (t - branch_time) / (1 - branch_time) * 0.5
                    branch_factor = branch_strength * (t - branch_time) / (1 - branch_time)
                
                return {'progression': progression, 'branch_factor': branch_factor}
        
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
        return timepoints, trajectory_func
    
    def generate_gene_expression(
        self,
        coords: np.ndarray,
        timepoints: np.ndarray,
        trajectory_func: Callable,
        n_genes: int = 2000,
        n_temporal_genes: int = 500,
        n_spatial_genes: int = 300,
        expression_noise: float = 0.2,
        batch_effects: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Generate gene expression data with spatial and temporal patterns.
        
        Parameters:
        -----------
        coords : np.ndarray, shape (n_spots_per_time, 2)
            Spatial coordinates template
        timepoints : np.ndarray
            Temporal timepoints
        trajectory_func : callable
            Temporal trajectory function
        n_genes : int
            Total number of genes
        n_temporal_genes : int
            Number of temporally regulated genes
        n_spatial_genes : int
            Number of spatially regulated genes
        expression_noise : float
            Expression noise level
        batch_effects : bool
            Whether to add batch effects
            
        Returns:
        --------
        X : np.ndarray, shape (n_total_spots, n_genes)
            Gene expression matrix
        coords_full : np.ndarray, shape (n_total_spots, 2)
            Full spatial coordinates
        time_labels : np.ndarray, shape (n_total_spots,)
            Time labels for each spot
        metadata : dict
            Generation metadata and ground truth information
        """
        
        n_timepoints = len(timepoints)
        n_spots_per_time = coords.shape[0]
        n_total_spots = n_timepoints * n_spots_per_time
        
        # Initialize expression matrix
        X = np.zeros((n_total_spots, n_genes))
        coords_full = np.tile(coords, (n_timepoints, 1))
        time_labels = np.repeat(np.arange(n_timepoints), n_spots_per_time)
        
        # Gene type assignments
        gene_types = np.array(['background'] * n_genes)
        temporal_indices = self.random_state.choice(n_genes, n_temporal_genes, replace=False)
        spatial_indices = self.random_state.choice(
            [i for i in range(n_genes) if i not in temporal_indices], 
            n_spatial_genes, 
            replace=False
        )
        
        gene_types[temporal_indices] = 'temporal'
        gene_types[spatial_indices] = 'spatial'
        
        # Generate baseline expression
        baseline_expression = self.random_state.lognormal(0, 1, size=(n_total_spots, n_genes))
        X += baseline_expression
        
        # Add temporal patterns
        for i, t in enumerate(timepoints):
            spot_start = i * n_spots_per_time
            spot_end = (i + 1) * n_spots_per_time
            
            trajectory_state = trajectory_func(t)
            progression = trajectory_state['progression']
            branch_factor = trajectory_state['branch_factor']
            
            # Temporal gene expression patterns
            for j, gene_idx in enumerate(temporal_indices):
                # Different temporal patterns
                pattern_type = j % 4
                
                if pattern_type == 0:  # Increasing
                    temporal_factor = 1 + 2 * progression
                elif pattern_type == 1:  # Decreasing
                    temporal_factor = 1 + 2 * (1 - progression)
                elif pattern_type == 2:  # Peak in middle
                    temporal_factor = 1 + 3 * (4 * progression * (1 - progression))
                else:  # Branching-sensitive
                    temporal_factor = 1 + progression + branch_factor
                
                X[spot_start:spot_end, gene_idx] *= temporal_factor
        
        # Add spatial patterns
        for j, gene_idx in enumerate(spatial_indices):
            # Different spatial patterns
            pattern_type = j % 3
            
            if pattern_type == 0:  # Gradient along x-axis
                spatial_factor = 1 + coords_full[:, 0] / np.max(coords_full[:, 0])
            elif pattern_type == 1:  # Gradient along y-axis
                spatial_factor = 1 + coords_full[:, 1] / np.max(coords_full[:, 1])
            else:  # Radial pattern
                center = np.mean(coords_full, axis=0)
                distances = np.linalg.norm(coords_full - center, axis=1)
                spatial_factor = 1 + np.exp(-distances / np.std(distances))
            
            X[:, gene_idx] *= spatial_factor
        
        # Add expression noise
        if expression_noise > 0:
            noise = self.random_state.lognormal(0, expression_noise, X.shape)
            X *= noise
        
        # Add batch effects if requested
        batch_labels = None
        if batch_effects:
            n_batches = min(3, n_timepoints)
            batch_labels = np.repeat(
                np.arange(n_batches), 
                n_total_spots // n_batches + 1
            )[:n_total_spots]
            
            # Add batch-specific expression shifts
            for batch_id in range(n_batches):
                batch_mask = (batch_labels == batch_id)
                batch_effect = self.random_state.normal(1, 0.1, n_genes)
                X[batch_mask] *= batch_effect[np.newaxis, :]
        
        # Ensure positive expression values
        X = np.maximum(X, 0.1)
        
        # Store metadata
        metadata = {
            'n_timepoints': n_timepoints,
            'n_spots_per_time': n_spots_per_time,
            'timepoints': timepoints,
            'trajectory_func': trajectory_func,
            'gene_types': gene_types,
            'temporal_genes': temporal_indices,
            'spatial_genes': spatial_indices,
            'expression_noise': expression_noise,
            'batch_effects': batch_effects
        }
        
        return X, coords_full, time_labels, metadata
    
    def generate_synthetic_dataset(
        self,
        n_timepoints: int = 4,
        n_spots_per_time: int = 500,
        n_genes: int = 2000,
        trajectory_type: str = "linear",
        grid_type: str = "hexagonal",
        expression_noise: float = 0.2,
        spatial_noise: float = 0.1,
        batch_effects: bool = False,
        **kwargs
    ) -> Tuple[SpatialTranscriptomicsDataset, Dict]:
        """
        Generate complete synthetic spatial transcriptomics dataset.
        
        Parameters:
        -----------
        n_timepoints : int
            Number of temporal timepoints
        n_spots_per_time : int
            Number of spots per timepoint
        n_genes : int
            Number of genes
        trajectory_type : str
            Type of temporal trajectory
        grid_type : str
            Type of spatial arrangement
        expression_noise : float
            Expression noise level
        spatial_noise : float
            Spatial coordinate noise
        batch_effects : bool
            Whether to include batch effects
        **kwargs
            Additional parameters for trajectory and spatial generation
            
        Returns:
        --------
        dataset : SpatialTranscriptomicsDataset
            Complete synthetic dataset
        metadata : dict
            Generation parameters and ground truth information
        """
        
        print(f"Generating synthetic dataset:")
        print(f"  - {n_timepoints} timepoints x {n_spots_per_time} spots = {n_timepoints * n_spots_per_time} total spots")
        print(f"  - {n_genes} genes")
        print(f"  - Trajectory: {trajectory_type}")
        print(f"  - Spatial: {grid_type}")
        
        # Generate spatial coordinates
        coords = self.generate_spatial_grid(
            n_spots_per_time, grid_type, spatial_noise
        )
        
        # Generate temporal trajectory
        timepoints, trajectory_func = self.generate_temporal_trajectory(
            n_timepoints, trajectory_type, kwargs.get('trajectory_params')
        )
        
        # Generate gene expression
        X, coords_full, time_labels, expr_metadata = self.generate_gene_expression(
            coords, timepoints, trajectory_func, n_genes,
            kwargs.get('n_temporal_genes', 500),
            kwargs.get('n_spatial_genes', 300),
            expression_noise, batch_effects
        )
        
        # Create batch labels if requested
        batch_labels = None
        if batch_effects:
            n_batches = min(3, n_timepoints)
            batch_labels = np.repeat(
                np.arange(n_batches),
                len(time_labels) // n_batches + 1
            )[:len(time_labels)]
        
        # Create gene and spot names
        gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
        spot_names = [f"Spot_T{t}_S{s:03d}" for t in range(n_timepoints) for s in range(n_spots_per_time)]
        
        # Create dataset
        dataset = SpatialTranscriptomicsDataset(
            X=X,
            coords=coords_full,
            time=time_labels,
            batch=batch_labels,
            gene_names=gene_names,
            sample_names=spot_names,
            config=self.config
        )
        
        # Combine metadata
        metadata = {
            'generation_params': {
                'n_timepoints': n_timepoints,
                'n_spots_per_time': n_spots_per_time,
                'n_genes': n_genes,
                'trajectory_type': trajectory_type,
                'grid_type': grid_type,
                'expression_noise': expression_noise,
                'spatial_noise': spatial_noise,
                'batch_effects': batch_effects
            },
            'ground_truth': expr_metadata,
            'true_trajectory': trajectory_func
        }
        
        print(f"Dataset generated successfully!")
        print(f"  - Final shape: {dataset.X_processed.shape}")
        
        return dataset, metadata


class SyntheticExperiment:
    """
    Complete synthetic data experiment framework.
    
    Runs comprehensive experiments on synthetic data including:
    - Model training and evaluation
    - Baseline comparisons
    - Ablation studies
    - Robustness testing
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.generator = SyntheticDataGenerator(config)
        self.results = {}
        
    def run_basic_experiment(
        self,
        dataset: SpatialTranscriptomicsDataset,
        metadata: Dict,
        run_baselines: bool = True,
        save_results: bool = False
    ) -> Dict:
        """
        Run basic synthetic data experiment.
        
        Parameters:
        -----------
        dataset : SpatialTranscriptomicsDataset
            Synthetic dataset
        metadata : dict
            Dataset generation metadata
        run_baselines : bool
            Whether to run baseline comparisons
        save_results : bool
            Whether to save results
            
        Returns:
        --------
        results : dict
            Complete experiment results
        """
        
        print("\n=== Running Basic Synthetic Experiment ===")
        
        # Split data into train/test
        n_spots = len(dataset)
        indices = np.arange(n_spots)
        np.random.shuffle(indices)
        
        split_idx = int(0.8 * n_spots)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Create train/test datasets
        # Note: This is a simplified split - in practice, should be more sophisticated
        train_X = dataset.X_processed[train_indices]
        train_coords = dataset.coords[train_indices]
        train_time = dataset.time[train_indices]
        train_batch = dataset.batch[train_indices] if dataset.batch is not None else None
        
        test_X = dataset.X_processed[test_indices]
        test_coords = dataset.coords[test_indices]
        test_time = dataset.time[test_indices]
        
        train_dataset = SpatialTranscriptomicsDataset(
            train_X, train_coords, train_time, train_batch, config=self.config
        )
        
        test_dataset = SpatialTranscriptomicsDataset(
            test_X, test_coords, test_time, config=self.config
        )
        
        # Train ST-Dynamics model
        print("\n--- Training ST-Dynamics Model ---")
        model = STDynamicsModel(self.config, encoder_type="mlp")
        
        # Train model
        training_history = model.fit(
            train_dataset,
            val_dataset=None,  # Use train for validation in synthetic case
            verbose=True
        )
        
        # Generate embeddings and infer latent time
        print("\n--- Inference and Evaluation ---")
        test_embeddings = model.infer_latent(test_dataset)
        
        # Infer latent time
        latent_time_inference = LatentTimeInference(self.config)
        inferred_time = latent_time_inference.infer_latent_time(
            test_embeddings,
            test_coords,
            method=self.config.latent_time_method
        )
        
        # Evaluate ST-Dynamics results
        evaluator = ComprehensiveEvaluation()
        st_dynamics_metrics = evaluator.evaluate_model(
            embeddings=test_embeddings,
            inferred_time=inferred_time,
            true_time=test_time,
            coords=test_coords
        )
        
        results = {
            'st_dynamics': {
                'embeddings': test_embeddings,
                'inferred_time': inferred_time,
                'metrics': st_dynamics_metrics,
                'training_history': training_history,
                'model': model
            },
            'dataset_info': {
                'n_train': len(train_dataset),
                'n_test': len(test_dataset),
                'metadata': metadata
            }
        }
        
        # Run baseline comparisons
        if run_baselines:
            print("\n--- Baseline Comparisons ---")
            baseline_comparison = BaselineComparison(self.config)
            
            # Use test dataset for baseline comparison
            baseline_results = baseline_comparison.run_comparison(test_dataset)
            
            # Evaluate all baselines
            for method_name, baseline_result in baseline_results.items():
                if baseline_result['success'] and baseline_result['inferred_time'] is not None:
                    
                    baseline_metrics = evaluator.evaluate_model(
                        embeddings=baseline_result.get('embedding'),
                        inferred_time=baseline_result['inferred_time'],
                        true_time=test_time,
                        coords=test_coords
                    )
                    
                    baseline_result['metrics'] = baseline_metrics
            
            results['baselines'] = baseline_results
            
            # Create comparison table
            comparison_data = {}
            
            # Add ST-Dynamics
            comparison_data['ST_Dynamics'] = st_dynamics_metrics
            
            # Add successful baselines
            for method_name, baseline_result in baseline_results.items():
                if baseline_result['success'] and 'metrics' in baseline_result:
                    comparison_data[method_name] = baseline_result['metrics']
            
            results['comparison_table'] = pd.DataFrame(comparison_data).T
            
            print("\n--- Method Comparison ---")
            print(results['comparison_table'][['kendall_tau', 'spearman_rho', 'temporal_performance', 'spatial_performance', 'overall_performance']].round(4))
        
        self.results['basic_experiment'] = results
        
        return results
    
    def run_ablation_study(
        self,
        base_dataset: SpatialTranscriptomicsDataset,
        metadata: Dict
    ) -> Dict:
        """
        Run ablation study on loss components.
        
        Tests the contribution of different loss components to model performance.
        """
        
        print("\n=== Running Ablation Study ===")
        
        # Define ablation configurations
        ablation_configs = {
            'full_model': {
                'lambda_recon': 1.0,
                'lambda_temporal': 0.5,
                'lambda_spatial': 0.3
            },
            'no_temporal': {
                'lambda_recon': 1.0,
                'lambda_temporal': 0.0,
                'lambda_spatial': 0.3
            },
            'no_spatial': {
                'lambda_recon': 1.0,
                'lambda_temporal': 0.5,
                'lambda_spatial': 0.0
            },
            'reconstruction_only': {
                'lambda_recon': 1.0,
                'lambda_temporal': 0.0,
                'lambda_spatial': 0.0
            }
        }
        
        ablation_results = {}
        evaluator = ComprehensiveEvaluation()
        
        for config_name, lambda_config in ablation_configs.items():
            print(f"\n--- Testing {config_name} ---")
            
            # Create modified config
            modified_config = Config()
            for key, value in vars(self.config).items():
                setattr(modified_config, key, value)
            
            for key, value in lambda_config.items():
                setattr(modified_config, key, value)
            
            # Train model with modified config
            model = STDynamicsModel(modified_config, encoder_type="mlp")
            
            try:
                training_history = model.fit(
                    base_dataset,
                    max_epochs=50,  # Shorter for ablation
                    verbose=False
                )
                
                # Evaluate
                embeddings = model.infer_latent(base_dataset)
                
                latent_time_inference = LatentTimeInference(modified_config)
                inferred_time = latent_time_inference.infer_latent_time(
                    embeddings,
                    base_dataset.coords
                )
                
                metrics = evaluator.evaluate_model(
                    embeddings=embeddings,
                    inferred_time=inferred_time,
                    true_time=base_dataset.time,
                    coords=base_dataset.coords
                )
                
                ablation_results[config_name] = {
                    'config': lambda_config,
                    'metrics': metrics,
                    'success': True
                }
                
                print(f"  Overall performance: {metrics['overall_performance']:.4f}")
                
            except Exception as e:
                print(f"  Failed: {e}")
                ablation_results[config_name] = {
                    'config': lambda_config,
                    'metrics': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Create comparison table
        comparison_data = {}
        for config_name, result in ablation_results.items():
            if result['success']:
                comparison_data[config_name] = result['metrics']
        
        ablation_table = pd.DataFrame(comparison_data).T
        
        print("\n--- Ablation Study Results ---")
        print(ablation_table[['temporal_performance', 'spatial_performance', 'overall_performance']].round(4))
        
        self.results['ablation_study'] = {
            'results': ablation_results,
            'comparison_table': ablation_table
        }
        
        return self.results['ablation_study']
    
    def run_robustness_test(
        self,
        base_params: Dict,
        noise_levels: List[float] = [0.1, 0.2, 0.3, 0.5]
    ) -> Dict:
        """
        Test model robustness to different noise levels.
        
        Parameters:
        -----------
        base_params : dict
            Base parameters for dataset generation
        noise_levels : list
            List of noise levels to test
            
        Returns:
        --------
        robustness_results : dict
            Results for each noise level
        """
        
        print("\n=== Running Robustness Test ===")
        
        robustness_results = {}
        evaluator = ComprehensiveEvaluation()
        
        for noise_level in noise_levels:
            print(f"\n--- Testing noise level: {noise_level} ---")
            
            # Generate dataset with specified noise level
            modified_params = base_params.copy()
            modified_params['expression_noise'] = noise_level
            modified_params['spatial_noise'] = noise_level * 0.5  # Scale spatial noise
            
            dataset, metadata = self.generator.generate_synthetic_dataset(**modified_params)
            
            # Train and evaluate model
            try:
                model = STDynamicsModel(self.config, encoder_type="mlp")
                
                training_history = model.fit(
                    dataset,
                    max_epochs=50,  # Shorter for robustness test
                    verbose=False
                )
                
                embeddings = model.infer_latent(dataset)
                
                latent_time_inference = LatentTimeInference(self.config)
                inferred_time = latent_time_inference.infer_latent_time(
                    embeddings,
                    dataset.coords
                )
                
                metrics = evaluator.evaluate_model(
                    embeddings=embeddings,
                    inferred_time=inferred_time,
                    true_time=dataset.time,
                    coords=dataset.coords
                )
                
                robustness_results[f"noise_{noise_level}"] = {
                    'noise_level': noise_level,
                    'metrics': metrics,
                    'success': True
                }
                
                print(f"  Overall performance: {metrics['overall_performance']:.4f}")
                
            except Exception as e:
                print(f"  Failed: {e}")
                robustness_results[f"noise_{noise_level}"] = {
                    'noise_level': noise_level,
                    'metrics': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Create robustness summary
        noise_performance = []
        for noise_level in noise_levels:
            result = robustness_results[f"noise_{noise_level}"]
            if result['success']:
                noise_performance.append({
                    'noise_level': noise_level,
                    'overall_performance': result['metrics']['overall_performance'],
                    'temporal_performance': result['metrics']['temporal_performance'],
                    'spatial_performance': result['metrics']['spatial_performance']
                })
        
        robustness_summary = pd.DataFrame(noise_performance)
        
        print("\n--- Robustness Test Results ---")
        print(robustness_summary.round(4))
        
        self.results['robustness_test'] = {
            'results': robustness_results,
            'summary': robustness_summary
        }
        
        return self.results['robustness_test']


def run_synthetic_validation_suite(
    config: Optional[Config] = None,
    quick_test: bool = False
) -> Dict:
    """
    Run complete synthetic validation suite.
    
    Parameters:
    -----------
    config : Config, optional
        Configuration object
    quick_test : bool
        Whether to run quick test version (smaller datasets, fewer experiments)
        
    Returns:
    --------
    all_results : dict
        Complete validation results
    """
    
    print("=" * 60)
    print("ST-DYNAMICS SYNTHETIC VALIDATION SUITE")
    print("=" * 60)
    
    config = config or Config()
    experiment = SyntheticExperiment(config)
    
    # Set parameters based on test mode
    if quick_test:
        dataset_params = {
            'n_timepoints': 3,
            'n_spots_per_time': 200,
            'n_genes': 500,
            'trajectory_type': 'linear'
        }
        noise_levels = [0.1, 0.3]
    else:
        dataset_params = {
            'n_timepoints': 4,
            'n_spots_per_time': 500,
            'n_genes': 2000,
            'trajectory_type': 'sigmoid'
        }
        noise_levels = [0.1, 0.2, 0.3, 0.5]
    
    # Generate base dataset
    print("\nGenerating base synthetic dataset...")
    base_dataset, metadata = experiment.generator.generate_synthetic_dataset(**dataset_params)
    
    # Run basic experiment
    basic_results = experiment.run_basic_experiment(
        base_dataset, metadata, run_baselines=True
    )
    
    # Run ablation study
    if not quick_test:
        ablation_results = experiment.run_ablation_study(base_dataset, metadata)
        
        # Run robustness test
        robustness_results = experiment.run_robustness_test(
            dataset_params, noise_levels
        )
    else:
        ablation_results = {'skipped': 'Quick test mode'}
        robustness_results = {'skipped': 'Quick test mode'}
    
    # Compile all results
    all_results = {
        'basic_experiment': basic_results,
        'ablation_study': ablation_results,
        'robustness_test': robustness_results,
        'config': config,
        'dataset_params': dataset_params
    }
    
    print("\n" + "=" * 60)
    print("SYNTHETIC VALIDATION SUITE COMPLETED")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    # Run quick validation
    results = run_synthetic_validation_suite(quick_test=True)
    
    print("\nValidation completed successfully!")
    if 'comparison_table' in results['basic_experiment']:
        print("\nMethod performance summary:")
        print(results['basic_experiment']['comparison_table'][['overall_performance']].round(4))