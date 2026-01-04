"""
Gastric longitudinal experiment for ST-Dynamics validation on real data.

This module implements experiments on gastric tissue spatial transcriptomics
data with multiple timepoints from the same patients, specifically designed
for validating temporal dynamics analysis methods.

Experimental design:
- Same patient, multiple time points
- Each timepoint has multiple spatial spots
- Focus on gastric cancer progression or treatment response
- Validation against known biological processes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

from ..config import Config
from ..data.dataset import SpatialTranscriptomicsDataset
from ..data.preprocessing import SpatialTranscriptomicsPreprocessor
from ..model.model import STDynamicsModel
from ..inference.latent_time import LatentTimeInference
from ..evaluation.metrics import ComprehensiveEvaluation
from ..evaluation.baselines import BaselineComparison


class GastricLongitudinalExperiment:
    """
    Gastric tissue longitudinal experiment framework.
    
    Handles loading, preprocessing, and analysis of gastric tissue
    spatial transcriptomics data with temporal components.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.preprocessor = SpatialTranscriptomicsPreprocessor(self.config)
        self.datasets = {}
        self.results = {}
        
    def load_gastric_data(
        self,
        data_path: Union[str, Path],
        patient_ids: Optional[List[str]] = None,
        timepoint_mapping: Optional[Dict[str, int]] = None
    ) -> Dict[str, SpatialTranscriptomicsDataset]:
        """
        Load gastric longitudinal spatial transcriptomics data.
        
        Expected data format:
        - Gene expression matrices for each patient/timepoint
        - Spatial coordinates for each spot
        - Metadata including patient ID, timepoint, clinical information
        
        Parameters:
        -----------
        data_path : str or Path
            Path to data directory
        patient_ids : list, optional
            Specific patient IDs to load
        timepoint_mapping : dict, optional
            Mapping from timepoint labels to integers
            
        Returns:
        --------
        datasets : dict
            Dictionary of datasets by patient ID
        """
        
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        print(f"Loading gastric longitudinal data from: {data_path}")
        
        datasets = {}
        
        # This is a template implementation - actual loading depends on data format
        # Common formats: H5AD, CSV, HDF5, etc.
        
        try:
            # Example loading logic - adapt based on actual data format
            for patient_dir in data_path.glob("patient_*"):
                if not patient_dir.is_dir():
                    continue
                    
                patient_id = patient_dir.name
                
                if patient_ids is not None and patient_id not in patient_ids:
                    continue
                
                print(f"  Loading patient: {patient_id}")
                
                # Load patient data
                patient_data = self._load_patient_data(patient_dir, timepoint_mapping)
                
                if patient_data is not None:
                    datasets[patient_id] = patient_data
                    
        except Exception as e:
            warnings.warn(f"Error loading data: {e}")
            # Fallback: create synthetic gastric-like data for demonstration
            print("Creating synthetic gastric data for demonstration...")
            datasets = self._create_demo_gastric_data()
        
        self.datasets = datasets
        print(f"Loaded data for {len(datasets)} patients")
        
        return datasets
    
    def _load_patient_data(
        self,
        patient_dir: Path,
        timepoint_mapping: Optional[Dict[str, int]] = None
    ) -> Optional[SpatialTranscriptomicsDataset]:
        """
        Load data for a single patient.
        
        This is a template function - implement based on actual data format.
        """
        
        try:
            # Look for common file patterns
            h5ad_files = list(patient_dir.glob("*.h5ad"))
            csv_files = list(patient_dir.glob("*expression*.csv"))
            
            if h5ad_files:
                # Load from H5AD format (scanpy/anndata)
                import anndata as ad
                
                all_timepoints = []
                coords_all = []
                time_labels = []
                
                for h5ad_file in sorted(h5ad_files):
                    # Extract timepoint from filename
                    timepoint_str = self._extract_timepoint_from_filename(h5ad_file.name)
                    timepoint = timepoint_mapping.get(timepoint_str, len(all_timepoints)) if timepoint_mapping else len(all_timepoints)
                    
                    adata = ad.read_h5ad(h5ad_file)
                    
                    # Extract expression data
                    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
                    
                    # Extract spatial coordinates
                    if 'spatial' in adata.obsm:
                        coords = adata.obsm['spatial']
                    elif 'X_spatial' in adata.obsm:
                        coords = adata.obsm['X_spatial']
                    else:
                        # Create dummy coordinates if not available
                        coords = np.random.rand(X.shape[0], 2) * 100
                    
                    all_timepoints.append(X)
                    coords_all.append(coords)
                    time_labels.extend([timepoint] * X.shape[0])
                
                # Combine all timepoints
                X_combined = np.vstack(all_timepoints)
                coords_combined = np.vstack(coords_all)
                time_combined = np.array(time_labels)
                
                # Create dataset
                gene_names = adata.var_names.tolist()
                spot_names = [f"Spot_T{t}_S{i}" for t in time_combined for i in range(len(time_combined))]
                
                dataset = SpatialTranscriptomicsDataset(
                    X=X_combined,
                    coords=coords_combined,
                    time=time_combined,
                    gene_names=gene_names,
                    sample_names=spot_names[:len(X_combined)],  # Adjust length
                    config=self.config
                )
                
                return dataset
            
            elif csv_files:
                # Load from CSV format
                return self._load_from_csv(patient_dir, timepoint_mapping)
            
            else:
                print(f"    No recognized data files found in {patient_dir}")
                return None
                
        except Exception as e:
            print(f"    Error loading patient data: {e}")
            return None
    
    def _load_from_csv(
        self,
        patient_dir: Path,
        timepoint_mapping: Optional[Dict[str, int]] = None
    ) -> Optional[SpatialTranscriptomicsDataset]:
        """Load data from CSV files."""
        
        try:
            # Look for expression, coordinates, and metadata files
            expr_file = patient_dir / "expression.csv"
            coords_file = patient_dir / "coordinates.csv"
            meta_file = patient_dir / "metadata.csv"
            
            if not all(f.exists() for f in [expr_file, coords_file, meta_file]):
                return None
            
            # Load expression data
            expr_df = pd.read_csv(expr_file, index_col=0)
            X = expr_df.values
            gene_names = expr_df.columns.tolist()
            
            # Load coordinates
            coords_df = pd.read_csv(coords_file, index_col=0)
            coords = coords_df[['x', 'y']].values
            
            # Load metadata
            meta_df = pd.read_csv(meta_file, index_col=0)
            time_labels = meta_df['timepoint'].values
            
            # Map timepoint labels to integers if provided
            if timepoint_mapping:
                time_labels = np.array([timepoint_mapping.get(str(t), t) for t in time_labels])
            
            # Create dataset
            spot_names = expr_df.index.tolist()
            
            dataset = SpatialTranscriptomicsDataset(
                X=X,
                coords=coords,
                time=time_labels,
                gene_names=gene_names,
                sample_names=spot_names,
                config=self.config
            )
            
            return dataset
            
        except Exception as e:
            print(f"    Error loading CSV data: {e}")
            return None
    
    def _extract_timepoint_from_filename(self, filename: str) -> str:
        """Extract timepoint information from filename."""
        
        # Common patterns: "timepoint_1.h5ad", "t1_data.h5ad", "pre_treatment.h5ad", etc.
        import re
        
        # Try different patterns
        patterns = [
            r't(\d+)',
            r'timepoint[_\-](\d+)',
            r'time[_\-](\d+)',
            r'day[_\-](\d+)',
            r'week[_\-](\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return match.group(1)
        
        # Fallback: use full filename
        return filename.replace('.h5ad', '').replace('.csv', '')
    
    def _create_demo_gastric_data(self) -> Dict[str, SpatialTranscriptomicsDataset]:
        """
        Create synthetic gastric-like data for demonstration.
        
        Simulates gastric cancer progression with realistic spatial patterns.
        """
        
        print("Creating synthetic gastric demonstration data...")
        
        from .synthetic import SyntheticDataGenerator
        generator = SyntheticDataGenerator(self.config)
        
        datasets = {}
        
        # Create data for 2 demo patients
        for patient_id in ["patient_001", "patient_002"]:
            
            # Gastric cancer progression parameters
            if patient_id == "patient_001":
                # Patient with clear progression
                trajectory_type = "sigmoid"
                n_timepoints = 4
                expression_noise = 0.15
            else:
                # Patient with more variable progression
                trajectory_type = "oscillatory"
                n_timepoints = 3
                expression_noise = 0.25
            
            dataset, metadata = generator.generate_synthetic_dataset(
                n_timepoints=n_timepoints,
                n_spots_per_time=400,  # Typical for gastric tissue
                n_genes=2000,
                trajectory_type=trajectory_type,
                grid_type="hexagonal",  # Tissue-like organization
                expression_noise=expression_noise,
                spatial_noise=0.1,
                batch_effects=False,
                trajectory_params={'steepness': 3.0, 'midpoint': 0.4}
            )
            
            datasets[patient_id] = dataset
        
        return datasets
    
    def run_patient_analysis(
        self,
        patient_id: str,
        run_baselines: bool = True,
        save_results: bool = False
    ) -> Dict:
        """
        Run complete analysis for a single patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        run_baselines : bool
            Whether to run baseline comparisons
        save_results : bool
            Whether to save results
            
        Returns:
        --------
        results : dict
            Complete analysis results for the patient
        """
        
        if patient_id not in self.datasets:
            raise ValueError(f"Patient {patient_id} not found in loaded datasets")
        
        print(f"\n=== Analyzing Patient: {patient_id} ===")
        
        dataset = self.datasets[patient_id]
        print(f"Dataset shape: {dataset.X_processed.shape}")
        print(f"Timepoints: {np.unique(dataset.time)}")
        print(f"Spots per timepoint: {[np.sum(dataset.time == t) for t in np.unique(dataset.time)]}")
        
        # Split data for training/testing
        # For longitudinal data, we can use early timepoints for training
        unique_times = np.unique(dataset.time)
        
        if len(unique_times) >= 3:
            # Use first timepoints for training, last for testing
            train_times = unique_times[:-1]
            test_times = unique_times[-1:]
        else:
            # Use random split if only 2 timepoints
            train_times = unique_times[:1]
            test_times = unique_times[1:]
        
        train_mask = np.isin(dataset.time, train_times)
        test_mask = np.isin(dataset.time, test_times)
        
        # Create training dataset
        train_dataset = SpatialTranscriptomicsDataset(
            X=dataset.X_processed[train_mask],
            coords=dataset.coords[train_mask],
            time=dataset.time[train_mask],
            config=self.config
        )
        
        # Train ST-Dynamics model
        print("\n--- Training ST-Dynamics Model ---")
        model = STDynamicsModel(self.config, encoder_type="mlp", use_decoder=True)
        
        try:
            training_history = model.fit(
                train_dataset,
                max_epochs=self.config.max_epochs,
                verbose=True
            )
            
            # Inference on full dataset
            print("\n--- Generating Embeddings and Latent Time ---")
            full_embeddings = model.infer_latent(dataset)
            
            # Infer latent time
            latent_time_inference = LatentTimeInference(self.config)
            inferred_time = latent_time_inference.infer_latent_time(
                full_embeddings,
                dataset.coords,
                method=self.config.latent_time_method
            )
            
            # Evaluate results
            evaluator = ComprehensiveEvaluation()
            st_dynamics_metrics = evaluator.evaluate_model(
                embeddings=full_embeddings,
                inferred_time=inferred_time,
                true_time=dataset.time,
                coords=dataset.coords
            )
            
            results = {
                'patient_id': patient_id,
                'dataset_info': {
                    'n_spots': len(dataset),
                    'n_timepoints': len(unique_times),
                    'timepoints': unique_times.tolist(),
                    'spatial_range': {
                        'x': (dataset.coords[:, 0].min(), dataset.coords[:, 0].max()),
                        'y': (dataset.coords[:, 1].min(), dataset.coords[:, 1].max())
                    }
                },
                'st_dynamics': {
                    'embeddings': full_embeddings,
                    'inferred_time': inferred_time,
                    'metrics': st_dynamics_metrics,
                    'training_history': training_history,
                    'model': model
                }
            }
            
            print(f"\nST-Dynamics Results:")
            print(f"  Kendall Tau: {st_dynamics_metrics['kendall_tau']:.4f}")
            print(f"  Spearman Rho: {st_dynamics_metrics['spearman_rho']:.4f}")
            print(f"  Overall Performance: {st_dynamics_metrics['overall_performance']:.4f}")
            
        except Exception as e:
            print(f"ST-Dynamics training failed: {e}")
            results = {
                'patient_id': patient_id,
                'st_dynamics': {'error': str(e)},
                'dataset_info': {'n_spots': len(dataset)}
            }
        
        # Run baseline comparisons
        if run_baselines:
            print("\n--- Baseline Comparisons ---")
            baseline_comparison = BaselineComparison(self.config)
            
            try:
                baseline_results = baseline_comparison.run_comparison(dataset)
                
                # Evaluate baselines
                evaluator = ComprehensiveEvaluation()
                for method_name, baseline_result in baseline_results.items():
                    if baseline_result['success'] and baseline_result['inferred_time'] is not None:
                        baseline_metrics = evaluator.evaluate_model(
                            embeddings=baseline_result.get('embedding'),
                            inferred_time=baseline_result['inferred_time'],
                            true_time=dataset.time,
                            coords=dataset.coords
                        )
                        baseline_result['metrics'] = baseline_metrics
                
                results['baselines'] = baseline_results
                
                # Create comparison table
                comparison_data = {}
                if 'st_dynamics' in results and 'metrics' in results['st_dynamics']:
                    comparison_data['ST_Dynamics'] = results['st_dynamics']['metrics']
                
                for method_name, baseline_result in baseline_results.items():
                    if baseline_result['success'] and 'metrics' in baseline_result:
                        comparison_data[method_name] = baseline_result['metrics']
                
                if comparison_data:
                    results['comparison_table'] = pd.DataFrame(comparison_data).T
                    
                    print("\n--- Method Comparison ---")
                    print(results['comparison_table'][['kendall_tau', 'spearman_rho', 'overall_performance']].round(4))
                
            except Exception as e:
                print(f"Baseline comparison failed: {e}")
                results['baselines'] = {'error': str(e)}
        
        # Save results if requested
        if save_results:
            save_path = Path(f"results_{patient_id}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"\nResults saved to: {save_path}")
        
        self.results[patient_id] = results
        
        return results
    
    def run_cohort_analysis(
        self,
        patient_ids: Optional[List[str]] = None,
        run_baselines: bool = True
    ) -> Dict:
        """
        Run analysis across multiple patients (cohort analysis).
        
        Parameters:
        -----------
        patient_ids : list, optional
            Specific patients to analyze (all if None)
        run_baselines : bool
            Whether to run baseline comparisons
            
        Returns:
        --------
        cohort_results : dict
            Aggregated results across the cohort
        """
        
        if patient_ids is None:
            patient_ids = list(self.datasets.keys())
        
        print(f"\n=== Cohort Analysis: {len(patient_ids)} Patients ===")
        
        cohort_results = {
            'patient_results': {},
            'cohort_summary': {}
        }
        
        # Run analysis for each patient
        for patient_id in patient_ids:
            print(f"\n{'='*50}")
            try:
                patient_results = self.run_patient_analysis(
                    patient_id, 
                    run_baselines=run_baselines,
                    save_results=False
                )
                cohort_results['patient_results'][patient_id] = patient_results
                
            except Exception as e:
                print(f"Failed to analyze patient {patient_id}: {e}")
                cohort_results['patient_results'][patient_id] = {'error': str(e)}
        
        # Aggregate cohort-level results
        print(f"\n{'='*50}")
        print("COHORT SUMMARY")
        print(f"{'='*50}")
        
        # Collect metrics across patients
        st_dynamics_metrics = []
        baseline_metrics = {}
        
        for patient_id, patient_results in cohort_results['patient_results'].items():
            if 'st_dynamics' in patient_results and 'metrics' in patient_results['st_dynamics']:
                st_dynamics_metrics.append(patient_results['st_dynamics']['metrics'])
            
            if 'baselines' in patient_results:
                for method_name, baseline_result in patient_results['baselines'].items():
                    if baseline_result.get('success') and 'metrics' in baseline_result:
                        if method_name not in baseline_metrics:
                            baseline_metrics[method_name] = []
                        baseline_metrics[method_name].append(baseline_result['metrics'])
        
        # Compute cohort-level statistics
        cohort_summary = {}
        
        # ST-Dynamics cohort performance
        if st_dynamics_metrics:
            st_dynamics_summary = {}
            for metric_name in st_dynamics_metrics[0].keys():
                values = [m[metric_name] for m in st_dynamics_metrics if metric_name in m]
                if values:
                    st_dynamics_summary[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            cohort_summary['ST_Dynamics'] = st_dynamics_summary
            
            print(f"\nST-Dynamics Cohort Performance (n={len(st_dynamics_metrics)}):")
            print(f"  Overall Performance: {st_dynamics_summary['overall_performance']['mean']:.4f} ± {st_dynamics_summary['overall_performance']['std']:.4f}")
            print(f"  Kendall Tau: {st_dynamics_summary['kendall_tau']['mean']:.4f} ± {st_dynamics_summary['kendall_tau']['std']:.4f}")
            print(f"  Spearman Rho: {st_dynamics_summary['spearman_rho']['mean']:.4f} ± {st_dynamics_summary['spearman_rho']['std']:.4f}")
        
        # Baseline cohort performance
        for method_name, method_metrics in baseline_metrics.items():
            if method_metrics:
                method_summary = {}
                for metric_name in method_metrics[0].keys():
                    values = [m[metric_name] for m in method_metrics if metric_name in m]
                    if values:
                        method_summary[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                cohort_summary[method_name] = method_summary
                
                print(f"\n{method_name} Cohort Performance (n={len(method_metrics)}):")
                if 'overall_performance' in method_summary:
                    print(f"  Overall Performance: {method_summary['overall_performance']['mean']:.4f} ± {method_summary['overall_performance']['std']:.4f}")
        
        cohort_results['cohort_summary'] = cohort_summary
        
        # Create cohort comparison table
        comparison_data = {}
        for method_name, method_summary in cohort_summary.items():
            method_means = {f"{metric}_mean": values['mean'] 
                          for metric, values in method_summary.items()
                          if isinstance(values, dict) and 'mean' in values}
            comparison_data[method_name] = method_means
        
        if comparison_data:
            cohort_comparison_df = pd.DataFrame(comparison_data).T
            cohort_results['cohort_comparison_table'] = cohort_comparison_df
            
            print(f"\n{'='*50}")
            print("COHORT METHOD COMPARISON")
            print(f"{'='*50}")
            
            key_metrics = ['overall_performance_mean', 'kendall_tau_mean', 'spearman_rho_mean']
            available_metrics = [m for m in key_metrics if m in cohort_comparison_df.columns]
            
            if available_metrics:
                print(cohort_comparison_df[available_metrics].round(4))
        
        self.results['cohort'] = cohort_results
        
        return cohort_results
    
    def generate_cohort_report(
        self,
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Generate comprehensive cohort analysis report.
        
        Parameters:
        -----------
        output_dir : str or Path, optional
            Directory to save report files
        """
        
        if 'cohort' not in self.results:
            print("No cohort results available. Run cohort analysis first.")
            return
        
        if output_dir is None:
            output_dir = Path("gastric_cohort_report")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating cohort report in: {output_dir}")
        
        cohort_results = self.results['cohort']
        
        # Save summary tables
        if 'cohort_comparison_table' in cohort_results:
            cohort_results['cohort_comparison_table'].to_csv(output_dir / "cohort_method_comparison.csv")
        
        # Save individual patient results
        for patient_id, patient_results in cohort_results['patient_results'].items():
            if 'comparison_table' in patient_results:
                patient_results['comparison_table'].to_csv(output_dir / f"{patient_id}_comparison.csv")
        
        # Generate summary report
        report_text = self._generate_text_report(cohort_results)
        with open(output_dir / "cohort_analysis_report.txt", 'w') as f:
            f.write(report_text)
        
        print(f"Report generated successfully in {output_dir}")
    
    def _generate_text_report(self, cohort_results: Dict) -> str:
        """Generate text summary report."""
        
        report = []
        report.append("GASTRIC LONGITUDINAL COHORT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {pd.Timestamp.now()}")
        report.append("")
        
        # Cohort overview
        n_patients = len(cohort_results['patient_results'])
        successful_patients = sum(1 for r in cohort_results['patient_results'].values() 
                                 if 'error' not in r)
        
        report.append(f"COHORT OVERVIEW:")
        report.append(f"  Total Patients: {n_patients}")
        report.append(f"  Successful Analyses: {successful_patients}")
        report.append("")
        
        # Method performance summary
        if 'cohort_summary' in cohort_results:
            report.append("METHOD PERFORMANCE SUMMARY:")
            
            for method_name, method_summary in cohort_results['cohort_summary'].items():
                report.append(f"\n  {method_name}:")
                
                key_metrics = ['overall_performance', 'kendall_tau', 'spearman_rho']
                for metric in key_metrics:
                    if metric in method_summary:
                        stats = method_summary[metric]
                        report.append(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                                    f"[{stats['min']:.4f}, {stats['max']:.4f}]")
            
            report.append("")
        
        # Individual patient summary
        report.append("INDIVIDUAL PATIENT RESULTS:")
        
        for patient_id, patient_results in cohort_results['patient_results'].items():
            report.append(f"\n  {patient_id}:")
            
            if 'error' in patient_results:
                report.append(f"    Status: Failed ({patient_results['error']})")
            else:
                if 'st_dynamics' in patient_results and 'metrics' in patient_results['st_dynamics']:
                    metrics = patient_results['st_dynamics']['metrics']
                    report.append(f"    Overall Performance: {metrics['overall_performance']:.4f}")
                    report.append(f"    Kendall Tau: {metrics['kendall_tau']:.4f}")
                
                if 'dataset_info' in patient_results:
                    info = patient_results['dataset_info']
                    report.append(f"    Spots: {info['n_spots']}, Timepoints: {info['n_timepoints']}")
        
        return "\n".join(report)


def run_gastric_longitudinal_analysis(
    data_path: Optional[Union[str, Path]] = None,
    patient_ids: Optional[List[str]] = None,
    config: Optional[Config] = None,
    generate_report: bool = True
) -> Dict:
    """
    Run complete gastric longitudinal analysis.
    
    Parameters:
    -----------
    data_path : str or Path, optional
        Path to gastric data (uses demo data if None)
    patient_ids : list, optional
        Specific patients to analyze
    config : Config, optional
        Configuration object
    generate_report : bool
        Whether to generate analysis report
        
    Returns:
    --------
    results : dict
        Complete analysis results
    """
    
    print("=" * 60)
    print("GASTRIC LONGITUDINAL ANALYSIS")
    print("=" * 60)
    
    config = config or Config()
    experiment = GastricLongitudinalExperiment(config)
    
    # Load data
    if data_path is not None:
        datasets = experiment.load_gastric_data(data_path, patient_ids)
    else:
        print("No data path provided, using synthetic demo data...")
        datasets = experiment._create_demo_gastric_data()
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return {'error': 'No data loaded'}
    
    # Run cohort analysis
    cohort_results = experiment.run_cohort_analysis(
        patient_ids=list(datasets.keys()),
        run_baselines=True
    )
    
    # Generate report
    if generate_report:
        experiment.generate_cohort_report()
    
    print("\n" + "=" * 60)
    print("GASTRIC LONGITUDINAL ANALYSIS COMPLETED")
    print("=" * 60)
    
    return cohort_results


if __name__ == "__main__":
    # Run demo analysis
    results = run_gastric_longitudinal_analysis()
    
    print("\nDemo analysis completed successfully!")
    if 'cohort_comparison_table' in results:
        print("\nCohort method comparison:")
        print(results['cohort_comparison_table'].round(4))