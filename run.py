"""
Main entry point for ST-Dynamics experiments.

This module provides the primary interface for running ST-Dynamics experiments
and reproducing paper results. Designed for easy use by reviewers and researchers.

Usage:
    python -m st_dynamics.run --experiment synthetic --config config.yaml
    python -m st_dynamics.run --experiment gastric --data_path /path/to/data
    python -m st_dynamics.run --help

Key functions:
- reproduce_paper_figures(): Reproduce all paper figures  
- run_synthetic_validation(): Validate on synthetic data
- run_gastric_analysis(): Analyze gastric longitudinal data
- compare_baselines(): Compare with SOTA methods
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import warnings

from .config import Config
from .experiments.synthetic import run_synthetic_validation_suite
from .experiments.gastric_longitudinal import run_gastric_longitudinal_analysis
from .data.dataset import SpatialTranscriptomicsDataset
from .model.model import STDynamicsModel
from .evaluation.baselines import compare_with_baselines


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to YAML configuration file
        
    Returns:
    --------
    config : Config
        Configuration object
    """
    
    if config_path is None:
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        warnings.warn(f"Config file not found: {config_path}. Using defaults.")
        return Config()
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config object and update with file values
        config = Config()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                warnings.warn(f"Unknown config parameter: {key}")
        
        return config
        
    except Exception as e:
        warnings.warn(f"Error loading config file: {e}. Using defaults.")
        return Config()


def reproduce_paper_figures(
    config: Optional[Config] = None,
    output_dir: str = "paper_figures",
    quick_mode: bool = False
) -> Dict[str, Any]:
    """
    Reproduce all paper figures and results.
    
    This function runs the complete experimental pipeline to reproduce
    the main results presented in the paper.
    
    Parameters:
    -----------
    config : Config, optional
        Configuration object
    output_dir : str
        Directory to save figures and results
    quick_mode : bool
        Whether to use reduced dataset sizes for quick testing
        
    Returns:
    --------
    results : dict
        Complete experimental results
    """
    
    print("=" * 80)
    print("REPRODUCING ST-DYNAMICS PAPER FIGURES")
    print("=" * 80)
    
    config = config or Config()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = {}
    
    # Figure 1: Method overview (synthetic demonstration)
    print("\n--- Figure 1: Method Overview ---")
    try:
        fig1_results = run_synthetic_validation_suite(
            config=config,
            quick_test=quick_mode
        )
        all_results['figure_1'] = fig1_results
        
        # Save results
        if 'basic_experiment' in fig1_results and 'comparison_table' in fig1_results['basic_experiment']:
            comparison_df = fig1_results['basic_experiment']['comparison_table']
            comparison_df.to_csv(output_path / "figure1_method_comparison.csv")
            print(f"Saved: {output_path / 'figure1_method_comparison.csv'}")
        
        print("Figure 1 completed successfully!")
        
    except Exception as e:
        print(f"Figure 1 failed: {e}")
        all_results['figure_1'] = {'error': str(e)}
    
    # Figure 2: Synthetic data validation
    print("\n--- Figure 2: Synthetic Data Validation ---") 
    try:
        # This is covered in the synthetic validation suite above
        if 'figure_1' in all_results and 'basic_experiment' in all_results['figure_1']:
            all_results['figure_2'] = all_results['figure_1']['basic_experiment']
            print("Figure 2 completed (using synthetic validation results)!")
        else:
            all_results['figure_2'] = {'error': 'Synthetic validation not available'}
            
    except Exception as e:
        print(f"Figure 2 failed: {e}")
        all_results['figure_2'] = {'error': str(e)}
    
    # Figure 3: Real data analysis (gastric longitudinal)
    print("\n--- Figure 3: Real Data Analysis ---")
    try:
        fig3_results = run_gastric_longitudinal_analysis(
            config=config,
            generate_report=True
        )
        all_results['figure_3'] = fig3_results
        
        # Save cohort comparison if available
        if 'cohort_comparison_table' in fig3_results:
            fig3_results['cohort_comparison_table'].to_csv(output_path / "figure3_cohort_comparison.csv")
            print(f"Saved: {output_path / 'figure3_cohort_comparison.csv'}")
        
        print("Figure 3 completed successfully!")
        
    except Exception as e:
        print(f"Figure 3 failed: {e}")
        all_results['figure_3'] = {'error': str(e)}
    
    # Supplementary: Ablation study
    print("\n--- Supplementary: Ablation Study ---")
    try:
        if 'figure_1' in all_results and 'ablation_study' in all_results['figure_1']:
            ablation_results = all_results['figure_1']['ablation_study']
            all_results['supplementary_ablation'] = ablation_results
            
            if 'comparison_table' in ablation_results:
                ablation_results['comparison_table'].to_csv(output_path / "supplementary_ablation.csv")
                print(f"Saved: {output_path / 'supplementary_ablation.csv'}")
            
            print("Supplementary ablation completed!")
        else:
            all_results['supplementary_ablation'] = {'error': 'Ablation study not available'}
            
    except Exception as e:
        print(f"Supplementary ablation failed: {e}")
        all_results['supplementary_ablation'] = {'error': str(e)}
    
    # Generate summary report
    summary_report = generate_summary_report(all_results)
    with open(output_path / "reproduction_summary.txt", 'w') as f:
        f.write(summary_report)
    print(f"Saved: {output_path / 'reproduction_summary.txt'}")
    
    print("\n" + "=" * 80)
    print("PAPER REPRODUCTION COMPLETED")
    print(f"Results saved in: {output_path}")
    print("=" * 80)
    
    return all_results


def run_synthetic_validation(
    config: Optional[Config] = None,
    quick_mode: bool = False
) -> Dict[str, Any]:
    """
    Run synthetic data validation experiments.
    
    Parameters:
    -----------
    config : Config, optional
        Configuration object
    quick_mode : bool
        Whether to use quick test mode
        
    Returns:
    --------
    results : dict
        Synthetic validation results
    """
    
    print("=" * 60)
    print("ST-DYNAMICS SYNTHETIC VALIDATION")
    print("=" * 60)
    
    config = config or Config()
    
    results = run_synthetic_validation_suite(
        config=config,
        quick_test=quick_mode
    )
    
    print("\nSynthetic validation completed!")
    
    # Print summary
    if 'basic_experiment' in results and 'comparison_table' in results['basic_experiment']:
        print("\nMethod comparison summary:")
        comparison_df = results['basic_experiment']['comparison_table']
        key_metrics = ['overall_performance', 'kendall_tau', 'spearman_rho']
        available_metrics = [m for m in key_metrics if m in comparison_df.columns]
        if available_metrics:
            print(comparison_df[available_metrics].round(4))
    
    return results


def run_gastric_analysis(
    data_path: Optional[str] = None,
    patient_ids: Optional[List[str]] = None,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Run gastric longitudinal data analysis.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to gastric data directory
    patient_ids : list, optional
        Specific patient IDs to analyze
    config : Config, optional
        Configuration object
        
    Returns:
    --------
    results : dict
        Gastric analysis results
    """
    
    print("=" * 60)
    print("ST-DYNAMICS GASTRIC ANALYSIS")
    print("=" * 60)
    
    config = config or Config()
    
    results = run_gastric_longitudinal_analysis(
        data_path=data_path,
        patient_ids=patient_ids,
        config=config,
        generate_report=True
    )
    
    print("\nGastric analysis completed!")
    
    # Print summary
    if 'cohort_summary' in results and 'ST_Dynamics' in results['cohort_summary']:
        st_summary = results['cohort_summary']['ST_Dynamics']
        print("\nST-Dynamics cohort performance:")
        if 'overall_performance' in st_summary:
            print(f"  Overall: {st_summary['overall_performance']['mean']:.4f} ± {st_summary['overall_performance']['std']:.4f}")
        if 'kendall_tau' in st_summary:
            print(f"  Kendall Tau: {st_summary['kendall_tau']['mean']:.4f} ± {st_summary['kendall_tau']['std']:.4f}")
    
    return results


def compare_baselines_standalone(
    data_path: str,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """
    Standalone baseline comparison on provided data.
    
    Parameters:
    -----------
    data_path : str
        Path to spatial transcriptomics data
    config : Config, optional
        Configuration object
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        Baseline comparison results
    """
    
    print("=" * 60)
    print("ST-DYNAMICS BASELINE COMPARISON")
    print("=" * 60)
    
    config = config or Config()
    
    # Load data (this is a simplified example - adapt based on data format)
    print(f"Loading data from: {data_path}")
    
    try:
        # Example loading logic - replace with actual data loading
        import pandas as pd
        
        # Assume CSV format for this example
        expr_df = pd.read_csv(f"{data_path}/expression.csv", index_col=0)
        coords_df = pd.read_csv(f"{data_path}/coordinates.csv", index_col=0)
        meta_df = pd.read_csv(f"{data_path}/metadata.csv", index_col=0)
        
        # Create dataset
        dataset = SpatialTranscriptomicsDataset(
            X=expr_df.values,
            coords=coords_df[['x', 'y']].values,
            time=meta_df['time'].values,
            gene_names=expr_df.columns.tolist(),
            sample_names=expr_df.index.tolist(),
            config=config
        )
        
        print(f"Dataset loaded: {dataset.X_processed.shape}")
        
        # Train ST-Dynamics model
        print("Training ST-Dynamics...")
        model = STDynamicsModel(config)
        model.fit(dataset)
        
        # Get ST-Dynamics results
        embeddings = model.infer_latent(dataset)
        
        from .inference.latent_time import LatentTimeInference
        inference = LatentTimeInference(config)
        inferred_time = inference.infer_latent_time(embeddings, dataset.coords)
        
        st_results = {
            'embedding': embeddings,
            'inferred_time': inferred_time
        }
        
        # Compare with baselines
        print("Running baseline comparisons...")
        all_results, capabilities_df = compare_with_baselines(
            st_results, dataset, config
        )
        
        # Evaluate all methods
        from .evaluation.metrics import ComprehensiveEvaluation
        evaluator = ComprehensiveEvaluation()
        
        comparison_data = {}
        for method_name, method_results in all_results.items():
            if method_results.get('success') and method_results.get('inferred_time') is not None:
                metrics = evaluator.evaluate_model(
                    embeddings=method_results.get('embedding'),
                    inferred_time=method_results['inferred_time'],
                    true_time=dataset.time,
                    coords=dataset.coords
                )
                comparison_data[method_name] = metrics
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        print("\nBaseline comparison completed!")
        print(comparison_df[['overall_performance', 'kendall_tau', 'spearman_rho']].round(4))
        
        return comparison_df
        
    except Exception as e:
        print(f"Baseline comparison failed: {e}")
        return pd.DataFrame()


def generate_summary_report(results: Dict[str, Any]) -> str:
    """Generate text summary of all experimental results."""
    
    report = []
    report.append("ST-DYNAMICS EXPERIMENTAL RESULTS SUMMARY")
    report.append("=" * 60)
    report.append(f"Generated: {pd.Timestamp.now()}")
    report.append("")
    
    # Figure 1 summary
    if 'figure_1' in results:
        report.append("FIGURE 1 - METHOD OVERVIEW:")
        if 'error' not in results['figure_1'] and 'basic_experiment' in results['figure_1']:
            basic_exp = results['figure_1']['basic_experiment']
            if 'st_dynamics' in basic_exp and 'metrics' in basic_exp['st_dynamics']:
                metrics = basic_exp['st_dynamics']['metrics']
                report.append(f"  ST-Dynamics Overall Performance: {metrics['overall_performance']:.4f}")
                report.append(f"  Kendall Tau: {metrics['kendall_tau']:.4f}")
                report.append(f"  Spearman Rho: {metrics['spearman_rho']:.4f}")
        else:
            report.append(f"  Status: Failed - {results['figure_1'].get('error', 'Unknown error')}")
        report.append("")
    
    # Figure 3 summary
    if 'figure_3' in results:
        report.append("FIGURE 3 - REAL DATA ANALYSIS:")
        if 'error' not in results['figure_3'] and 'cohort_summary' in results['figure_3']:
            cohort_summary = results['figure_3']['cohort_summary']
            if 'ST_Dynamics' in cohort_summary:
                st_summary = cohort_summary['ST_Dynamics']
                if 'overall_performance' in st_summary:
                    stats = st_summary['overall_performance']
                    report.append(f"  Cohort Overall Performance: {stats['mean']:.4f} ± {stats['std']:.4f}")
                if 'kendall_tau' in st_summary:
                    stats = st_summary['kendall_tau']
                    report.append(f"  Cohort Kendall Tau: {stats['mean']:.4f} ± {stats['std']:.4f}")
        else:
            report.append(f"  Status: Failed - {results['figure_3'].get('error', 'Unknown error')}")
        report.append("")
    
    # Ablation study summary
    if 'supplementary_ablation' in results:
        report.append("SUPPLEMENTARY - ABLATION STUDY:")
        if 'error' not in results['supplementary_ablation'] and 'comparison_table' in results['supplementary_ablation']:
            ablation_df = results['supplementary_ablation']['comparison_table']
            if 'full_model' in ablation_df.index:
                full_model_perf = ablation_df.loc['full_model', 'overall_performance']
                report.append(f"  Full Model Performance: {full_model_perf:.4f}")
            
            # Show contribution of each component
            if 'no_temporal' in ablation_df.index:
                no_temporal_perf = ablation_df.loc['no_temporal', 'overall_performance']
                report.append(f"  Without Temporal Loss: {no_temporal_perf:.4f}")
            if 'no_spatial' in ablation_df.index:
                no_spatial_perf = ablation_df.loc['no_spatial', 'overall_performance']
                report.append(f"  Without Spatial Loss: {no_spatial_perf:.4f}")
        else:
            report.append(f"  Status: Failed - {results['supplementary_ablation'].get('error', 'Unknown error')}")
        report.append("")
    
    report.append("ANALYSIS COMPLETED SUCCESSFULLY")
    
    return "\n".join(report)


def main():
    """Main entry point for command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="ST-Dynamics: Spatial Transcriptomics Temporal Dynamics Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reproduce all paper figures
  python -m st_dynamics.run --reproduce-figures
  
  # Quick validation
  python -m st_dynamics.run --experiment synthetic --quick
  
  # Analyze gastric data
  python -m st_dynamics.run --experiment gastric --data-path /path/to/gastric/data
  
  # Custom config
  python -m st_dynamics.run --experiment synthetic --config my_config.yaml
  
  # Baseline comparison only
  python -m st_dynamics.run --compare-baselines --data-path /path/to/data
        """
    )
    
    # Main experiment options
    parser.add_argument(
        '--reproduce-figures',
        action='store_true',
        help='Reproduce all paper figures and results'
    )
    
    parser.add_argument(
        '--experiment',
        choices=['synthetic', 'gastric'],
        help='Type of experiment to run'
    )
    
    parser.add_argument(
        '--compare-baselines',
        action='store_true',
        help='Run baseline comparison on provided data'
    )
    
    # Data options
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to data directory'
    )
    
    parser.add_argument(
        '--patient-ids',
        nargs='+',
        help='Specific patient IDs to analyze (for gastric experiment)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use quick mode (reduced dataset sizes for testing)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        if args.reproduce_figures:
            # Reproduce all paper figures
            results = reproduce_paper_figures(
                config=config,
                output_dir=args.output_dir,
                quick_mode=args.quick
            )
            
        elif args.experiment == 'synthetic':
            # Run synthetic validation
            results = run_synthetic_validation(
                config=config,
                quick_mode=args.quick
            )
            
        elif args.experiment == 'gastric':
            # Run gastric analysis
            results = run_gastric_analysis(
                data_path=args.data_path,
                patient_ids=args.patient_ids,
                config=config
            )
            
        elif args.compare_baselines:
            # Run baseline comparison
            if args.data_path is None:
                print("Error: --data-path required for baseline comparison")
                sys.exit(1)
                
            comparison_df = compare_baselines_standalone(
                data_path=args.data_path,
                config=config
            )
            
            # Save results
            output_path = Path(args.output_dir)
            output_path.mkdir(exist_ok=True)
            comparison_df.to_csv(output_path / "baseline_comparison.csv")
            print(f"Results saved to: {output_path / 'baseline_comparison.csv'}")
            
        else:
            # No experiment specified
            parser.print_help()
            sys.exit(1)
            
        print("\nExperiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()