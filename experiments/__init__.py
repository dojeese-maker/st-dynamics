"""
Experimental validation modules for ST-Dynamics.

This package provides:
- Synthetic data experiments for validation and benchmarking
- Gastric longitudinal data analysis for real-world validation
- Comprehensive experimental frameworks
"""

from .synthetic import (
    SyntheticDataGenerator,
    SyntheticExperiment,
    run_synthetic_validation_suite
)
from .gastric_longitudinal import (
    GastricLongitudinalExperiment,
    run_gastric_longitudinal_analysis
)

__all__ = [
    "SyntheticDataGenerator",
    "SyntheticExperiment", 
    "run_synthetic_validation_suite",
    "GastricLongitudinalExperiment",
    "run_gastric_longitudinal_analysis"
]