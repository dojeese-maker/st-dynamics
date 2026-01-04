"""
Evaluation and baseline comparison modules for ST-Dynamics.

This package provides:
- Comprehensive evaluation metrics for spatial-temporal dynamics
- Baseline method implementations and comparisons
- Performance assessment tools
"""

from .metrics import (
    ComprehensiveEvaluation,
    TemporalMonotonicityMetrics,
    SpatialSmoothnessMetrics,
    ReconstructionMetrics,
    evaluate_latent_time_inference
)
from .baselines import (
    BaselineComparison,
    DiffusionPseudotimeBaseline,
    OptimalTransportBaseline,
    PASTEBaseline,
    DestOTBaseline,
    DimensionalityReductionBaseline,
    compare_with_baselines
)

__all__ = [
    "ComprehensiveEvaluation",
    "TemporalMonotonicityMetrics", 
    "SpatialSmoothnessMetrics",
    "ReconstructionMetrics",
    "evaluate_latent_time_inference",
    "BaselineComparison",
    "DiffusionPseudotimeBaseline",
    "OptimalTransportBaseline",
    "PASTEBaseline",
    "DestOTBaseline", 
    "DimensionalityReductionBaseline",
    "compare_with_baselines"
]