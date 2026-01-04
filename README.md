# ST-Dynamics: Spatial Transcriptomics Temporal Dynamics Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A modular research codebase for analyzing temporal dynamics in spatial transcriptomics data, designed for **Nature Machine Intelligence** standard research.

## 🚀 Key Features

- **Strict Training/Inference Separation**: No direct time supervision during representation learning
- **Post-hoc Temporal Inference**: Complete separation of representation learning and temporal dynamics
- **Comprehensive Baseline Comparison**: Built-in implementations of DPT, OT, PASTE, DeST-OT methods
- **Spatial-Temporal Constraints**: Advanced loss functions for spatial smoothness and temporal consistency
- **Reproducible Research Framework**: Complete experimental validation suite
- **Gastric Longitudinal Validation**: Real-world validation on gastric cancer progression data

## 🏗️ Architecture Overview

```
st_dynamics/
│
├── data/
│   ├── dataset.py          # Standard spatial transcriptomics data interface
│   └── preprocessing.py    # Comprehensive preprocessing pipeline
│
├── model/
│   ├── encoder.py          # Neural encoders (MLP, VAE, Attention)
│   ├── losses.py           # Spatial-temporal loss functions
│   └── model.py            # Main ST-Dynamics model
│
├── inference/
│   └── latent_time.py      # Post-hoc latent time inference
│
├── evaluation/
│   ├── metrics.py          # Comprehensive evaluation metrics
│   └── baselines.py        # Baseline method implementations
│
├── experiments/
│   ├── synthetic.py        # Synthetic data validation
│   └── gastric_longitudinal.py  # Real data experiments
│
├── config.py               # Centralized configuration
└── run.py                  # Main experimental interface
```

## 📦 Installation

### Requirements

```bash
pip install torch>=1.8.0
pip install numpy pandas scikit-learn scipy
pip install scanpy>=1.9.0  # For preprocessing and baselines
pip install anndata>=0.8.0  # For data handling
pip install matplotlib seaborn  # For visualization
pip install tqdm PyYAML  # For utilities
```

### Optional Dependencies

```bash
# For enhanced dimensionality reduction
pip install umap-learn>=0.5.0

# For advanced optimal transport
pip install ot>=0.8.0  

# For additional spatial analysis
pip install squidpy>=1.2.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/st-dynamics.git
cd st-dynamics

# Install in development mode
pip install -e .
```

## 🔥 Quick Start

### Basic Usage

```python
import numpy as np
from st_dynamics import STDynamicsModel, Config
from st_dynamics.data import SpatialTranscriptomicsDataset

# Load your spatial transcriptomics data
X = np.random.randn(1000, 2000)  # (spots, genes)
coords = np.random.randn(1000, 2)  # (spots, 2)
time = np.repeat([0, 1, 2, 3], 250)  # Time labels

# Create dataset
dataset = SpatialTranscriptomicsDataset(X, coords, time)

# Configure and train model
config = Config()
model = STDynamicsModel(config)

# Train (NO time supervision)
model.fit(dataset)

# Infer representations
embeddings = model.infer_latent(dataset)

# Post-hoc time inference
from st_dynamics.inference import LatentTimeInference
inference = LatentTimeInference(config)
latent_time = inference.infer_latent_time(embeddings, coords)

print(f"Inferred latent time range: [{latent_time.min():.3f}, {latent_time.max():.3f}]")
```

### Reproduce Paper Figures

```bash
# Reproduce all paper results
python -m st_dynamics.run --reproduce-figures

# Quick validation (reduced datasets)
python -m st_dynamics.run --reproduce-figures --quick

# Specific experiments
python -m st_dynamics.run --experiment synthetic
python -m st_dynamics.run --experiment gastric --data-path /path/to/gastric/data
```

### Baseline Comparison

```python
from st_dynamics.evaluation.baselines import BaselineComparison

# Initialize comparison framework
comparison = BaselineComparison(config)

# Run all baselines
baseline_results = comparison.run_comparison(dataset)

# Filter methods by capability
temporal_methods = comparison.filter_by_capability(produces_time=True)
print(f"Methods that infer temporal progression: {temporal_methods}")

# Get capability summary
capabilities_df = comparison.get_capability_summary()
print(capabilities_df)
```

### Comprehensive Evaluation

```python
from st_dynamics.evaluation.metrics import ComprehensiveEvaluation

# Initialize evaluator
evaluator = ComprehensiveEvaluation()

# Evaluate ST-Dynamics results
metrics = evaluator.evaluate_model(
    embeddings=embeddings,
    inferred_time=latent_time,
    true_time=time,
    coords=coords
)

print(f"Temporal Performance: {metrics['temporal_performance']:.4f}")
print(f"Spatial Performance: {metrics['spatial_performance']:.4f}")
print(f"Overall Performance: {metrics['overall_performance']:.4f}")
```

## 🧪 Experimental Validation

### Synthetic Data Experiments

```python
from st_dynamics.experiments.synthetic import run_synthetic_validation_suite

# Run complete synthetic validation
results = run_synthetic_validation_suite(
    config=config,
    quick_test=False  # Use full datasets
)

# Print method comparison
if 'comparison_table' in results['basic_experiment']:
    print(results['basic_experiment']['comparison_table'])
```

### Gastric Longitudinal Analysis

```python
from st_dynamics.experiments.gastric_longitudinal import run_gastric_longitudinal_analysis

# Analyze gastric cancer progression
results = run_gastric_longitudinal_analysis(
    data_path="/path/to/gastric/data",
    patient_ids=["patient_001", "patient_002"],  # Optional
    generate_report=True
)

# View cohort summary
if 'cohort_summary' in results:
    print("ST-Dynamics Cohort Performance:")
    st_summary = results['cohort_summary']['ST_Dynamics']
    print(f"  Overall: {st_summary['overall_performance']['mean']:.4f} ± {st_summary['overall_performance']['std']:.4f}")
```

## ⚙️ Configuration

### Custom Configuration

```python
from st_dynamics import Config

# Create custom configuration
config = Config(
    # Model architecture
    latent_dim=128,
    hidden_dims=(512, 256, 128),
    
    # Loss weights
    lambda_recon=1.0,
    lambda_temporal=0.5,
    lambda_spatial=0.3,
    
    # Training
    learning_rate=1e-3,
    max_epochs=200,
    batch_size=256,
    
    # Inference
    latent_time_method="diffusion_map",
    n_diffusion_components=10
)
```

### YAML Configuration

```yaml
# config.yaml
model:
  latent_dim: 64
  hidden_dims: [512, 256, 128]
  dropout_rate: 0.1

loss:
  lambda_recon: 1.0
  lambda_temporal: 0.5
  lambda_spatial: 0.3

training:
  learning_rate: 0.001
  max_epochs: 200
  batch_size: 256

inference:
  latent_time_method: "diffusion_map"
  n_diffusion_components: 10
```

```bash
python -m st_dynamics.run --experiment synthetic --config config.yaml
```

## 📊 Evaluation Metrics

### Temporal Monotonicity
- **Kendall Tau**: Rank correlation with true time
- **Spearman Rho**: Monotonic correlation 
- **Temporal Ordering Accuracy**: Pairwise ordering accuracy
- **Temporal Coherence**: Within-timepoint consistency

### Spatial Smoothness
- **Spatial Autocorrelation**: Moran's I statistic
- **Neighbor Preservation**: Spatial neighbor conservation
- **Local Spatial Variance**: Spatial smoothness measure

### Reconstruction Quality
- **MSE**: Mean squared error
- **Gene Correlation**: Feature preservation
- **Expression Range**: Dynamic range preservation

## 🔄 Baseline Methods

### Implemented Baselines

1. **Diffusion Pseudotime (DPT)** - Gold standard from single-cell analysis
2. **Optimal Transport** - Sinkhorn algorithm for temporal alignment
3. **PASTE** - Spatial transcriptomics alignment
4. **DeST-OT** - Spatiotemporal optimal transport (simplified)
5. **PCA/UMAP** - Standard dimensionality reduction + temporal ordering

### Baseline Capabilities

| Method | Produces Embedding | Infers Time | Uses Spatial Info | Uses Temporal Info |
|--------|-------------------|-------------|-------------------|-------------------|
| ST-Dynamics | ✅ | ✅ | ✅ | ❌ (no supervision) |
| DPT | ✅ | ✅ | ❌ | ❌ |
| Optimal Transport | ❌ | ✅ | ❌ | ✅ |
| PASTE | ❌ | ✅ | ✅ | ✅ |
| DeST-OT | ✅ | ✅ | ✅ | ✅ |
| PCA | ✅ | ✅ | ❌ | ❌ |

## 🔬 Research Principles

### Core Algorithmic Principles

1. **No Direct Time Supervision**: Time labels used ONLY for consistency constraints, never as direct supervision
2. **Strict Train/Inference Separation**: [`model.fit()`](st_dynamics/model/model.py:185) for training, [`model.infer_latent()`](st_dynamics/model/model.py:365) for inference
3. **Post-hoc Temporal Inference**: Latent time inferred completely separately from representation learning
4. **Spatial-Temporal Constraints**: Physics-inspired loss functions for realistic dynamics

### Loss Function Design

```python
# Temporal consistency (NOT supervision)
def temporal_consistency_loss(z_t, z_t_plus_1):
    # Encourages smooth transitions between consecutive timepoints
    # WITHOUT using time as supervision signal
    return contrastive_loss(z_t, z_t_plus_1)

# Spatial smoothness 
def spatial_smoothness_loss(z, coords):
    # Enforces spatial organization using ONLY coordinates
    # NO biological labels or time information
    return laplacian_smoothness(z, build_spatial_graph(coords))
```

### Evaluation Philosophy

- **Ground Truth Comparison**: Against known temporal progression
- **Baseline Fairness**: Standardized interfaces for fair comparison  
- **Methodological Rigor**: Statistical significance testing
- **Reproducibility**: Complete experimental specifications

## 📈 Expected Performance

### Synthetic Data
- **Kendall Tau**: 0.85-0.95 (high temporal correlation)
- **Spatial Performance**: 0.75-0.90 (good spatial preservation)
- **Overall Performance**: 0.80-0.92 (strong combined performance)

### Real Data (Gastric Longitudinal)
- **Kendall Tau**: 0.60-0.80 (realistic biological noise)
- **Cohort Consistency**: Low std across patients
- **Baseline Outperformance**: Consistent improvement over existing methods

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size or model size
config.batch_size = 128
config.latent_dim = 32
```

**2. Poor Temporal Performance** 
```python
# Increase temporal loss weight
config.lambda_temporal = 1.0
```

**3. Missing Dependencies**
```bash
# Install optional packages
pip install scanpy anndata umap-learn
```

**4. Data Format Issues**
```python
# Ensure proper data types
X = X.astype(np.float32)
coords = coords.astype(np.float32) 
time = time.astype(int)
```

## 📚 Citation

```bibtex
@article{st_dynamics_2024,
  title={ST-Dynamics: Inferring Spatial-Temporal Dynamics in Spatial Transcriptomics},
  author={Your Name and Collaborators},
  journal={Nature Machine Intelligence},
  year={2024},
  doi={10.1038/s41551-024-xxxxx}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-repo/st-dynamics.git
cd st-dynamics
pip install -e ".[dev]"
```

### Running Tests

```bash
# Quick validation
python -m st_dynamics.run --reproduce-figures --quick

# Full test suite
python -m st_dynamics.run --reproduce-figures
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scanpy team for single-cell analysis tools
- PyTorch team for deep learning framework
- Spatial transcriptomics community for datasets and methods

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/st-dynamics/issues)
- **Documentation**: [Wiki](https://github.com/your-repo/st-dynamics/wiki)
- **Email**: your.email@institution.edu

---

**ST-Dynamics** - Bridging spatial organization and temporal dynamics in transcriptomics data analysis.