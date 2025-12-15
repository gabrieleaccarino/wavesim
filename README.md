# WaveSim: Wavelet-based Similarity Metric

WaveSim is a Python library for computing perceptually-aware similarity metrics between 2D spatial fields using wavelet transformations. It provides a robust framework for comparing spatial patterns across multiple scales, particularly useful for climate data, weather forecasts, and Earth system model evaluation.

## Overview

WaveSim decomposes spatial fields using discrete wavelet transforms (DWT) and evaluates similarity through three complementary components:

- **Magnitude Component**: Compares energy distributions across wavelet scales
- **Displacement Component**: Measures spatial displacement using Jensen-Shannon divergence of marginal distributions
- **Structural Component**: Assesses pattern similarity independent of position and magnitude

The metric is particularly effective at detecting differences that are perceptually important but may be missed by traditional point-wise metrics like RMSE or correlation.

## Key Features

- **Multi-scale Analysis**: Decomposes fields into different spatial scales using wavelets (db4, haar, etc.)
- **Perceptually Motivated**: Three independent components capture different aspects of similarity
- **Flexible Weighting**: Customizable weights for both components (α, β, γ) and scales
- **PyTorch Backend**: GPU-accelerated computations using PyTorch
- **Multiple Input Formats**: Supports NumPy arrays, PyTorch tensors, and xarray DataArrays
- **Additional Metrics**: Includes DSSIM and NRMSE implementations
- **Visualization Tools**: Utilities for plotting and analyzing results
- **Perturbation Tools**: Functions for synthetic case generation and testing

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- xarray
- PyWavelets (pywt)
- pytorch_wavelets

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/gabrieleaccarino/wavesim.git
cd wavesim

# Install dependencies (recommended: use a virtual environment)
pip install torch numpy xarray pywt pytorch_wavelets matplotlib cartopy pandas
```

### Environment Setup

For working with the full examples and notebooks:

```bash
# Additional dependencies for notebooks
pip install jupyter zarr gcsfs scipy
```

## Quickstart

### Basic Usage

```python
import numpy as np
import torch
from wavesim import WaveSim

# Prepare your data (shape: B, C, H, W)
map1 = np.random.rand(1, 1, 256, 256)  # Reference field
map2 = np.random.rand(1, 1, 256, 256)  # Comparison field

# Define parameters
params = {
    'wavelet': 'db4',           # Wavelet family (db4, haar, etc.)
    'mode': 'zero',             # Padding mode
    'levels': 3,                # Number of decomposition levels
    'operation': 'sum',         # Combine orientations: 'sum' or 'mean'
    'components_weight': {
        'alpha': 1.0,           # Magnitude component weight
        'beta': 1.0,            # Displacement component weight
        'gamma': 1.0            # Structural component weight
    },
    'scales_weight': [1/3, 1/3, 1/3],  # Weight for each scale
    'use_approx': False         # Include approximation coefficients
}

# Compute similarity
ws = WaveSim(map1, map2, params)
similarity_score = ws.compute()

print(f"WaveSim Score: {similarity_score:.4f}")

# Access individual components
print(f"Magnitude: {ws.magnitude_sim_score}")
print(f"Displacement: {ws.displacement_sim_score}")
print(f"Structural: {ws.structural_sim_score}")
```

### Using with xarray DataArrays

```python
import xarray as xr

# Load data from NetCDF or Zarr
data1 = xr.open_dataarray('field1.nc')
data2 = xr.open_dataarray('field2.nc')

# WaveSim automatically handles xarray inputs
ws = WaveSim(data1, data2, params)
score = ws.compute()
```

### Using Other Metrics

```python
from wavesim.metrics import DataStructuralSimilarityIndex, NormalizedRootMeanSquaredError

# DSSIM
dssim_params = {'C1': 1e-8, 'C2': 1e-8, 'kernel_size': 11, 'sigma': 1.5}
dssim = DataStructuralSimilarityIndex(map1, map2, dssim_params)
dssim_score = dssim.compute()

# NRMSE
nrmse = NormalizedRootMeanSquaredError(map1, map2, {})
nrmse_score = nrmse.compute()
```

## Repository Structure

```
wavesim/
├── README.md                   # This file
├── .gitignore                  # Git ignore patterns
│
├── wavesim/                    # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── wavesim.py             # WaveSim main class
│   ├── wavelet_base.py        # Wavelet transform base class
│   ├── wavesim.ipynb          # Development notebook
│   │
│   ├── metrics/               # Metric implementations
│   │   ├── __init__.py
│   │   ├── metric_base.py     # Base metric class
│   │   ├── DSSIM.py           # Data Structural Similarity Index
│   │   └── NRMSE.py           # Normalized RMSE
│   │
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── plot.py            # Plotting utilities
│   │   └── perturbations.py   # Data perturbation tools
│   │
│   └── case_studies/          # Example applications
│       ├── synthetic_cases.ipynb          # Synthetic test cases
│       ├── enso_composites.ipynb          # ENSO pattern analysis
│       ├── pna_bias.ipynb                 # PNA pattern bias evaluation
│       ├── test_displacement.ipynb        # Displacement component tests
│       ├── synthetic_cases_scores.csv     # Results from synthetic tests
│       └── similarity_table.md            # ESM comparison results
│
├── notebooks/                  # Data processing notebooks
│   ├── gathering/             # Data retrieval
│   │   └── retrieve_cmip6_monthly.ipynb
│   └── remapping/             # Regridding examples
│       ├── pr_remapping.ipynb
│       ├── tos_remapping.ipynb
│       └── zg_remapping.ipynb
│
└── figures/                    # Generated figures
    ├── fig1.png
    ├── fig2.png
    └── figS1.png
```

## Environment Details

### Core Dependencies

- **PyTorch**: Backend for tensor operations and GPU acceleration
- **NumPy**: Numerical array operations
- **xarray**: Labeled multi-dimensional arrays (particularly for climate/weather data)
- **PyWavelets (pywt)**: Wavelet transform definitions
- **pytorch_wavelets**: PyTorch-compatible wavelet transforms

### Optional Dependencies

- **matplotlib**: Visualization and plotting
- **cartopy**: Geospatial plotting for Earth science data
- **pandas**: Data analysis and CSV handling
- **jupyter**: Interactive notebooks
- **zarr**: Chunked array storage (for large datasets)
- **gcsfs**: Google Cloud Storage access (for WeatherBench2 data)

### Tested Python Versions

- Python 3.8+
- PyTorch 1.10+

## Examples and Use Cases

### Case Studies

The `wavesim/case_studies/` directory contains comprehensive examples:

1. **Synthetic Cases** (`synthetic_cases.ipynb`): Demonstrates WaveSim's behavior on controlled perturbations (noise, shifts, blurring, etc.)

2. **ENSO Composites** (`enso_composites.ipynb`): Evaluates climate model performance in representing El Niño patterns

3. **PNA Bias** (`pna_bias.ipynb`): Assesses Pacific-North American pattern biases in Earth System Models

4. **Displacement Tests** (`test_displacement.ipynb`): Detailed analysis of the displacement component

### Perturbation Utilities

```python
from wavesim.utils.perturbations import noise_at_level, crop_centered_window

# Add noise at a specific wavelet level
perturbed_field = noise_at_level(
    wavelet=wavelet_obj,
    magnitude=0.5,
    level=2,
    orientation='H'  # Horizontal, 'V' for vertical, 'D' for diagonal
)

# Extract a spatial window
window = crop_centered_window(
    field=data,
    center_lon=180,
    center_lat=45,
    size_lon=256,
    size_lat=256
)
```

### Visualization

```python
from wavesim.utils.plot import SHIFTED_BRBG_COLORS
import matplotlib.pyplot as plt

# Custom colormaps and plotting utilities available
# See wavesim/utils/plot.py for more details
```

## Parameter Guidelines

### Wavelet Selection

- **'db4'** (Daubechies 4): Good general-purpose choice, smooth features
- **'haar'**: Simple, edge detection
- **'sym4'** (Symlet 4): Nearly symmetric, good for natural patterns

### Number of Levels

- Depends on spatial resolution and features of interest
- Maximum levels determined by field size: `pywt.dwt_max_level(min(H, W), wavelet)`
- Typical range: 3-5 levels for 256×256 fields

### Component Weights (α, β, γ)

- Equal weights (1.0, 1.0, 1.0): Balanced evaluation
- Adjust based on application priorities:
  - High α: Emphasize energy/magnitude matching
  - High β: Emphasize spatial alignment
  - High γ: Emphasize structural patterns

### Scale Weights

- Uniform: `[1/L, 1/L, ..., 1/L]` for L scales
- Custom: Emphasize specific scales (e.g., `[0.5, 0.3, 0.2]` to prioritize fine scales)

## Citation

If you use WaveSim in your research, please cite the associated paper (add citation details when available).

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments

This work builds upon wavelet transform theory and structural similarity metrics developed by the computer vision and signal processing communities, adapted for Earth system model evaluation.