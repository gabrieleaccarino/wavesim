# WaveSim: A Wavelet-based Multi-scale Similarity Metric for Weather and Climate Fields
*Gabriele Accarino*, *Viviana Acquaviva*, *Sara Shamekh*, *Duncan Watson-Parris*, *David Lawrence*

WaveSim is a multi-scale similarity metric for the evaluation of spatial fields in weather and climate applications. WaveSim exploits wavelet transforms to decompose input fields into scale-specific wavelet coefficients. The metric is built by combining three components of similarity which are derived from these coefficients and capture orthogonal information. Each component yields a scale-specific similarity score ranging from 0 (no similarity) to 1 (perfect similarity), which are then multiplied across scales to produce an overall similarity measure. WaveSim comes with component and scale weights to allow users modify the behavior of the metric according to specific needs and case studies. Moreover, the selection of the mother wavelet to use, as well as the number of scales can be tailored according to the type of fields being compared. Overall, it provides a robust framework for comparing spatial patterns across multiple scales, particularly useful for climate data, weather forecasts, and Earth system model evaluation.

<!--
## Table of Contents
- Overview 
-->

> [NOTE]
>
> - I’m actively improving this page and adding useful information to help you get comfortable using WaveSim. Stay tuned for updates 🙂
> - The pre-print is coming soon!

If you find our work useful, please consider to ⭐ star this repository 

<!--and 📝 cite our paper: -->

## Overview 
WaveSim decomposes spatial fields using Discrete Wavelet Transforms (DWT) and evaluates similarity through three orthogonal components:

- Magnitude ($\mathcal{M}$), which quantifies similarities in the energy distribution of the coefficients, i.e., the intensity of the field; 
- Displacement ($\mathcal{D}$), which captures spatial shift by comparing the centers of mass of normalized energy distributions;
- Structure ($\mathcal{S}$), which assesses pattern organization independent of location and amplitude.

The overall WaveSim score combines these three components and aggregates across scales with tunable weights:

$$WaveSim(X,Y) = \sum_{s=1}^{S} w^{s} \cdot \Big(\mathcal{M}(\tilde{X}^{s}, \tilde{Y}^{s})^{\alpha} \cdot \mathcal{D}(\tilde{X}^{s}, \tilde{Y}^{s})^{\beta} \cdot \mathcal{S}(\tilde{X}^{s}, \tilde{Y}^{s})^{\gamma} \Big)$$

where $\tilde{X}^{s}$, $\tilde{Y}^{s}$ are maps of wavelet coefficients at scale $s$ obtained by the scale separation technique described in the paper; $\alpha, \beta, \gamma \in [0,1]$ control the relative importance of each component (with $\alpha=\beta=\gamma=1$ corresponding to equal importance), and $S$ in the total number of scales of decomposition (user-defined). Parameters $w^s$ are scale-dependent weights normalized such that $\sum_{s=1}^{S} w^{s} = 1$, which can be adjusted to emphasize specific scales depending on the application.

<!--
## Installation

### Prerequisites

- Python 3.8+
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
-->


## Quickstart

### Basic Usage

```python
from wavesim import WaveSim
import numpy as np

# Prepare your data (shape: B, C, H, W)
map1 = np.random.rand(1, 1, 256, 256)  # Reference field
map2 = np.random.rand(1, 1, 256, 256)  # Perturbed field

# Define parameters
params = {
    'wavelet': 'db4',           # Wavelet family (db4, haar, etc.)
    'mode': 'zero',             # Padding mode (zero, periodic, etc.)
    'levels': 3,                # Number of scales
    'operation': 'sum',         # Combine orientations: 'sum' or 'mean'
    'components_weight': {
        'alpha' : 1.0,          # Magnitude component weight
        'beta'  : 1.0,          # Displacement component weight
        'gamma' : 1.0           # Structural component weight
    },
    'scales_weight': [1/3, 1/3, 1/3],   # Weight for each scale, equal weighting here, they should sum up to 1.0
    'use_approx': False                 # Whether to include approximation coefficients
}

# Compute similarity
wavesim = WaveSim(map1, map2, params)
wavesim_score = wavesim.compute()

print(f"WaveSim Score: {similarity_score:.4f}")

# Access individual components
print(f"Magnitude       : {wavesim.magnitude_sim_score.numpy().ravel()}")
print(f"Displacement    : {wavesim.displacement_sim_score.numpy().ravel()}")
print(f"Structural      : {wavesim.structural_sim_score.numpy().ravel()}")
```

<!-- 
## Repository Structure

```
wavesim/
├── README.md                  # This file
│
├── wavesim/                   # Main package directory
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
│   └── case_studies/                       # Example applications
│       ├── synthetic_cases.ipynb           # Synthetic test cases
│       ├── enso_composites.ipynb           # ENSO pattern analysis
│       ├── pna_bias.ipynb                  # PNA pattern bias evaluation
│       ├── test_displacement.ipynb         # Displacement component tests
│       ├── synthetic_cases_scores.csv      # Results from synthetic tests
│       └── similarity_table.md             # ESM comparison results
│
├── notebooks/                              # Data processing notebooks
│   ├── gathering/                          # Data retrieval
│   │   └── retrieve_cmip6_monthly.ipynb
│   └── remapping/                          # Regridding examples
│       ├── pr_remapping.ipynb
│       ├── tos_remapping.ipynb
│       └── zg_remapping.ipynb
│
└── figures/                                # Generated figures
    ├── fig1.png
    ├── fig2.png
    └── figS1.png
```
--> 

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please contact me or submit a pull request.

## Contact

Feel free to reach out!! ☺️ 
You can contact me at: <ga2673@columbia.edu>

## Acknowledgments

We acknowledge funding from NSF through the Learning the Earth with Artificial intelligence and Physics (LEAP) Science and Technology Center (STC) (Award \#2019625). VA acknowledges support from a PSC-CUNY Cycle 55 grant (Award \#67628), a PIVOT fellowship grant (Award \#981849), and a PIVOT Research award (Award \#12871) from the Simons Foundation. 

## Citation

TBA