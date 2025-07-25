from typing import Dict, Tuple, Union
import xarray as xr
import numpy as np
import torch

from wavesim.wavelet_base import Wavelet2DBaseTorch
from wavesim.metrics import Metric

class WaveSim(Wavelet2DBaseTorch, Metric):
    """
    WaveSim class that inherits from Wavelet2DBaseTorch and Metric.
    This class is designed to handle wavelet transformations and metric calculations.
    """

    def __init__(self, 
                 map1: Union[torch.Tensor, xr.DataArray, np.ndarray], 
                 map2: Union[torch.Tensor, xr.DataArray, np.ndarray],
                 params: Dict):
        """
        Initialize the WaveSim class with two maps, parameters, wavelet object, mode, and levels.
        """
        # Initialize Metric (with both maps)
        Metric.__init__(self, map1, map2, params)
        
        self.B, self.C, self.H, self.W = self.map1.shape
        
        self.eps = 1e-8  # Small value to avoid division by zero
        
        # wavelet params
        self.wavelet = params['wavelet']
        self.mode = params['mode']
        self.levels = params['levels']
        self.operation = params['operation']
        self.components_weight = params['components_weight']
        self.alpha = self.components_weight['alpha']    # (magnitude component weight)
        self.beta = self.components_weight['beta']      # (displacement component weight)
        self.gamma = self.components_weight['gamma']    # (structural component weight)
        scales_weight = params['scales_weight']
        self.scales_weight_tensor = torch.Tensor(scales_weight).view(self.B, self.C, self.levels + 1) if isinstance(scales_weight, (list, np.ndarray)) else scales_weight

        # Initialize Wavelet2DBaseTorch with the maps
        # compute DWT for both maps according to the wavelet, mode, and levels
        self.map1_dwt = Wavelet2DBaseTorch(data=self.map1, wavelet=self.wavelet, mode=self.mode, levels=self.levels)
        self.map2_dwt = Wavelet2DBaseTorch(data=self.map2, wavelet=self.wavelet, mode=self.mode, levels=self.levels)

    def _magnitude_component(self, m1_energy: torch.Tensor, m2_energy: torch.Tensor) -> torch.Tensor:
        """
        Compute the magnitude component of WaveSim.
        """
        self.m1_mean_energy = torch.mean(m1_energy, dim=(2, 3))                     # (B, C, L+1)
        self.m2_mean_energy = torch.mean(m2_energy, dim=(2, 3))                     # (B, C, L+1)
        sum_energy = self.m1_mean_energy + self.m2_mean_energy                      # (B, C, L+1)   
        magnitude_difference = torch.abs(self.m1_mean_energy - self.m2_mean_energy) # (B, C, L+1)
        relative_difference = magnitude_difference / (sum_energy + self.eps)           # (B, C, L+1)
        return 1 - relative_difference                                              # (B, C, L+1)
    
    def _displacement_component(self, m1_energy: torch.Tensor, m2_energy: torch.Tensor) -> torch.Tensor: 
        """
        Compute the displacement component of WaveSim.
        """
        
        def marginals(energy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute the normalized marginals of the energy tensor across latitudes and longitudes.
            """
            total_energy = energy.sum(dim=(-3, -2), keepdim=True)
            lat_marginal = energy.sum(dim=-2) / (total_energy.squeeze(-2) + self.eps)
            lon_marginal = energy.sum(dim=-3) / (total_energy.squeeze(-3) + self.eps)
            return lat_marginal, lon_marginal
        
        def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            p = torch.clamp(p, min=eps)
            q = torch.clamp(q, min=eps)
            return (p * torch.log2(p / q)).sum(dim=-2)
        
        self.m1_marginal_lat, self.m1_marginal_lon = marginals(m1_energy)           # (B, C, H, L+1), (B, C, W, L+1)
        self.m2_marginal_lat, self.m2_marginal_lon = marginals(m2_energy)           # (B, C, H, L+1), (B, C, W, L+1)
        
        mixture_lat = 0.5 * (self.m1_marginal_lat + self.m2_marginal_lat)           # (B, C, H, L+1)
        mixture_lon = 0.5 * (self.m1_marginal_lon + self.m2_marginal_lon)           # (B, C, W, L+1)
        
        # Compute the Kullback-Leibler divergence for both latitudes and longitudes against the mixture marginals
        # KL(p || m) = sum(p * log(p / m))
        # where p is the marginal distribution of the energy (lat or lon), and m is the mixture marginal
        kl_map1_lat = kl_divergence(self.m1_marginal_lat, mixture_lat)              # (B, C, L+1)
        kl_map2_lat = kl_divergence(self.m2_marginal_lat, mixture_lat)              # (B, C, L+1)
        kl_map1_lon = kl_divergence(self.m1_marginal_lon, mixture_lon)              # (B, C, L+1)
        kl_map2_lon = kl_divergence(self.m2_marginal_lon, mixture_lon)              # (B, C, L+1)
        
        # Compute the Jensen-Shannon divergence for both latitudes and longitudes
        # JSD = 0.5 * (KL(p || m) + KL(q || m))
        # where p and q are the marginals of the two maps, and m is the mixture marginal
        # JSD is symmetric and bounded between 0 and 1
        jsd_lat = 0.5 * (kl_map1_lat + kl_map2_lat)                                 # (B, C, L+1)
        jsd_lon = 0.5 * (kl_map1_lon + kl_map2_lon)                                 # (B, C, L+1)
        return (1 - jsd_lat) * (1 - jsd_lon)
 
    def _structural_component(self, coeffs1: torch.Tensor, coeffs2: torch.Tensor) -> torch.Tensor: 
        """
        Compute the structural component of WaveSim.
        """
        B, C, H, W, S = coeffs1.shape       # S is L+1, the number of scales
        
        # Reshape to (B, C, spatial_dims, S) for processing
        # treat spatial locations as a flat vector of features for each scale S
        c1 = coeffs1.view(B, C, H*W, S)                                             # (B, C, H*W, S)
        c2 = coeffs2.view(B, C, H*W, S)                                             # (B, C, H*W, S)
        
        # Sort along spatial dimension to capture structural patterns
        # This removes spatial positional information, retaining only sorted structural patterns (e.g. edges, textures). 
        # This makes the metric invariant to translations and permutations across the spatial domain.
        c1_sorted, _ = torch.sort(c1, dim=2)                                        # (B, C, H*W, S)
        c2_sorted, _ = torch.sort(c2, dim=2)                                        # (B, C, H*W, S)
        
        # Center the sorted coefficients 
        # Removes global mean and make the component contrast-invariant
        # This ensures we're measuring structural patterns, not absolute levels
        c1_centered = c1_sorted - c1_sorted.mean(dim=2, keepdim=True)               # (B, C, H*W, S)
        c2_centered = c2_sorted - c2_sorted.mean(dim=2, keepdim=True)               # (B, C, H*W, S)     
        
        # L2 normalization to make similarity magnitude-invariant
        # ||x||_2 = sqrt(sum(x^2))
        c1_norm = c1_centered / (c1_centered.norm(dim=2, keepdim=True) + self.eps)  # (B, C, 1, S)
        c2_norm = c2_centered / (c2_centered.norm(dim=2, keepdim=True) + self.eps)  # (B, C, 1, S)

        # Compute normalized dot product (cosine similarity)
        # sim in [-1, 1], then map to [0, 1]
        cosine_sim = (c1_norm * c2_norm).sum(dim=2)                                 # (B, C, S)
        cosine_sim = 0.5 * (cosine_sim + 1.0)  # Map [-1,1] to [0,1]                # (B, C, S)
        
        # Magnitude variation penalty
        #mean_gap = (c1_sorted - c2_sorted).abs().mean(dim=2)  # [0, 2]
        #penalty = 1.0 - mean_gap / (c1_sorted.abs().mean(dim=2) + c2_sorted.abs().mean(dim=2) + self.eps)
        return cosine_sim #* penalty                                                 # (B, C, S)
    
    def compute(self) -> float:
        """
        """
        self.m1_coeffs = self.map1_dwt.scale_inversion(operation=self.operation)    # (B, C, H, W, L+1)
        self.m2_coeffs = self.map2_dwt.scale_inversion(operation=self.operation)    # (B, C, H, W, L+1)
        
        m1_energy = self.m1_coeffs ** 2                                             # (B, C, H, W, L+1)
        m2_energy = self.m2_coeffs ** 2                                             # (B, C, H, W, L+1)       
        
        self.magnitude_sim_score = self._magnitude_component(m1_energy, m2_energy)
        self.magnitude_sim_score_weighted = torch.pow(self.magnitude_sim_score, self.alpha)
       
        self.displacement_sim_score = self._displacement_component(m1_energy, m2_energy)
        self.displacement_sim_score_weighted = torch.pow(self.displacement_sim_score, self.beta)
        
        self.structural_sim_score = self._structural_component(self.m1_coeffs, self.m2_coeffs)
        self.structural_sim_score_weighted = torch.pow(self.structural_sim_score, self.gamma)
        
        self.scale_score = self.magnitude_sim_score_weighted * self.displacement_sim_score_weighted * self.structural_sim_score_weighted        # (B, C, L+1)
        self.score = torch.sum(self.scales_weight_tensor * self.scale_score, dim=-1).item()                                                     # (B, C)

        return self.score