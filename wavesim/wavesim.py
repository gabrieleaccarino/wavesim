from typing import Dict, Tuple, Union
import xarray as xr
import numpy as np
import torch

from wavesim.wavelet_base import Wavelet2DBaseTorch
from wavesim.metrics import Metric

class WaveSim(Wavelet2DBaseTorch, Metric):
    """
    A wavelet-based metric for comparing spatial fields using multiple components:
    TODO
    1. displacement similarity (via Wasserstein distance), 
    2. centroid shift, 
    3. and magnitude differences.
    """
    def __init__(
        self,
        map1: Union[np.ndarray, xr.DataArray, torch.Tensor],
        map2: Union[np.ndarray, xr.DataArray, torch.Tensor],
        params: Dict = None,
    ):
        """
        Initialize WaveSimSVFinal with two spatial maps and metric parameters.

        Params
        ------
        map1 : array-like
            First input map.
        map2 : array-like
            Second input map.
        params : dict
            Configuration dictionary including:
                - wavelet: str
                - mode: str
                - levels: int
                - operation: str
                - components_weight: dict with 'alpha', 'beta', 'gamma'
                - scales_weight: list or 1D tensor
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.map1, self.map2 = self._process_input(map1, map2)

        assert self.map1.shape == self.map2.shape
        self.B, self.C, self.H, self.W = self.map1.shape #[-2:]

        # wavelet params
        self.wavelet = params['wavelet']
        self.mode = params['mode']
        self.levels = params['levels']
        self.operation = params['operation']
        self.components_weight = params['components_weight']
        self.alpha = self.components_weight['alpha']
        self.beta = self.components_weight['beta']
        self.gamma = self.components_weight['gamma']
        scales_weight = params['scales_weight']
        self.scales_weight_tensor = torch.Tensor(scales_weight).view(1, 1, -1) if isinstance(scales_weight, (list, np.ndarray)) else scales_weight

        self.map1_dwt = Wavelet2DBaseTorch(data=self.map1, wavelet=self.wavelet, mode=self.mode, levels=self.levels)
        self.map2_dwt = Wavelet2DBaseTorch(data=self.map2, wavelet=self.wavelet, mode=self.mode, levels=self.levels)

    
    def _process_input(
        self,
        map1: Union[np.ndarray, xr.DataArray, torch.Tensor],
        map2: Union[np.ndarray, xr.DataArray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts inputs to proper shape (1, 1, H, W) tensors."""
        def convert(x):
            if isinstance(x, xr.DataArray):
                x = x.values
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            if isinstance(x, torch.Tensor):
                if x.ndim == 2:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.ndim == 3:
                    x = x.unsqueeze(0)
                elif x.ndim != 4:
                    raise ValueError(f"Unsupported tensor shape: {x.shape}")
                return x.to(self.device)
            raise ValueError(f"Unsupported input type: {type(x)}")
        
        return convert(map1), convert(map2)

    def _center_of_mass(self, marginal: torch.Tensor) -> torch.Tensor:
        N, L = marginal.shape[-2:]
        coords = torch.arange(N, device=marginal.device, dtype=marginal.dtype)
        coords = coords.view(1, 1, N, 1).expand(-1, -1, -1, L)
        return torch.sum(marginal * coords, dim=2)

    def _torch_wasserstein_loss(self, P: torch.Tensor, Q: torch.Tensor, p: int = 1) -> torch.Tensor:
        return self.__torch_cdf_loss(P, Q, p=p)

    def __torch_cdf_loss(self, P: torch.Tensor, Q: torch.Tensor, p: int = 1) -> torch.Tensor:
        P = P / (P.sum(dim=2, keepdim=True) + 1e-14)
        Q = Q / (Q.sum(dim=2, keepdim=True) + 1e-14)
        cdf_P = torch.cumsum(P, dim=2)
        cdf_Q = torch.cumsum(Q, dim=2)
        diff = cdf_P - cdf_Q
        if p == 1:
            return torch.sum(torch.abs(diff), dim=2)
        elif p == 2:
            return torch.sqrt(torch.sum(diff ** 2, dim=2))
        else:
            return torch.sum(torch.abs(diff) ** p, dim=2) ** (1 / p)

    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        return (p * torch.log2(p / q)).sum(dim=-2)
    
    def _marginals(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_energy = data.sum(dim=(-3, -2), keepdim=True)
        lat_marginal = data.sum(dim=-2) / (total_energy.squeeze(-2) + 1e-14)
        lon_marginal = data.sum(dim=-3) / (total_energy.squeeze(-3) + 1e-14)
        return lat_marginal, lon_marginal
    
    def compute(self) -> float:
        """Compute the final similarity score."""
        self.m1_coeffs = self.map1_dwt.scale_inversion(operation=self.operation)          # (B, C, H, W, L+1)
        self.m2_coeffs = self.map2_dwt.scale_inversion(operation=self.operation)          # (B, C, H, W, L+1)

        m1_energy = self.m1_coeffs ** 2                                                   # (B, C, H, W, L+1)
        m2_energy = self.m2_coeffs ** 2                                                   # (B, C, H, W, L+1)

        self.m1_marginal_lat, self.m1_marginal_lon = self._marginals(m1_energy)           # (B, C, H, L+1), (B, C, W, L+1)
        self.m2_marginal_lat, self.m2_marginal_lon = self._marginals(m2_energy)           # (B, C, H, L+1), (B, C, W, L+1)
        #self.m1_marginal_lat, self.m1_marginal_lon = self._marginals(self.m1_coeffs)           # (B, C, H, L+1), (B, C, W, L+1)
        #self.m2_marginal_lat, self.m2_marginal_lon = self._marginals(self.m2_coeffs)           # (B, C, H, L+1), (B, C, W, L+1)

        # Component 1: Displacement Similarity Score (DSS)
        mixture_lat = 0.5 * (self.m1_marginal_lat + self.m2_marginal_lat)                 # (B, C, H, L+1)
        mixture_lon = 0.5 * (self.m1_marginal_lon + self.m2_marginal_lon)                 # (B, C, W, L+1)
        kl_map1_lat = self._kl_divergence(self.m1_marginal_lat, mixture_lat)              # (B, C, L+1)
        kl_map2_lat = self._kl_divergence(self.m2_marginal_lat, mixture_lat)              # (B, C, L+1)
        kl_map1_lon = self._kl_divergence(self.m1_marginal_lon, mixture_lon)              # (B, C, L+1)
        kl_map2_lon = self._kl_divergence(self.m2_marginal_lon, mixture_lon)              # (B, C, L+1)
        jsd_lat = 0.5 * (kl_map1_lat + kl_map2_lat)                                       # (B, C, L+1)
        jsd_lon = 0.5 * (kl_map1_lon + kl_map2_lon)                                       # (B, C, L+1)
        self.displacement_similarity_score = (1 - jsd_lat) * (1 - jsd_lon)
        self.displacement_similarity_score_weighted = torch.pow(self.displacement_similarity_score, self.alpha)

        # Component 2: Normalized Magnitude Difference Score (NMDS)
        """
        self.m1_mean_energy = torch.mean(m1_energy, dim=(2,3))                                                               # (B, C, L+1)
        self.m2_mean_energy = torch.mean(m2_energy, dim=(2,3))                                                               # (B, C, L+1)
        self.max_energy = torch.maximum(self.m1_mean_energy, self.m2_mean_energy)                                            # (B, C, L+1)
        magnitude_difference = torch.abs(self.m1_mean_energy - self.m2_mean_energy)/(self.max_energy + 1e-14)                # (B, C, L+1)
        self.magnitude_difference_score = 1 - magnitude_difference                                                           # (B, C, L+1)
        self.magnitude_difference_score_weighted = torch.pow(self.magnitude_difference_score, self.gamma)                    # (B, C, L+1)
        """
        # Component 2 (NEW): Symmetric Relative Difference (SRD)
        # THIS IS THE VALUE REPORTED IN THE REPORT June 03, 2025
        self.m1_mean_energy = torch.mean(m1_energy, dim=(2,3))                                                               # (B, C, L+1)
        self.m2_mean_energy = torch.mean(m2_energy, dim=(2,3))                                                               # (B, C, L+1)
        sum_energy = self.m1_mean_energy + self.m2_mean_energy
        magnitude_difference = torch.abs(self.m1_mean_energy - self.m2_mean_energy)
        relative_difference = magnitude_difference / (sum_energy + 1e-14)
        self.magnitude_difference_score = 1 - relative_difference                                                            # (B, C, L+1)
        self.magnitude_difference_score_weighted = torch.pow(self.magnitude_difference_score, self.gamma)                    # (B, C, L+1)

        #self.m1_mean_coeff = torch.mean(self.m1_coeffs, dim=(2,3))                                                               # (B, C, L+1)
        #self.m2_mean_coeff = torch.mean(self.m2_coeffs, dim=(2,3))
        #relative_difference = 2 * self.m1_mean_coeff * self.m2_mean_coeff / (self.m1_mean_coeff ** 2 + self.m2_mean_coeff **2 + 1e-14)
        #self.magnitude_difference_score = 1 - relative_difference
        #self.magnitude_difference_score_weighted = torch.pow(self.magnitude_difference_score, self.gamma) 

        # Component 2 (NEW SIGNED): Symmetric Relative Difference (SRD)
        # TODO : this is SENSITIVE TO THE SIGN BUT THE MEAN VALUES ARE TOO LOW
        #self.m1_mean_coeff = torch.mean(torch.abs(self.m1_coeffs), dim=(2,3))                                                               # (B, C, L+1)
        #self.m2_mean_coeff = torch.mean(torch.abs(self.m2_coeffs), dim=(2,3))                                                               # (B, C, L+1)
        ##self.m1_mean_coeff = torch.mean(self.m1_coeffs, dim=(2,3))                                                               # (B, C, L+1)
        ##self.m2_mean_coeff = torch.mean(self.m2_coeffs, dim=(2,3))  
        ##self.numerator = torch.abs(self.m1_mean_coeff - self.m2_mean_coeff)
        ##self.denominator = torch.abs(self.m1_mean_coeff) + torch.abs(self.m2_mean_coeff) + 1e-6
        ###self.denominator = torch.sqrt(torch.square(self.m1_mean_coeff) + torch.square(self.m2_mean_coeff) + 1e-6)
        ##relative_signed_diff = self.numerator / self.denominator
        ##self.magnitude_difference_score = 1 - relative_signed_diff                                                      # (B, C, L+1)
        ##self.magnitude_difference_score_weighted = torch.pow(self.magnitude_difference_score, self.gamma)               # (B, C, L+1)

        # Component 3: 
        """
        def l1_similarity_sorted(x: torch.Tensor, y: torch.Tensor, dim: int = 2) -> torch.Tensor:
            x_sorted = torch.sort(x, dim=dim)[0]
            y_sorted = torch.sort(y, dim=dim)[0]
            dist = (x_sorted - y_sorted).abs().sum(dim=dim)
            sim = 1 - 0.5 * dist    
            #sim = 1 / (1 + dist)  # normalize to (0, 1)
            return sim
        
        def l2_similarity_sorted(x: torch.Tensor, y: torch.Tensor, dim: int = 2) -> torch.Tensor:
            x_sorted = torch.sort(x, dim=dim)[0]
            y_sorted = torch.sort(y, dim=dim)[0]
            dist = (x_sorted - y_sorted).pow(2).sum(dim=dim).sqrt()
            #sim = 1 / (1 + dist)  # normalize to (0, 1)
            sim = 1 - 0.5 * dist
            return sim
        """
        def structural_similarity_nsc(coeffs1: torch.Tensor, coeffs2: torch.Tensor, dim=2) -> torch.Tensor:
            """
            Compute structural similarity via normalized sorted covariance (NSC).
            
            coeffs1, coeffs2: (B, C, H, W, S)
            Returns: (B, C, S)
            """
            B, C, H, W, S = coeffs1.shape
            x = coeffs1.view(B, C, H*W, S)  # (B, C, H*W, S)
            y = coeffs2.view(B, C, H*W, S)  # (B, C, H*W, S)
            x_sorted, _ = torch.sort(x, dim=2)
            y_sorted, _ = torch.sort(y, dim=2)

            # Step 2: Normalize to remove magnitude dependence
            x_centered = x_sorted - x_sorted.mean(dim=2, keepdim=True)
            y_centered = y_sorted - y_sorted.mean(dim=2, keepdim=True)
            x_norm = x_centered / (x_centered.norm(dim=2, keepdim=True) + 1e-8)
            y_norm = y_centered / (y_centered.norm(dim=2, keepdim=True) + 1e-8)

            # Similarity via normalized covariance
            sim = (x_norm * y_norm).sum(dim=2)
            sim = 0.5 * (sim + 1.0)                                                 # [0, 1]

            # Magnitude variation penalty
            mean_gap = (x_sorted - y_sorted).abs().mean(dim=2)  # [0, 2]
            penalty = 1.0 - mean_gap / (x_sorted.abs().mean(dim=2) + y_sorted.abs().mean(dim=2) + 1e-8)

            return sim * penalty  # shape: (B, C, S)
    
        #cs_lat = l1_similarity_sorted(self.m1_marginal_lat, self.m2_marginal_lat, dim=2)
        #cs_lon = l1_similarity_sorted(self.m1_marginal_lon, self.m2_marginal_lon, dim=2)
        #self.structural_similarity_score = cs_lat * cs_lon  # (B, C, L+1)
        #self.structural_similarity_score = 0.5 * (cs_lat + cs_lon)  # (B, C, L+1)
        
        self.structural_similarity_score = structural_similarity_nsc(self.m1_coeffs, self.m2_coeffs)
        self.structural_similarity_score_weighted = torch.pow(self.structural_similarity_score, self.beta)

        self.scale_score = self.magnitude_difference_score_weighted * self.displacement_similarity_score_weighted * self.structural_similarity_score_weighted       # (B, C, L+1)

        self.score = torch.sum(self.scales_weight_tensor * self.scale_score, dim=-1).item()                                  # (B, C)

        return self.score