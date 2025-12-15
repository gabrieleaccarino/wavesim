import torch.nn.functional as F
from typing import Dict, Tuple
import torch

from .metric_base import Metric

def create_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Creates a 2D Gaussian kernel."""
    ax = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)

def crop(tensor: torch.Tensor, k: int) -> torch.Tensor:
    """Crops border of width k."""
    if k == 0:
        return tensor
    return tensor[..., k:-k, k:-k]


class DataStructuralSimilarityIndex(Metric):
    """
    Computes the Data Structural Similarity Index (DSSIM) between two Tensors.

    This metric assesses the structural similarity between `map1` and `map2` by 
    considering local means, variances, and covariances using a Gaussian filter.
    
    Params
    ------
    map1 (np.ndarray)
        First input map.
    map2 (np.ndarray)
        Second input map.
    params (dict)
        Dictionary containing optional parameters:
        - `C1` (float)
            Stabilization constant for mean term. Default: 1.0e-8.
        - `C2` (float)
            Stabilization constant for contrast term. Default: 1.0e-8.
        - `k` (int)
            Cropping border size. Default: 5.
        - `x_stddev` (float)
            Gaussian kernel standard deviation. Default: 1.5.
        - `x_size` (int)
            Gaussian kernel width. Default: 11.
        - `y_size` (int)
            Gaussian kernel height. Default: 11.
        - `discretize` (bool)
            Whether to discretize the input data. Default: False.
        - `return_contrast` (bool)
            If True, return separate mean and contrast terms. Default: False.
    Methods
    -------
    `_normalize(map1, map2)` 
        Min-max normalization of `map1` and `map2` to the range [0,1].
    `_discretize(map1, map2, nbins)` 
        Discretizes `map1` and `map2` into `nbins` levels.
    `compute()` 
        Computes DSSIM, optionally returning mean and contrast components.

    """
    def __init__(self, 
                 map1: torch.Tensor, 
                 map2: torch.Tensor, 
                 params: Dict = None):
        """
        Initializes the DSSIM metric.
        """
        super().__init__(map1, map2, params)
        
        self.eps = 1e-12  # Small value to avoid division by zero

        self.C1 = params.get('C1', 1e-8)
        self.C2 = params.get('C2', 1e-8)
        self.k = params.get('k', 5)
        self.std = params.get('std', 1.5)
        self.ksize = params.get('k_size', 11)
        self.discretize = params.get('discretize', False)
        self.return_contrast = params.get('return_contrast', False)

        self.kernel = create_gaussian_kernel(self.ksize, self.std) #.to(map1.device)

    def _normalize(self, 
                   map1: torch.Tensor, 
                   map2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Min-max normalization of `map1` and `map2` to the range [0,1].
        """
        #smin = torch.min(torch.nanmin(map1), torch.nanmin(map2))
        #smax = torch.max(torch.nanmax(map1), torch.nanmax(map2))
        smin = torch.minimum(torch.nan_to_num(map1, nan=float('inf')).min(),
                     torch.nan_to_num(map2, nan=float('inf')).min())

        smax = torch.maximum(torch.nan_to_num(map1, nan=float('-inf')).max(),
                     torch.nan_to_num(map2, nan=float('-inf')).max())
        s1 = (map1 - smin) / (smax - smin + self.eps)
        s2 = (map2 - smin) / (smax - smin + self.eps)
        return s1, s2

    def _discretize(self, 
                    map1: torch.Tensor, 
                    map2: torch.Tensor, 
                    nbins: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretizes `map1` and `map2` into `nbins` levels.
        """
        q1 = torch.round(map1 * (nbins - 1))/(nbins - 1)
        q2 = torch.round(map2 * (nbins - 1))/(nbins - 1) 
        return q1, q2
    
    def _convolve(self, x: torch.Tensor) -> torch.Tensor:
        """2D Gaussian filtering with padding."""
        # expects shape (B, 1, H, W)
        return F.conv2d(x, self.kernel, padding=self.ksize // 2)
    
    def compute(self) -> float:
        """
        Computes the DSSIM metric.

        Returns
        -------
            tuple(float, str)
                Mean DSSIM value and unit. Optionally, returns `tuple(tuple, str)` separate mean and contrast terms if `return_contrast` is `True`.
        """
        out1, out2 = self._normalize(map1=self.map1, map2=self.map2)
        
        if self.discretize:
            out1, out2 = self._discretize(map1=out1, map2=out2)

        if len(out1.shape) and len(out2.shape) == 2:
            # Add batch and channel dims --> (1, 1, H, W)
            out1 = out1.unsqueeze(0).unsqueeze(0)
            out2 = out2.unsqueeze(0).unsqueeze(0)
       
        # local mean
        out1_mu = self._convolve(out1)
        out2_mu = self._convolve(out2)
        
        # local mean of squared values
        out1out1 = self._convolve(out1 * out1)
        out2out2 = self._convolve(out2 * out2)
        
        # local cross-product
        out1out2 = self._convolve(out1 * out2)
        
        # variance and covariance
        var_out1 = out1out1 - out1_mu ** 2
        var_out2 = out2out2 - out2_mu ** 2
        cov_out1out2 = out1out2 - out1_mu * out2_mu

        ssim_1 = (2 * out1_mu * out2_mu + self.C1)/(out1_mu ** 2 + out2_mu ** 2 + self.C1)
        ssim_2 = (2 * cov_out1out2 + self.C2)/(var_out1 + var_out2 + self.C2)
        ssim_matrix = ssim_1 * ssim_2

        # cropping (the border region)
        ssim_matrix = crop(ssim_matrix, self.k)
        mean_ssim = torch.nanmean(ssim_matrix)
  
        if self.return_contrast:
            ssim_1_cropped = crop(ssim_1, self.k)
            ssim_2_cropped = crop(ssim_2, self.k)
            return (torch.nanmean(ssim_1_cropped).item(), torch.nanmean(ssim_2_cropped).item())
        else:
            return mean_ssim.item()