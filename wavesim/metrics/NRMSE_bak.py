from typing import Dict
import torch

from .metric_base import Metric

class NormalizedRootMeanSquaredError(Metric):
    """
    Computes the Normalized Root Mean Squared Error (NRMSE) between two input maps.
    
    Params
    ------
        map1 (torch.Tensor)
            First input map.
        map2 (torch.Tensor)
            Second input map.
        params (Dict, optional)
            Additional parameters. Default is `None`.
    
    Methods
    -------
        `compute()`
            Computes RMSE.  
    """
    def __init__(self, 
                 map1: torch.Tensor, 
                 map2: torch.Tensor, 
                 params: Dict = None):
        
        super().__init__(map1, map2, params)
        
        self.eps = 1e-8
        self.mode = params.get('mode', 'range')

    def compute(self) -> tuple[float, str]:
        """
        Computes the NRMSE between `map1` and `map2`.
        
        Returns
        -------
            tuple (float, str)
                The computed NRMSE value and the unit (if any).
        """
        #rmse_max = np.sqrt(np.max((self.map1 - self.map2)**2)) if self.rmse_max is None else self.rmse_max

        if self.mode == 'range':
            smin = torch.nan_to_num(self.map1, nan=float('inf')).min()
            smax = torch.nan_to_num(self.map1, nan=float('-inf')).max()
            norm = smax - smin
        elif self.mode == 'mean':
            norm = torch.nanmean(self.map1)
        elif self.mode == 'std' or self.mode == 'standard_deviation':
            nan_mean = torch.nanmean(self.map1, keepdim=True)
            squared_diff = torch.pow(self.map1 - nan_mean, 2)
            nan_variance = torch.nanmean(squared_diff, keepdim=True)
            norm = torch.sqrt(nan_variance)
        elif self.mode == 'null':
            null = torch.randperm(self.map1.nelement())
            null = self.map1.view(-1)[null].view(self.map1.size())
            R = torch.mean((self.map1 - self.map2)**2)
            R_null = torch.sqrt(torch.mean((self.map1 - null)**2))
            #return 1.0 - (R / (R_null + self.eps)).item()
            return torch.exp(-(R / (R_null + self.eps) ** 1)).item()
        else:
            ValueError(f'Mode not recognized. Use "range", "mean", or "std". Got {self.mode}.')
        
        mse = torch.mean((self.map1 - self.map2)**2)
        rmse = torch.sqrt(mse)
        nrmse = rmse/(norm + self.eps)
        
        return 1.0 - nrmse.item()