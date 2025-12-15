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
        self.p = params.get('p', 1)
        
        if not isinstance(self.p, int):
            raise ValueError(f'Parameter p must be an integer. Got {type(self.p)}.')

    def compute(self) -> tuple[float, str]:
        """
        Computes the NRMSE between `map1` and `map2`.
        
        Returns
        -------
            tuple (float, str)
                The computed NRMSE value and the unit (if any).
        """

        if self.p != 0:
            rand = torch.randperm(self.map1.nelement())
            rand = self.map1.view(-1)[rand].view(self.map1.size())
            R = torch.sqrt(torch.mean((self.map1 - self.map2)**2))
            R_rand = torch.sqrt(torch.mean((self.map1 - rand)**2))
            nrmse = (R / (R_rand + self.eps)) ** self.p
            sim_nrmse = torch.exp(-nrmse)
        else:
            ValueError(f'Parameter p must be different from 0. Got {self.p}.')
        
        return sim_nrmse.item()