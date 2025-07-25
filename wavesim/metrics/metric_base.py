from typing import Dict, Union, Tuple
import xarray as xr
import numpy as np
import torch

class Metric():
    """
    Base class for all metrics.

    Parameters
    ----------
        map1 (torch.Tensor or np.ndarray) 
            The first tensor/ndarray.
        map2 (torch.Tensor or np.ndarray) 
            The second tensor/ndarray.
        params (dict)
            A dictionary of parameters to customize the metric behavior.

    Methods
    -------
        `compute()`
            Raises `NotImplementedError` as instances of this class should override the method.

    Raises
    ------
        `ValueError` if the two tensors/ndarray have different shapes.
    """

    def __init__(self, 
                 map1: Union[torch.Tensor, np.ndarray], 
                 map2: Union[torch.Tensor, np.ndarray], 
                 params: Dict):

        if map1.shape != map2.shape:
            raise ValueError(f"Input maps must have the same shape. Got {map1.shape} and {map2.shape}.")

        self.map1, self.map2 = self._process_input(map1, map2)
        assert self.map1.shape == self.map2.shape
        
        self.params = params if params is not None else {}
    
    def _process_input(self,
                   map1: Union[np.ndarray, xr.DataArray, torch.Tensor],
                   map2: Union[np.ndarray, xr.DataArray, torch.Tensor]
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts inputs to torch.Tensor with shape (B, C, H, W)"""
    
        def to_tensor(x):
            if isinstance(x, xr.DataArray):
                x = torch.tensor(x.values, dtype=torch.float32)
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Unsupported input type: {type(x)}")
            
            while x.ndim < 4:
                x = x.unsqueeze(0)
            return x

        return to_tensor(map1), to_tensor(map2)

    def compute(self):
        raise NotImplementedError("This method should be overridden in subclasses.")