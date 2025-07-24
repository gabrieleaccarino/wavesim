from typing import Dict, Union
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

        self.map1 = map1
        self.map2 = map2
        self.params = params #if params is not None else {}

    def compute(self):
        raise NotImplementedError("This method should be overridden in subclasses.")