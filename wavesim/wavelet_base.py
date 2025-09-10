
from pytorch_wavelets import DWTForward, DWTInverse
from typing import Optional
import numpy as np
import torch
import pywt

class Wavelet2DBaseTorch(object):
    def __init__(self, 
                data: np.ndarray, 
                wavelet: object, 
                mode: Optional[str] = 'zero', 
                levels: Optional[int] = 3):
        """Initialize the Wavelet2D object."""

        self.data = data
        self.B, self.C, self.H, self.W = self.data.shape
        self.wavelet = pywt.Wavelet(wavelet)
        self.mode = mode
        self.levels = min(levels, pywt.dwt_max_level(min(self.H, self.W), self.wavelet))

        self._coeffs = None
        #self._full_coeffs = None
        #self._signature = None
        #self._thresholds = None
        #self._thr_coeffs = None

        self.forward_wavelet = DWTForward(J=self.levels, mode=self.mode, wave=self.wavelet)
        self.inverse_wavelet = DWTInverse(mode=self.mode, wave=self.wavelet)

    @property
    def coeffs(self):
        """Perform wavelet decomposition across the specified levels and cache coefficients."""
        if self._coeffs is None:
            self._coeffs = self.forward_wavelet(self.data)
        return self._coeffs
    
    def scale_inversion(self, operation: str=None):
        """
        Reconstructs each detail component at each scale independently, summing or averaging orientations.
        Returns list of reconstructed maps from each detail scale [finest -> coarsest] and approx.
        """
        approx, details = self.coeffs
        rec_maps = []

        # approximation
        zeroed_details = [torch.zeros_like(detail) for detail in details]
        rec_map_approx = self.inverse_wavelet((approx, zeroed_details)) # no details
    
        # details
        # range across levels
        for level in range(self.levels):
            detail_coeff = details[level] # shape (B, C, 3, H_l, W_l)
            orientation_maps = []
            # range across orientations
            for ori in range(3):
                detail_mask = torch.zeros_like(detail_coeff)
                detail_mask[:, :, ori, :, :] = detail_coeff[:, :, ori, :, :]
                
                details_zeroed = [torch.zeros_like(d) for d in details]
                details_zeroed[level] = detail_mask
                
                rec_map_o = self.inverse_wavelet((torch.zeros_like(approx), details_zeroed))  # no approx
                orientation_maps.append(rec_map_o)

            stacked = torch.stack(orientation_maps, dim=-1)  # shape (B, C, H, W, 3)

            if operation == 'sum':
                rec_maps.append(stacked.sum(dim=-1)) # shape (B, C, H, W)
            elif operation == 'mean':
                rec_maps.append(stacked.mean(dim=-1)) # shape (B, C, H, W)
            else:
                raise ValueError(f'Operation not supported. Got {operation}.')

        rec_maps.append(rec_map_approx)
        return torch.stack(rec_maps, dim=-1)  # (B, C, H, W, L+1)
    
    def scale_inversion_ori(self):
        """
        Reconstructs each detail component at each scale independently.
        Instead of inverse transforming each orientation separately and then summing/averaging, keep them as they are.
        This approach is approximately the same as scale_inversion with operations applied later.
        Returns list of reconstructed maps from each detail scale [finest -> coarsest] and approx.
        """
        approx, details = self.coeffs
        rec_maps = []

        # approximation
        zeroed_details = [torch.zeros_like(detail) for detail in details]
        rec_map_approx = self.inverse_wavelet((approx, zeroed_details)) # no details
    
        # details
        # range across levels
        for level in range(self.levels):
            detail_coeff = details[level] # shape (B, C, 3, H_l, W_l)
            # range across orientations
            detail_mask = torch.zeros_like(detail_coeff)
            for ori in range(3):
                detail_mask[:, :, ori, :, :] = detail_coeff[:, :, ori, :, :]
                
            details_zeroed = [torch.zeros_like(d) for d in details]
            details_zeroed[level] = detail_mask
                
            rec_map_ori = self.inverse_wavelet((torch.zeros_like(approx), details_zeroed))  # no approx
            rec_maps.append(rec_map_ori)

        rec_maps.append(rec_map_approx)
        return torch.stack(rec_maps, dim=-1)  # (B, C, H, W, L+1)