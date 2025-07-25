import xarray as xr
import numpy as np
import torch

def noise_at_level(wavelet, magnitude, level, orientation, rand_state=32):
    """
    level = -1 --> approx
    level from 1 to levels --> details
    """

    torch.manual_seed(rand_state)

    orientations = {'H': 0, 'V': 1, 'D': 2}
    approx, details = wavelet.coeffs

    if level == -1 and orientation is None:
        coeff = approx
        std = torch.std(coeff)
        noise = torch.randn_like(coeff)
        approx += noise * std * magnitude
    else:
        coeff = details[level-1][0, 0, orientations[orientation], :, :]
        std = torch.std(coeff)
        noise = torch.randn_like(coeff)
        details[level-1][0, 0, orientations[orientation], :, :] += noise * std * magnitude
    
    wavelet._coeffs = (approx, details)
    out = wavelet.inverse_wavelet(wavelet._coeffs).cpu().numpy()[0,0,::]
    out = np.where(out < 0.0, 0.0, out)
    return out


def crop_centered_window(
    field,
    center_lon,
    center_lat,
    size_lon=256,
    size_lat=256,
    shift_lon_deg=0.0,
    shift_lat_deg=0.0
):
    """
    Extract a `size_lon` x `size_lat` window centered around a point,
    optionally shifted by `shift_lon_deg` / `shift_lat_deg`.

    Parameters
    ----------
    field (xarray DataArray or Dataset)
        xarray DataArray or Dataset with global coverage
    center_lon (float)
        center point ([-180, 180])
    center_lat (float)
        center point ([-90, 90])
    size_lon (int)
        number of grid points in lon (default 256)
    size_lat (int)
        number of grid points in lat (default 256)
    shift_lon_deg (float)
        degrees to shift the window across lon
    shift_lat_deg (float)
        degrees to shift the window across lat

    Returns
    -------
        cropped window of shape (size_lat, size_lon)
    """
    # Grid resolution
    lon_vals = field.longitude
    lat_vals = field.latitude
    dlon = float(lon_vals[1] - lon_vals[0])
    dlat = float(lat_vals[1] - lat_vals[0])

    # Apply shift
    center_lon_shifted = center_lon + shift_lon_deg
    center_lat_shifted = center_lat + shift_lat_deg

    # Define bounds in degrees
    #half_width_deg = (size_lon - 1) / 2 * dlon
    #half_height_deg = (size_lat - 1) / 2 * dlat

    half_width_deg = (size_lon) / 2 * dlon
    half_height_deg = (size_lat) / 2 * dlat

    lon_min = center_lon_shifted - half_width_deg
    lon_max = center_lon_shifted + half_width_deg
    lat_min = center_lat_shifted - half_height_deg
    lat_max = center_lat_shifted + half_height_deg

    # Wrap longitudes to [0, 360]
    def wrap_to_360(lon): 
        return lon % 360
    
    lon_min_wrapped = wrap_to_360(lon_min)
    lon_max_wrapped = wrap_to_360(lon_max)

    # Longitude selection with wrapping if needed
    if lon_min_wrapped < lon_max_wrapped:
        subset = field.sel(
            longitude=slice(lon_min_wrapped, lon_max_wrapped),
            latitude=slice(lat_min, lat_max)
        )
    else:
        subset = xr.concat([
            field.sel(longitude=slice(lon_min_wrapped, 360)),
            field.sel(longitude=slice(0, lon_max_wrapped))
        ], dim='longitude').sel(latitude=slice(lat_min, lat_max))

    # Convert longitudes to [-180, 180] and sort
    subset = subset.assign_coords(
        longitude=(((subset.longitude + 180) % 360) - 180)
    ).sortby('longitude')

    # Ensure exact output shape
    return subset.isel(
        longitude=slice(0, size_lon),
        latitude=slice(0, size_lat)
    )