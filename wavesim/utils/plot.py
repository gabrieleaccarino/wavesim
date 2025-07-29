
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np

SHIFTED_BRBG_COLORS = np.array([[0.95724721, 0.95993849, 0.95955402, 1.        ],
       [0.90065359, 0.94640523, 0.93986928, 1.        ],
       #[0.83698577, 0.93118032, 0.91772395, 1.        ],
       [0.78039216, 0.91764706, 0.89803922, 1.        ],
       #[0.68212226, 0.87750865, 0.84821223, 1.        ],
       [0.59477124, 0.84183007, 0.80392157, 1.        ],
       #[0.49619377, 0.79976932, 0.75301807, 1.        ],
       [0.40392157, 0.73333333, 0.69150327, 1.        ],
       #[0.30011534, 0.65859285, 0.62229912, 1.        ],
       [0.20784314, 0.59215686, 0.56078431, 1.        ],
       #[0.13587082, 0.52433679, 0.49296424, 1.        ],
       [0.07189542, 0.46405229, 0.43267974, 1.        ],
       #[0.00384468, 0.39677047, 0.36509035, 1.        ],
       [0.00261438, 0.34509804, 0.30849673, 1.        ],
       [0.0012303 , 0.28696655, 0.24482891, 1.        ],
       [0.        , 0.23529412, 0.18823529, 1.        ]])


def custom_format(x, pos):
    return f"{int(x)}" if x == int(x) else f"{x:.1f}"

def plot_perturbation(reference, perturbed, reference_coords, perturbed_coords, reference_label, perturbed_label, cbar_label, cbar_lim = None):
    gridline_settings = {'draw_labels': True, 'linewidth': 0.0, 
                        'color': 'gray', 'alpha': 0.5, 'linestyle': '--'}

    color_levels = [0.0, 0.1, 0.5, 1, 2, 5, 10, 15, 20]
    color_levels_bias = [-20, -15, -10, -5, -2, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 2, 5, 10, 15, 20]
    cmap = ListedColormap(SHIFTED_BRBG_COLORS)
    norm = BoundaryNorm(list(color_levels), ncolors=len(color_levels))
    norm_bias = BoundaryNorm(list(color_levels_bias), ncolors=len(color_levels_bias))
    formatter = FuncFormatter(custom_format)

    # figure
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Left Panel : reference map
    lon, lat = reference_coords
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    domain_extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    ax0.set_extent(domain_extent, crs=ccrs.PlateCarree())
    ax0.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, zorder=2)
    ax0.set_title(f'{reference_label}')
    reference_map = ax0.contourf(lon_grid, lat_grid, reference, levels=color_levels, cmap=cmap, norm=norm, extend='max', transform=ccrs.PlateCarree())
    gl = ax0.gridlines(**gridline_settings)
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(reference_map, ax=ax0, fraction=0.04, pad=0.08, orientation='horizontal', format=formatter)
    cbar.set_label(f'{cbar_label}', fontsize=10)
    cbar.set_ticks(color_levels)
    cbar.ax.tick_params(labelsize=10)

    # Central Panel : perturbed map (plotted over the same reference grid)
    lon, lat = perturbed_coords
    domain_extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    ax1.set_extent(domain_extent, crs=ccrs.PlateCarree())  # Use reference extent
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, zorder=2)
    ax1.set_title(f'{perturbed_label}')
    perturbed_map = ax1.contourf(lon_grid, lat_grid, perturbed, levels=color_levels,
                                  cmap=cmap, norm=norm, extend='max', transform=ccrs.PlateCarree())
    gl = ax1.gridlines(**gridline_settings)
    gl.top_labels = False
    gl.right_labels = False
    cbar = plt.colorbar(perturbed_map, ax=ax1, fraction=0.04, pad=0.08, orientation='horizontal', format=formatter)
    cbar.set_label(f'{cbar_label}', fontsize=10)
    cbar.set_ticks(color_levels)
    cbar.ax.tick_params(labelsize=10)

    # Right Panel : bias map (reference - perturbed)
    #ax2.set_extent(domain_extent, crs=ccrs.PlateCarree())
    #ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, zorder=2)
    ax2.set_title(f'Reference - Perturbed')
    bias = reference - perturbed
    #norm = mcolors.TwoSlopeNorm(vmin=bias.min(), vcenter=0.0, vmax=bias.max())
    cbar_lim = cbar_lim if cbar_lim is not None else 20
    norm = mcolors.TwoSlopeNorm(vmin=-cbar_lim, vcenter=0.0, vmax=cbar_lim)
    bias_map = ax2.pcolormesh(lon_grid, lat_grid, bias, cmap='BrBG', norm=norm, shading='auto', transform=ccrs.PlateCarree())
    #bias_map = ax2.contourf(lon_grid, lat_grid, bias, levels=color_levels_bias,
    #                              cmap='BrBG', norm=norm_bias, extend='max', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(bias_map, ax=ax2, fraction=0.04, pad=0.08, orientation='horizontal', format=ScalarFormatter())
    cbar.set_label(f'Precipitation difference (mm)', fontsize=10)
    print(f'Bias min: {bias.min()}, max: {bias.max()}')