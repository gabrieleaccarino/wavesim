
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import pywt

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
    
    
# Function to plot approximation and detail coefficients at each level
def plot_wavelet_decomposition(coeffs: list):
    fig, axes = plt.subplots(len(coeffs)-1, 4, figsize=(12, 3 * (len(coeffs)-1)))
    cA = coeffs[0]
    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        min_val = min(np.min(cH), np.min(cV), np.min(cD))
        max_val = max(np.max(cH), np.max(cV), np.max(cD))
        # Plot approximation and detail coefficients at each level

        if i == 1:
            im = axes[i-1, 0].imshow(cA, cmap='gray')
            fig.colorbar(im, ax=axes[i-1, 0], fraction=0.046, pad=0.04)
            axes[i-1, 0].set_title(f'Approximation Level {len(coeffs)-i}')
            axes[i-1, 0].axis('on')
        else:
            axes[i-1, 0].axis('off')

        im = axes[i-1, 1].imshow(cH, cmap='gray', vmin=min_val, vmax=max_val)
        fig.colorbar(im, ax=axes[i-1, 1], fraction=0.046, pad=0.04)
        axes[i-1, 1].set_title(f'Horizontal Detail Level {len(coeffs)-i}')

        im = axes[i-1, 2].imshow(cV, cmap='gray', vmin=min_val, vmax=max_val)
        fig.colorbar(im, ax=axes[i-1, 2], fraction=0.046, pad=0.04)
        axes[i-1, 2].set_title(f'Vertical Detail Level {len(coeffs)-i}')

        im = axes[i-1, 3].imshow(cD, cmap='gray', vmin=min_val, vmax=max_val)
        fig.colorbar(im, ax=axes[i-1, 3], fraction=0.046, pad=0.04)
        axes[i-1, 3].set_title(f'Diagonal Detail Level {len(coeffs)-i}')

    plt.tight_layout()
    plt.show()


# Function to plot approximation and detail coefficients at each level, nested representation
# TODO : dump the generated picture on disk
def multi_scale_wavelet_plot(signal: np.array, 
                        max_levels: int, 
                        level: int = 1,
                        wavelet: pywt = 'haar',
                        mode: str = 'sym', 
                        figsize: tuple = (15, 15),
                        cmap: str = 'RdBu_r',
                        subplot_spec: plt.figure = None, 
                        fig: plt.figure = None,
                        outer_grid: plt.figure = None,
                        wspace: float = 0.01,
                        hspace: float = 0.01):
    
    if (level > max_levels) or (max_levels < level):
        return
    
    if (fig is None) and (outer_grid is None):
        fig = plt.figure(figsize=figsize if not None else (15, 15))
        outer_grid = fig.add_gridspec(2, 2, wspace=wspace, hspace=hspace)
    
    # If it's the first call, set up the outer grid
    if (subplot_spec is None):
        subplot_spec = outer_grid[0, 0]  # Start with the top-left corner
    
    # Perform one level of 2D DWT
    cA, (cH, cV, cD) = pywt.dwt2(signal, wavelet=wavelet, mode=mode)
    # Arrange approximation and details in a list
    coeffs = [cA, cH, cV, cD]
    # normalize each coefficient array independently for better visibility
    coeffs = [c/np.abs(c).max() for c in coeffs]
    
    # Set up a 2x2 grid for the current level within the provided subplot_spec
    inner_grid = subplot_spec.subgridspec(2, 2, wspace=wspace*level, hspace=hspace*level)
    
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(inner_grid[i, j])
            ax.axis('off')           # Disable axis ticks and labels
            ax.set_frame_on(False)   # Remove the border/frame

            if (i == j == 0) and (level < max_levels):
                # Recursively plot in the top-left cell with the approximation coefficients
                multi_scale_wavelet_plot(coeffs[0], level=level + 1, max_levels=max_levels, subplot_spec=inner_grid[i, j], fig=fig, outer_grid=outer_grid)
            else:
                # Plot detail coefficients (cH, cV, cD) in other cells
                ax.imshow(coeffs[i * 2 + j], cmap=cmap)
                
                
class PlotWaveletDecomposition:
    def __init__(self, data: np.ndarray, wavelet: str = 'haar',
                 mode: str = 'periodization', levels: int = 3, dst_path=None):
        self.xrdata = data
        self.data = np.asarray(data[::-1], dtype=float)
        self.wavelet, self.mode, self.levels = wavelet, mode, levels
        self.flat_coeff = self._compute_all_coeffs()
        self.cbar_min = min(np.min(c) for c in self.flat_coeff)
        self.cbar_max = max(np.max(c) for c in self.flat_coeff)
        self.dst_path = dst_path
        #print("Coeff range:", self.cbar_min, self.cbar_max)

    def _compute_all_coeffs(self):
        coeffs_list, signal = [], self.data.copy()
        for _ in range(self.levels):
            cA, (cH, cV, cD) = pywt.dwt2(signal, self.wavelet, self.mode)
            coeffs_list.extend([cA, cH, cV, cD])
            signal = cA
        return coeffs_list

    def _multi_scale_wavelet_plot(self, signal, max_levels, level,
                                  cmap, fig, subplot_spec,
                                  vmin, vmax):
        """Recursive multi-scale wavelet plotting."""
        if level > max_levels:
            return
        outer_grid = subplot_spec.subgridspec(2, 2, wspace=0.02, hspace=0.02)
        cA, (cH, cV, cD) = pywt.dwt2(signal, self.wavelet, self.mode)
        coeffs = [cA, cH, cV, cD]
        
        for i in range(2):
            for j in range(2):
                ax = fig.add_subplot(outer_grid[i, j])
                ax.axis('off')
                #ax.set_xticks([])
                #ax.set_yticks([])
                if i == j == 0 and level < max_levels:
                    self._multi_scale_wavelet_plot(cA, max_levels, level + 1,
                                                   cmap, fig, outer_grid[i, j],
                                                   vmin, vmax)
                else:
                    ax.imshow(coeffs[i * 2 + j], cmap=cmap, vmin=vmin, vmax=vmax)

    def plot_decomposition(self, cmap: str = 'RdBu_r', figsize: tuple = (12, 6),
                           lon=None, lat=None):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[20, 0.5],
                              width_ratios=[1, 1], hspace=0.05, wspace=0.05)

        # --- Left plot ---
        proj = ccrs.PlateCarree()
        ax0 = fig.add_subplot(gs[0, 0], projection=proj)
        ax0.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)

        if lon is None or lat is None:
            ny, nx = self.data.shape
            lon, lat = np.linspace(-180, 180, nx), np.linspace(-90, 90, ny)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        cmap_left = ListedColormap(SHIFTED_BRBG_COLORS)
        color_levels = [0.0, 0.1, 0.5, 1, 2, 5, 10, 15, 20]
        norm_left = BoundaryNorm(color_levels, ncolors=len(color_levels))
        im0 = ax0.contourf(lon_grid, lat_grid, self.xrdata,
                           levels=color_levels, cmap=cmap_left,
                           norm=norm_left, extend='max',
                           transform=proj)

        gl = ax0.gridlines(draw_labels=True, linewidth=0.5,
                           color='gray', alpha=0.5, linestyle='--')
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}
        gl.top_labels = gl.right_labels = False

        # --- Right plot ---
        ax1 = fig.add_subplot(gs[0, 1], projection=proj)
        ax1.axis("off")

        # Fixed discrete levels for colorbar
        discrete_levels = [-10, -8, -5, -3, 0, 3, 5, 8, 10]
        norm_right = BoundaryNorm(discrete_levels, ncolors=plt.get_cmap(cmap).N, extend="both")

        # Apply common color scale to all wavelet plots
        self._multi_scale_wavelet_plot(self.data, self.levels, 1,
                                       cmap, fig, gs[0, 1],
                                       vmin=discrete_levels[0],
                                       vmax=discrete_levels[-1])

        # --- Colorbars ---
        left_cbar = self.add_cbar(fig, ax0, im0,
                      formatter=FuncFormatter(self.custom_format),
                      #label='Precipitation (mm/6hr)',
                      ticks=color_levels,
                      )
        
        left_cbar.set_label('Precipitation (mm/6hr)', fontsize=14)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_right)
        self.add_cbar(fig, ax1, sm, ticks=discrete_levels)

        if self.dst_path is not None:
            fig.savefig(self.dst_path, bbox_inches='tight', dpi=300)
            print(f"Saved figure to {self.dst_path}")
        plt.show()

    @staticmethod
    def add_cbar(fig, ax, mappable,
                 formatter=None, label=None,
                 ticks=None, shrink_factor=0.7):
        bbox = ax.get_position()
        cbar_width = bbox.width * shrink_factor
        cax = fig.add_axes([bbox.x0 + (bbox.width - cbar_width) / 2,
                            bbox.y0 - 0.06, cbar_width, 0.02])
        cbar = fig.colorbar(mappable, cax=cax,
                            orientation='horizontal', format=formatter,
                            extend='both')
        if label:
            cbar.set_label(label, fontsize=12)
        if ticks:
            cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=12)
        return cbar

    @staticmethod
    def custom_format(x, _):
        return f"{int(x)}" if x == int(x) else f"{x:.1f}"
