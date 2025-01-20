# Standard library
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Third-party
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseDatastore, BaseRegularGridDatastore


@dataclass
class PlotCoordinates:
    """Container for coordinate information needed for plotting.

    Parameters
    ----------
    grid_index : xr.DataArray
        Grid indices for the data points
    feature_names : list[str]
        Names of the features in the data
    feature_units : list[str]
        Units for each feature
    x_coords : xr.DataArray, optional
        X coordinates of data points
    y_coords : xr.DataArray, optional
        Y coordinates of data points
    projection : ccrs.Projection, optional
        Cartographic projection to use
    """

    grid_index: xr.DataArray
    feature_names: List[str]
    feature_units: List[str]
    x_coords: Optional[xr.DataArray] = None
    y_coords: Optional[xr.DataArray] = None
    projection: Optional[ccrs.Projection] = None

    def __post_init__(self):
        """Validate the coordinates after initialization."""
        if not isinstance(self.grid_index, xr.DataArray):
            raise TypeError("grid_index must be an xarray DataArray")

        if len(self.feature_names) != len(self.feature_units):
            raise ValueError(
                f"Number of feature names ({len(self.feature_names)}) must "
                f"match number of units ({len(self.feature_units)})"
            )
        if (self.x_coords is None) != (self.y_coords is None):
            raise ValueError(
                "Both x_coords and y_coords must be provided together"
            )


class Visualizer:
    """Handles conversion of tensors to plotable arrays with coordinates.

    Parameters
    ----------
    interior_datastore : BaseDatastore
        Datastore containing the interior domain data
    boundary_datastore : BaseDatastore, optional
        Datastore containing boundary data
    boundary_var_map : Dict[str, str], optional
        Mapping between interior and boundary variable names
    """

    def __init__(
        self,
        interior_datastore: BaseDatastore,
        boundary_datastore: Optional[BaseDatastore] = None,
        boundary_var_map: Optional[Dict[str, str]] = None,
    ):
        if not isinstance(interior_datastore, BaseDatastore):
            raise TypeError("interior_datastore must be a BaseDatastore")
        if boundary_datastore is not None and not isinstance(
            boundary_datastore, BaseDatastore
        ):
            raise TypeError("boundary_datastore must be a BaseDatastore")
        # Add type checking for boundary_var_map
        if boundary_var_map is not None and not isinstance(
            boundary_var_map, dict
        ):
            raise TypeError("boundary_var_map must be a dictionary")

        self._interior_datastore = interior_datastore
        self._boundary_datastore = boundary_datastore
        self.boundary_var_map = boundary_var_map or {}

        # Cache coordinate information
        self.interior_coords = self._extract_coords(interior_datastore, "state")
        if boundary_datastore:
            try:
                self.boundary_coords = self._extract_coords(
                    boundary_datastore, "forcing"
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to extract boundary coordinates"
                ) from e
        else:
            self.boundary_coords = None

    @staticmethod
    def _extract_coords(
        datastore: BaseDatastore, category: str
    ) -> PlotCoordinates:
        """Extract coordinate information from a datastore.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore to extract coordinates from
        category : str
            Data category to extract ('state' or 'forcing')

        Returns
        -------
        PlotCoordinates
            Container with extracted coordinate information

        Raises
        ------
        ValueError
            If required data is missing from datastore
        """
        da = datastore.get_dataarray(category=category, split="train")
        if da is None:
            raise ValueError(f"No {category} data found in datastore")

        try:
            return PlotCoordinates(
                grid_index=da.grid_index,
                feature_names=datastore.get_vars_names(category),
                feature_units=datastore.get_vars_units(category),
                x_coords=da.x if "x" in da.coords else None,
                y_coords=da.y if "y" in da.coords else None,
                projection=getattr(datastore, "coords_projection", None),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to extract coordinates for {category}"
            ) from e

    def tensor_to_dataarray(
        self,
        tensor: torch.Tensor,
        times: Union[int, List[int], torch.Tensor],
        category: str,
        is_boundary: bool = False,
    ) -> xr.DataArray:
        """Convert tensor to DataArray with proper coordinates.

        Parameters
        ----------
        tensor : torch.Tensor
            Data tensor to convert
        times : Union[int, List[int], torch.Tensor]
            Time points in nanoseconds since epoch
        category : str
            Data category ('state' or 'forcing')
        is_boundary : bool, optional
            Whether this is boundary data

        Returns
        -------
        xr.DataArray
            DataArray with proper coordinates
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if category not in ("state", "forcing"):
            raise ValueError("category must be 'state' or 'forcing'")

        # Move to CPU and convert to numpy
        tensor = tensor.detach().cpu().numpy()

        # Handle times input
        if isinstance(times, (int, np.integer)):
            times = np.array([times], dtype="datetime64[ns]")
        elif isinstance(times, torch.Tensor):
            times = times.detach().cpu().numpy().astype("datetime64[ns]")
        else:
            if not isinstance(times, (list, np.ndarray)):
                raise TypeError(
                    "times must be int, list, numpy array or torch tensor"
                )
            times = np.array(times, dtype="datetime64[ns]")

        # Select appropriate coordinates
        coords = self.boundary_coords if is_boundary else self.interior_coords

        # Build coordinates dict
        coord_dict = {
            "grid_index": coords.grid_index,
            f"{category}_feature": coords.feature_names,
        }

        # Handle boundary data with window dimension
        if is_boundary and len(tensor.shape) == 3:
            # For boundary data with shape (grid_index, window, feature)
            dims = ["grid_index", "window", f"{category}_feature"]
            grid_size, window_size, feat_size = tensor.shape

            coord_dict.update({
                "window": np.arange(window_size),
            })

            # Time becomes a scalar coordinate for boundary data
            if len(times) != 1:
                raise ValueError("Boundary data requires single time value")
            coord_dict["time"] = times[0]

        else:
            # Handle regular data
            if len(tensor.shape) == 2:
                # Shape: (grid_index, feature)
                dims = ["grid_index", f"{category}_feature"]
                if len(times) != 1:
                    raise ValueError(
                        f"Expected single time value for 2D tensor, got {len(times)}"
                    )
                coord_dict["time"] = times[0]
            else:
                # Shape: (time, grid_index, feature)
                dims = ["time", "grid_index", f"{category}_feature"]
                coord_dict["time"] = times

        # Create DataArray
        da = xr.DataArray(tensor, dims=dims, coords=coord_dict)

        # Add x/y coordinates if needed
        if not isinstance(da.coords["grid_index"].to_index(), pd.MultiIndex):
            da.coords["x"] = coords.x_coords
            da.coords["y"] = coords.y_coords

        return da

    @matplotlib.rc_context(utils.fractional_plot_bundle(1))
    def plot_error_map(
        errors: Union[np.ndarray, torch.Tensor],
        datastore: BaseRegularGridDatastore,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a heatmap of errors of different variables at different prediction
        horizons.

        Parameters
        ----------
        errors : Union[np.ndarray, torch.Tensor]
            Array of errors to plot
        datastore : BaseRegularGridDatastore
            Datastore containing the data
        title : str, optional
            Title for the plot

        Returns
        -------
        plt.Figure
            The resulting figure

        Raises
        ------
        TypeError
            If errors is not a numpy array or torch tensor
        """
        if not isinstance(errors, (np.ndarray, torch.Tensor)):
            raise TypeError("errors must be numpy array or torch tensor")

        errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
        d_f, pred_steps = errors_np.shape
        step_length = datastore.step_length

        # Normalize all errors to [0,1] for color map
        max_errors = errors_np.max(axis=1)  # d_f
        errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

        fig, ax = plt.subplots(figsize=(15, 10))

        ax.imshow(
            errors_norm,
            cmap="OrRd",
            vmin=0,
            vmax=1.0,
            interpolation="none",
            aspect="auto",
            alpha=0.8,
        )

        # ax and labels
        for (j, i), error in np.ndenumerate(errors_np):
            # Numbers > 9999 will be too large to fit
            formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
            ax.text(
                i, j, formatted_error, ha="center", va="center", usetex=False
            )

        # Ticks and labels
        label_size = 15
        ax.set_xticks(np.arange(pred_steps))
        pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
        pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
        ax.set_xticklabels(pred_hor_h, size=label_size)
        ax.set_xlabel("Lead time (h)", size=label_size)

        ax.set_yticks(np.arange(d_f))
        var_names = datastore.get_vars_names(category="state")
        var_units = datastore.get_vars_units(category="state")
        y_ticklabels = [
            f"{name} ({unit})" for name, unit in zip(var_names, var_units)
        ]
        ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

        if title:
            ax.set_title(title, size=15)

        return fig


def find_closest_boundary_time(
    da_boundary_forcing: xr.DataArray, target_time: np.datetime64
) -> xr.DataArray:
    """Find the boundary forcing time closest to the target time.

    Parameters
    ----------
    da_boundary_forcing : xr.DataArray
        DataArray containing boundary forcing data with a 'window' dimension
    target_time : np.datetime64
        Target time to find closest match for

    Returns
    -------
    xr.DataArray
        Boundary forcing data at the closest time
    """
    window_times = da_boundary_forcing.window_time_deltas
    closest_idx = abs(window_times - target_time).argmin()
    return da_boundary_forcing.isel(window=closest_idx)


def plot_on_axis(
    ax: plt.Axes,
    da: xr.DataArray,
    datastore: BaseRegularGridDatastore,
    boundary_da: Optional[xr.DataArray] = None,
    boundary_datastore: Optional[BaseRegularGridDatastore] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "plasma",
) -> plt.Artist:
    """Plot weather state on given axis with optional boundary data.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to plot on
    da : xr.DataArray
        DataArray to plot
    datastore : BaseRegularGridDatastore
        Datastore containing the data
    boundary_da : xr.DataArray, optional
        Boundary data to plot
    boundary_datastore : BaseRegularGridDatastore, optional
        Datastore containing boundary data
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    cmap : str, optional
        Colormap to use

    Returns
    -------
    plt.Artist
        The plotted image artist
    """
    if not isinstance(ax, plt.Axes):
        raise TypeError("ax must be a matplotlib Axes object")

    if boundary_da is not None and boundary_datastore is None:
        raise ValueError("boundary_datastore required for boundary plotting")

    # Plot interior data
    extent = datastore.get_xy_extent("state")
    im = da.plot.imshow(
        ax=ax,
        origin="lower",
        x="x",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=datastore.coords_projection,
        zorder=2,  # Ensure interior is plotted on top
    )

    # Plot boundary data if provided
    if boundary_da is not None and boundary_datastore is not None:
        boundary_extent = boundary_datastore.get_xy_extent("forcing")
        boundary_da.plot.imshow(
            ax=ax,
            origin="lower",
            x="x",
            extent=boundary_extent,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            transform=boundary_datastore.coords_projection,
            alpha=0.5,  # Make boundary slightly transparent
            zorder=1,  # Ensure boundary is plotted below interior
        )

    ax.coastlines()  # Add coastline outlines
    return im


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    datastore: BaseRegularGridDatastore,
    da_prediction: Optional[xr.DataArray] = None,
    da_target: Optional[xr.DataArray] = None,
    da_boundary: Optional[xr.DataArray] = None,
    boundary_datastore: Optional[BaseRegularGridDatastore] = None,
    boundary_var_map: Optional[Dict[str, str]] = None,
    state_var_idx: Optional[int] = None,
    title: Optional[str] = None,
    vrange: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot example prediction and ground truth with optional boundary data.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
        Datastore containing the data
    da_prediction : xr.DataArray, optional
        Prediction data to plot
    da_target : xr.DataArray, optional
        Target data to plot
    da_boundary : xr.DataArray, optional
        Boundary data to plot
    boundary_datastore : BaseRegularGridDatastore, optional
        Datastore containing boundary data
    boundary_var_map : Dict[str, str], optional
        Mapping from interior variable names to boundary variable names
    state_var_idx : int, optional
        Index of the current state variable being plotted
    title : str, optional
        Title for the plot
    vrange : Tuple[float, float], optional
        Value range for the plot

    Returns
    -------
    plt.Figure
        The resulting figure

    Raises
    ------
    ValueError
        If boundary data is provided without boundary_datastore
    """
    if da_boundary is not None and boundary_datastore is None:
        raise ValueError(
            "boundary_datastore required when plotting boundary data"
        )

    if not hasattr(datastore, "coords_projection"):
        raise ValueError("datastore must have coords_projection")

    # Get common scale for values
    if vrange is None:
        vmin = min(da_prediction.min(), da_target.min())
        vmax = max(da_prediction.max(), da_target.max())
        # Only include boundary in value range if we will plot it
        if (
            da_boundary is not None
            and boundary_var_map
            and state_var_idx is not None
        ):
            state_var_name = datastore.get_vars_names("state")[state_var_idx]
            if state_var_name in boundary_var_map:
                vmin = min(vmin, da_boundary.min())
                vmax = max(vmax, da_boundary.max())
    else:
        vmin, vmax = vrange

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        subplot_kw={"projection": datastore.coords_projection},
    )

    # Only process boundary data if we have all required components
    boundary_da_to_plot = None
    if (
        da_boundary is not None
        and boundary_var_map
        and boundary_datastore is not None
        and state_var_idx is not None
    ):
        state_var_name = datastore.get_vars_names("state")[state_var_idx]
        if state_var_name in boundary_var_map:
            boundary_var_name = boundary_var_map[state_var_name]
            # Find index of boundary variable
            boundary_vars = boundary_datastore.get_vars_names("forcing")
            try:
                boundary_var_idx = boundary_vars.index(boundary_var_name)
                boundary_da_to_plot = da_boundary.isel(
                    forcing_feature=boundary_var_idx
                )
            except ValueError:
                print(
                    f"Warning: Boundary variable {boundary_var_name} not found"
                )

    # Plot pred and target with boundary
    for ax, da in zip(axes, (da_target, da_prediction)):
        plot_on_axis(
            ax,
            da,
            datastore,
            boundary_da=boundary_da_to_plot,
            boundary_datastore=boundary_datastore,
            vmin=vmin,
            vmax=vmax,
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(
    error: torch.Tensor,
    datastore: BaseRegularGridDatastore,
    title: Optional[str] = None,
    vrange: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot errors over spatial map.

    Parameters
    ----------
    error : torch.Tensor
        Error tensor to plot
    datastore : BaseRegularGridDatastore
        Datastore containing the data
    title : str, optional
        Title for the plot
    vrange : Tuple[float, float], optional
        Value range for the plot

    Returns
    -------
    plt.Figure
        The resulting figure
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    fig, ax = plt.subplots(
        figsize=(5, 4.8),
        subplot_kw={"projection": datastore.coords_projection},
    )

    error_grid = (
        error.reshape([
            datastore.grid_shape_state.x,
            datastore.grid_shape_state.y,
        ])
        .T.cpu()
        .numpy()
    )
    extent = datastore.get_xy_extent("state")

    # TODO: This needs to be converted to DA and use plot_on_axis
    im = ax.imshow(
        error_grid,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
