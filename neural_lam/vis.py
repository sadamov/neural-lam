# Standard library
from typing import Any, Dict, Optional, Tuple, Union

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseDatastore, BaseRegularGridDatastore


class PlotDataManager:
    """Handles data preparation and metadata management for plotting.

    This class manages metadata and coordinates for plotting weather data,
    including both interior and boundary data. It provides functionality to
    convert tensors to xarray DataArrays with proper coordinates and metadata.

    Parameters
    ----------
    interior_datastore : BaseDatastore
        Datastore containing the interior domain data
    boundary_datastore : BaseDatastore, optional
        Datastore containing boundary data
    boundary_var_map : Dict[str, str], optional
        Mapping between interior and boundary variable names

    Attributes
    ----------
    metadata_interior : Dict[str, Any]
        Metadata for interior domain plotting
    metadata_boundary : Optional[Dict[str, Any]]
        Metadata for boundary domain plotting, if boundary_datastore provided
    boundary_var_map : Dict[str, str]
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
        if boundary_var_map is not None and not isinstance(
            boundary_var_map, dict
        ):
            raise TypeError("boundary_var_map must be a dictionary")

        self._interior_datastore = interior_datastore
        self._boundary_datastore = boundary_datastore
        self.boundary_var_map = boundary_var_map or {}

        # Extract metadata
        self.metadata_interior = self._extract_metadata(
            interior_datastore, "state"
        )
        if boundary_datastore:
            try:
                self.metadata_boundary = self._extract_metadata(
                    boundary_datastore, "forcing"
                )
            except Exception as e:
                raise RuntimeError("Failed to extract boundary metadata") from e
        else:
            self.metadata_boundary = None

    @property
    def dims_interior(self) -> Tuple[str, ...]:
        """Get interior dimension names.

        Returns
        -------
        Tuple[str, ...]
            Dimension names for interior data
        """
        return self.metadata_interior["dims"]

    @property
    def dims_boundary(self) -> Optional[Tuple[str, ...]]:
        """Get boundary dimension names.

        Returns
        -------
        Optional[Tuple[str, ...]]
            Dimension names for boundary data, None if no boundary data
        """
        return (
            self.metadata_boundary["dims"] if self.metadata_boundary else None
        )

    @staticmethod
    def _extract_metadata(
        datastore: BaseDatastore, category: str
    ) -> Dict[str, Any]:
        """Extract metadata information from a datastore.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore to extract metadata from
        category : str
            Data category to extract ('state' or 'forcing')

        Returns
        -------
        Dict[str, Any]
            Dictionary containing metadata:
            - grid_index: xr.DataArray
            - feature_names: List[str]
            - feature_units: List[str]
            - x_coords: Optional[xr.DataArray]
            - y_coords: Optional[xr.DataArray]
            - projection: Optional[ccrs.Projection]
            - dims: Tuple[str, ...]

        Raises
        ------
        ValueError
            If required data is missing from datastore or extraction fails
        """
        da = datastore.get_dataarray(category=category, split="train")
        if da is None:
            raise ValueError(f"No {category} data found in datastore")

        try:
            return {
                "grid_index": da.grid_index,
                "feature_names": datastore.get_vars_names(category),
                "feature_units": datastore.get_vars_units(category),
                "x_coords": da.x if "x" in da.coords else None,
                "y_coords": da.y if "y" in da.coords else None,
                "projection": getattr(datastore, "coords_projection", None),
                "dims": tuple(da.dims),
            }
        except Exception as e:
            raise ValueError(
                f"Failed to extract metadata for {category}"
            ) from e

    def tensor_to_dataarray(
        self,
        tensor: torch.Tensor,
        times: torch.Tensor,
        category: str,
        is_boundary: bool = False,
    ) -> xr.DataArray:
        """Convert tensor to DataArray with proper coordinates and metadata.

        Parameters
        ----------
        tensor : torch.Tensor
            Data tensor to convert
        times : torch.Tensor
            Time points in nanoseconds since epoch
        category : str
            Data category ('state' or 'forcing')
        is_boundary : bool, optional
            Whether this is boundary data (to use boundary metadata)

        Returns
        -------
        xr.DataArray
            DataArray with proper coordinates, dimensions and metadata

        Raises
        ------
        TypeError
            If tensor is not a torch.Tensor
        ValueError
            If category is invalid or required metadata missing
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if category not in ("state", "forcing"):
            raise ValueError("category must be 'state' or 'forcing'")

        metadata = (
            self.metadata_boundary if is_boundary else self.metadata_interior
        )

        # Move to CPU and convert to numpy
        tensor = tensor.detach().cpu().numpy()
        times = times.detach().cpu().numpy().astype("datetime64[ns]")

        # Build coordinates dict
        coord_dict = {
            "time": times,
            "grid_index": metadata["grid_index"],
            f"{category}_feature": metadata["feature_names"],
        }

        # Create DataArray with dims from metadata
        da = xr.DataArray(tensor, dims=metadata["dims"], coords=coord_dict)

        # Add x/y coordinates if needed
        if not isinstance(da.coords["grid_index"].to_index(), pd.MultiIndex):
            da.coords["x"] = metadata["x_coords"]
            da.coords["y"] = metadata["y_coords"]

        return da


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(
    errors: Union[np.ndarray, torch.Tensor],
    datastore: BaseRegularGridDatastore,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot a heatmap of errors of different variables at different
    prediction horizons.

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
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

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
        x="x" if "x" in da.coords else None,  # Make x coordinate optional
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=datastore.coords_projection,
        zorder=2,  # Ensure interior is plotted on top
    )

    # Plot boundary data if provided
    if boundary_da is not None and boundary_datastore is not None:
        try:
            boundary_extent = boundary_datastore.get_xy_extent("forcing")
            boundary_da.plot.imshow(
                ax=ax,
                origin="lower",
                x="x" if "x" in boundary_da.coords else None,
                extent=boundary_extent,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                transform=boundary_datastore.coords_projection,
                alpha=0.5,  # Make boundary slightly transparent
                zorder=1,  # Ensure boundary is plotted below interior
            )
        except AttributeError as e:
            print(f"Warning: Could not plot boundary data: {e}")

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
        error.reshape(
            [
                datastore.grid_shape_state.x,
                datastore.grid_shape_state.y,
            ]
        )
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
