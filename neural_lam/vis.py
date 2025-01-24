# Standard library
from typing import Dict, Optional, Tuple, Union

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore


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
    """Plot weather state on given axis with optional boundary data."""
    if not isinstance(ax, plt.Axes):
        raise TypeError("ax must be a matplotlib Axes object")

    if boundary_da is not None and boundary_datastore is None:
        raise ValueError("boundary_datastore required for boundary plotting")

    # Always use PlateCarree for input data (assuming lat/lon coordinates)
    data_proj = ccrs.PlateCarree()

    # First set up the map features
    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
    gl = ax.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        transform=data_proj,
    )
    gl.top_labels = False
    gl.right_labels = False

    # Get map extent in lat/lon coordinates
    if boundary_datastore is not None:
        extent = boundary_datastore.get_xy_extent("forcing", use_latlon=True)
        print("extent", extent)
    else:
        print("using state")
        extent = datastore.get_xy_extent("state", use_latlon=True)

    # Set extent in lat/lon coordinates
    ax.set_extent(extent, crs=data_proj)

    def get_coords_from_dataarray(
        da: xr.DataArray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to extract lat/lon or x/y coordinates from a DataArray"""
        # Try different possible coordinate names
        if hasattr(da, "longitude") and hasattr(da, "latitude"):
            print("using latitude/longitude")
            x = da.longitude.values
            y = da.latitude.values
        elif hasattr(da, "lon") and hasattr(da, "lat"):
            print("using lon/lat")
            x = da.lon.values
            y = da.lat.values
        else:
            print("using datastore")
            # Fallback to getting coordinates from datastore
            coords = datastore.get_lat_lon("state")
            x = coords[:, 0].reshape(da.shape)
            y = coords[:, 1].reshape(da.shape)
        if x.max() > 180:
            x = np.where(x > 180, x - 360, x)
        if y.max() > 90:
            y = np.where(y > 90, y - 180, y)

        return x, y

    im_boundary = None

    # Handle boundary data first
    if boundary_da is not None and boundary_datastore is not None:
        try:
            "Working with Boundary Data"
            x_boundary, y_boundary = get_coords_from_dataarray(boundary_da)
            if len(x_boundary.shape) == 1:
                print("reshaping")
                Y_boundary, X_boundary = np.meshgrid(y_boundary, x_boundary)
            else:
                X_boundary, Y_boundary = x_boundary, y_boundary
            print("BminX", X_boundary.min())
            print("BmaxX", X_boundary.max())
            print("BminY", Y_boundary.min())
            print("BmaxY", Y_boundary.max())
            print("BshapeX", X_boundary.shape)
            print("BshapeY", Y_boundary.shape)

            im_boundary = ax.pcolormesh(
                X_boundary,
                Y_boundary,
                boundary_da.values,
                transform=data_proj,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                alpha=0.5,
                shading="nearest",
            )
        except Exception as e:
            print(f"Warning: Failed to plot boundary data: {e}")

    try:
        "Working with Interior Data"
        x, y = get_coords_from_dataarray(da)
        if len(x.shape) == 1:
            X, Y = np.meshgrid(y, x)
        else:
            X, Y = x, y
            print("minX", X.min())
            print("maxX", X.max())
            print("minY", Y.min())
            print("maxY", Y.max())
            print("shapeX", X.shape)
            print("shapeY", Y.shape)
        im = ax.pcolormesh(
            X,
            Y,
            da.values,
            transform=data_proj,  # Data is in lat/lon coordinates
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            shading="nearest",
        )
    except Exception as e:
        print(f"Warning: Failed to plot interior data: {e}")
        raise

    # Add map features after plotting data
    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)
    ax.gridlines(draw_labels=True, transform=data_proj)

    return im, im_boundary


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

    # Calculate figure size based on the larger extent
    extent = datastore.get_xy_extent("state")
    plot_width = extent[1] - extent[0]
    plot_height = extent[3] - extent[2]

    if boundary_datastore is not None:
        boundary_extent = boundary_datastore.get_xy_extent("forcing")
        plot_width = max(plot_width, boundary_extent[1] - boundary_extent[0])
        plot_height = max(plot_height, boundary_extent[3] - boundary_extent[2])

    # Adjust figure size while maintaining aspect ratio
    aspect_ratio = plot_width / plot_height
    base_width = 15
    fig_width = base_width
    fig_height = base_width / (
        2 * aspect_ratio
    )  # divide by 2 because we have two subplots

    # Get common scale for values
    if vrange is None:
        vmin = float("inf")
        vmax = float("-inf")

        # Calculate vmin and vmax for interior data
        vmin = min(vmin, da_prediction.min().item(), da_target.min().item())
        vmax = max(vmax, da_prediction.max().item(), da_target.max().item())

        # Calculate vmin and vmax for boundary data if available
        if (
            da_boundary is not None
            and boundary_var_map
            and state_var_idx is not None
        ):
            state_var_name = datastore.get_vars_names("state")[state_var_idx]
            if state_var_name in boundary_var_map:
                boundary_var_name = boundary_var_map[state_var_name]
                boundary_var_idx = boundary_datastore.get_vars_names(
                    "forcing"
                ).index(boundary_var_name)
                boundary_da_var = da_boundary.isel(
                    forcing_feature=boundary_var_idx
                )
                vmin = min(vmin, boundary_da_var.min().item())
                vmax = max(vmax, boundary_da_var.max().item())
    else:
        vmin, vmax = vrange
    print("vmin", vmin)
    print("vmax", vmax)

    # Create figure with the correct projection set from the start
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, 2)
    axes = [
        fig.add_subplot(gs[0, i], projection=datastore.coords_projection)
        for i in range(2)
    ]

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
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    # Convert error to DataArray
    error_da = xr.DataArray(
        error.reshape(
            [
                datastore.grid_shape_state.x,
                datastore.grid_shape_state.y,
            ]
        )
        .T.cpu()
        .numpy(),
        dims=("y", "x"),
    )
    # Use plot_on_axis
    im = plot_on_axis(
        ax=ax,
        da=error_da,
        datastore=datastore,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
