# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, datastore: BaseRegularGridDatastore, title=None):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
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


def find_closest_boundary_time(da_boundary_forcing, target_time):
    """Find the boundary forcing time closest to the target time.

    Parameters
    ----------
    da_boundary_forcing : xarray.DataArray
        DataArray containing boundary forcing data with a 'window' dimension
    target_time : numpy.datetime64
        Target time to find closest match for

    Returns
    -------
    xarray.DataArray
        Boundary forcing data at the closest time
    """
    window_times = da_boundary_forcing.window_time_deltas
    closest_idx = abs(window_times - target_time).argmin()
    return da_boundary_forcing.isel(window=closest_idx)


def plot_on_axis(
    ax,
    da,
    datastore,
    boundary_da=None,
    boundary_datastore=None,
    obs_mask=None,
    vmin=None,
    vmax=None,
    ax_title=None,
    cmap="plasma",
    grid_limits=None,
):
    """
    Plot weather state on given axis with optional boundary data
    """
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
    da_prediction: xr.DataArray = None,
    da_target: xr.DataArray = None,
    da_boundary: xr.DataArray = None,
    boundary_datastore: BaseRegularGridDatastore = None,
    boundary_var_map: dict = None,  # Add mapping parameter
    state_var_idx: int = None,  # Add current variable index
    title=None,
    vrange=None,
):
    """
    Plot example prediction and ground truth with optional boundary data.

    Parameters
    ----------
    boundary_var_map : dict
        Mapping from interior variable names to boundary variable names
    state_var_idx : int
        Index of the current state variable being plotted
    """
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
    error, datastore: BaseRegularGridDatastore, title=None, vrange=None
):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
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
