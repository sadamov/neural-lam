# Standard library
import os
import tempfile
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import xarray as xr

# First-party
from neural_lam.config import NeuralLAMConfig
from neural_lam.datastore import init_datastore
from neural_lam.vis import plot_prediction, plot_spatial_error


@pytest.fixture
def config_and_datastores():
    """Load config and initialize datastores for testing."""
    config_path = (
        Path(__file__).parent
        / "datastore_examples"
        / "mdp"
        / "era5_1000hPa_danra_100m_winds"
        / "config.yaml"
    )
    config = NeuralLAMConfig.from_yaml_file(config_path)

    # Initialize datastores using paths relative to config file
    config_dir = config_path.parent
    datastore = init_datastore(
        config.datastore.kind,
        config_dir / config.datastore.config_path,
    )
    datastore_boundary = init_datastore(
        config.datastore_boundary.kind,
        config_dir / config.datastore_boundary.config_path,
    )
    return config, datastore, datastore_boundary


def test_plot_prediction_with_real_config(config_and_datastores):
    """Test plotting with real config and datastores."""
    config, datastore, datastore_boundary = config_and_datastores

    # Get grid dimensions
    grid_shape = datastore.grid_shape_state
    boundary_shape = datastore_boundary.grid_shape_state

    # Create sample data matching the grid shape - remove extra dimension
    prediction_data = np.random.normal(0, 5, (grid_shape.x, grid_shape.y))
    target_data = np.random.normal(0, 5, (grid_shape.x, grid_shape.y))
    boundary_data = np.random.normal(0, 5, (boundary_shape.x, boundary_shape.y))

    # Create DataArrays with proper x/y coordinates - without unnecessary third dimension
    da_prediction = xr.DataArray(
        prediction_data,
        dims=["x", "y"],
        coords={
            "x": range(grid_shape.x),
            "y": range(grid_shape.y),
        },
    )

    da_target = xr.DataArray(
        target_data,
        dims=["x", "y"],
        coords={
            "x": range(grid_shape.x),
            "y": range(grid_shape.y),
        },
    )

    da_boundary = xr.DataArray(
        boundary_data,
        dims=["x", "y"],
        coords={
            "x": range(boundary_shape.x),
            "y": range(boundary_shape.y),
        },
    )

    # Test plotting with realistic value ranges
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig = plot_prediction(
            datastore=datastore,
            title="Wind Speed Test Plot",
            vrange=(-15, 15),
            da_prediction=da_prediction,
            da_target=da_target,
            da_boundary=da_boundary,
            boundary_datastore=datastore_boundary,
            boundary_var_map=config.datastore_boundary.variable_mapping,
            state_var_idx=0,
        )

        # Verify plot components
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # At least prediction and target subplots

        # Test if projections are properly set
        for ax in fig.axes:
            if hasattr(ax, "projection"):  # Check if axis has projection
                assert (
                    ax.projection.__class__.__name__.lower()
                    == datastore.coords_projection.__class__.__name__.lower()
                )

        # Save and verify
        save_path = os.path.join(tmp_dir, "test_wind_plot.png")
        fig.savefig(save_path)
        assert os.path.exists(save_path)
        plt.close(fig)


def test_plot_spatial_error_with_real_config(config_and_datastores):
    """Test spatial error plotting with real config."""
    _, datastore, _ = config_and_datastores

    grid_points = datastore.num_grid_points
    error_data = np.random.normal(0, 2, grid_points)
    error = torch.tensor(error_data, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fig = plot_spatial_error(
            error=error,
            datastore=datastore,
            title="Wind Speed Error Plot",
        )

        # Verify plot
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Main plot + colorbar axes
        assert (
            fig.axes[0].projection.__class__.__name__.lower()
            == datastore.coords_projection.__class__.__name__.lower()
        )
        # Save and verify
        save_path = os.path.join(tmp_dir, "test_wind_error.png")
        fig.savefig(save_path)
        assert os.path.exists(save_path)
        plt.close(fig)


def test_data_validation(config_and_datastores):
    """Test input data validation for plotting."""
    _, datastore, datastore_boundary = config_and_datastores

    # Test with mismatched grid points
    with pytest.raises(ValueError):
        wrong_size_data = np.random.randn(10)  # Wrong number of grid points
        da_wrong = xr.DataArray(
            wrong_size_data,
            dims=["grid_index"],
            coords={"grid_index": range(10)},
        )
        plot_prediction(
            datastore=datastore,
            title="Invalid Data Plot",
            vrange=(-2, 2),
            da_prediction=da_wrong,
            da_target=da_wrong,
            da_boundary=None,
            boundary_datastore=datastore_boundary,
            boundary_var_map={},
            state_var_idx=0,
        )


def test_coordinate_systems(config_and_datastores):
    """Test if coordinate systems are properly handled."""
    _, datastore, datastore_boundary = config_and_datastores

    # Create test data with proper grid shapes
    grid_shape = datastore.grid_shape_state
    boundary_shape = datastore_boundary.grid_shape_state

    # Create 2D arrays matching grid shapes
    da_prediction = xr.DataArray(
        np.random.randn(grid_shape.x, grid_shape.y),
        dims=["x", "y"],
        coords={
            "x": range(grid_shape.x),
            "y": range(grid_shape.y),
        },
    )

    da_target = xr.DataArray(
        np.random.randn(grid_shape.x, grid_shape.y),
        dims=["x", "y"],
        coords={
            "x": range(grid_shape.x),
            "y": range(grid_shape.y),
        },
    )

    da_boundary = xr.DataArray(
        np.random.randn(boundary_shape.x, boundary_shape.y),
        dims=["x", "y"],
        coords={
            "x": range(boundary_shape.x),
            "y": range(boundary_shape.y),
        },
    )
    # Test if plotting handles different projections
    fig = plot_prediction(
        datastore=datastore,
        title="Projection Test",
        vrange=(-2, 2),
        da_prediction=da_prediction,
        da_target=da_target,
        da_boundary=da_boundary,
        boundary_datastore=datastore_boundary,
        boundary_var_map={},
        state_var_idx=0,
    )

    plt.close(fig)


def test_variable_mapping(config_and_datastores):
    """Test if variable mapping between ERA5 and DANRA works correctly."""
    config, datastore, datastore_boundary = config_and_datastores

    # Get grid dimensions
    grid_shape = datastore.grid_shape_state
    boundary_shape = datastore_boundary.grid_shape_state

    # Create 3D arrays to include state variables
    prediction_data = np.random.normal(0, 5, (2, grid_shape.x, grid_shape.y))
    target_data = np.random.normal(0, 5, (2, grid_shape.x, grid_shape.y))
    boundary_data = np.random.normal(
        0, 5, (2, boundary_shape.x, boundary_shape.y)
    )

    # Create DataArrays with named coordinates
    da_prediction = xr.DataArray(
        prediction_data,
        dims=["state", "x", "y"],
        coords={
            "state": ["u100m", "v100m"],
            "x": range(grid_shape.x),
            "y": range(grid_shape.y),
        },
    )

    da_target = xr.DataArray(
        target_data,
        dims=["state", "x", "y"],
        coords={
            "state": ["u100m", "v100m"],
            "x": range(grid_shape.x),
            "y": range(grid_shape.y),
        },
    )

    da_boundary = xr.DataArray(
        boundary_data,
        dims=["state", "x", "y"],
        coords={
            "state": [
                "u_component_of_wind1000hPa",
                "v_component_of_wind1000hPa",
            ],
            "x": range(boundary_shape.x),
            "y": range(boundary_shape.y),
        },
    )

    # Test u component
    da_prediction_u = da_prediction.sel(state="u100m")
    da_target_u = da_target.sel(state="u100m")
    da_boundary_u = da_boundary.sel(state="u_component_of_wind1000hPa")

    # Update naming to match what plot_prediction expects
    da_prediction_u.name = "u100m"  # Simplified names
    da_target_u.name = "u100m"
    da_boundary_u.name = "u_component_of_wind1000hPa"

    fig = plot_prediction(
        datastore=datastore,
        title="u100m Wind Component",
        vrange=(-15, 15),
        da_prediction=da_prediction_u,
        da_target=da_target_u,
        da_boundary=da_boundary_u,
        boundary_datastore=datastore_boundary,
        boundary_var_map=config.datastore_boundary.variable_mapping,
        state_var_idx=0,
    )

    # Get all titles including figure suptitle
    axes_titles = [
        ax.get_title() for ax in fig.axes if hasattr(ax, "get_title")
    ]
    suptitle = fig._suptitle.get_text() if fig._suptitle else ""
    all_titles = axes_titles + [suptitle]

    print("All titles:", all_titles)

    # First verify we have titles
    assert len(all_titles) > 0, "No titles found"

    # Look for either "u100m" or "u component" in any of the titles
    assert any(
        ("u100m" in title.lower() or "u component" in title.lower())
        for title in all_titles
    ), f"Expected u wind component title not found in: {all_titles}"

    plt.close(fig)
