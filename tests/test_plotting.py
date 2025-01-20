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
from cartopy.mpl.geoaxes import GeoAxes

# First-party
from neural_lam.config import NeuralLAMConfig
from neural_lam.datastore import init_datastore
from neural_lam.vis import (
    PlotCoordinates,
    Visualizer,
    plot_prediction,
    plot_spatial_error,
)


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


@pytest.fixture
def visualizer(config_and_datastores):
    """Create a visualizer instance for testing."""
    _, datastore, datastore_boundary = config_and_datastores
    return Visualizer(
        interior_datastore=datastore,
        boundary_datastore=datastore_boundary,
        boundary_var_map={"u100m": "u_component_of_wind1000hPa"},
    )


def test_plot_coordinates():
    """Test PlotCoordinates validation."""
    grid_index = xr.DataArray([0, 1, 2])
    names = ["temp", "wind"]
    units = ["K", "m/s"]

    # Valid initialization
    coords = PlotCoordinates(grid_index, names, units)
    assert coords.grid_index.equals(grid_index)

    # Invalid grid_index type
    with pytest.raises(TypeError):
        PlotCoordinates([0, 1, 2], names, units)

    # Mismatched features/units
    with pytest.raises(ValueError):
        PlotCoordinates(grid_index, ["temp"], units)


def test_visualizer_initialization(config_and_datastores):
    """Test Visualizer initialization and coord extraction."""
    _, datastore, datastore_boundary = config_and_datastores

    # Valid initialization
    vis = Visualizer(datastore)
    assert vis.interior_coords is not None

    # Invalid datastore type
    with pytest.raises(TypeError):
        Visualizer("not_a_datastore")

    # Invalid boundary mapping
    with pytest.raises(TypeError):
        Visualizer(datastore, boundary_var_map="invalid")


def test_visualizer_tensor_conversion(visualizer):
    """Test tensor to DataArray conversion."""
    # Get correct grid size from interior datastore
    grid_points = visualizer._interior_datastore.num_grid_points
    n_features = len(visualizer._interior_datastore.get_vars_names("state"))

    # Create properly sized tensor
    tensor = torch.randn(grid_points, n_features)
    times = [np.datetime64("2023-01-01")]

    # Valid conversion
    da = visualizer.tensor_to_dataarray(tensor, times, "state")
    assert isinstance(da, xr.DataArray)
    assert "grid_index" in da.dims
    assert "state_feature" in da.dims

    # Invalid category
    with pytest.raises(ValueError):
        visualizer.tensor_to_dataarray(tensor, times, "invalid")

    # Invalid tensor type
    with pytest.raises(TypeError):
        visualizer.tensor_to_dataarray(
            np.zeros((grid_points, n_features)), times, "state"
        )


def test_plot_prediction_basic(config_and_datastores):
    """Test basic prediction plotting functionality."""
    _, datastore, datastore_boundary = config_and_datastores

    # Create minimal test data
    grid_shape = datastore.grid_shape_state
    da_prediction = xr.DataArray(
        np.random.randn(grid_shape.x, grid_shape.y), dims=["x", "y"]
    )
    da_target = da_prediction.copy()

    # Test basic plotting works
    fig = plot_prediction(
        datastore=datastore, da_prediction=da_prediction, da_target=da_target
    )
    assert isinstance(fig, plt.Figure)
    # Expect 4 axes total: 2 plots + 2 colorbars
    assert len(fig.axes) == 4

    # Check we have two main plot axes with correct titles
    # Only count GeoAxes, not colorbar Axes
    main_axes = [ax for ax in fig.axes if isinstance(ax, GeoAxes)]
    assert len(main_axes) == 2
    assert "Ground Truth" in main_axes[0].get_title()
    assert "Prediction" in main_axes[1].get_title()
    plt.close(fig)


def test_plot_spatial_error_basic(config_and_datastores):
    """Test basic spatial error plotting."""
    _, datastore, _ = config_and_datastores

    error = torch.randn(datastore.num_grid_points)
    fig = plot_spatial_error(error=error, datastore=datastore)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Main plot + colorbar
    plt.close(fig)


def test_save_plots(config_and_datastores):
    """Test plots can be saved."""
    _, datastore, _ = config_and_datastores
    grid_shape = datastore.grid_shape_state

    da_prediction = xr.DataArray(
        np.random.randn(grid_shape.x, grid_shape.y), dims=["x", "y"]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test prediction plot save
        fig1 = plot_prediction(
            datastore=datastore,
            da_prediction=da_prediction,
            da_target=da_prediction,
        )
        save_path1 = os.path.join(tmp_dir, "pred_test.png")
        fig1.savefig(save_path1)
        assert os.path.exists(save_path1)
        plt.close(fig1)

        # Test error plot save
        error = torch.randn(datastore.num_grid_points)
        fig2 = plot_spatial_error(error=error, datastore=datastore)
        save_path2 = os.path.join(tmp_dir, "error_test.png")
        fig2.savefig(save_path2)  # Add missing save call
        assert os.path.exists(save_path2)
        plt.close(fig2)
