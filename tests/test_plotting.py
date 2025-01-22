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
    PlotDataManager,  # Updated from PlotMetadata and Visualizer
)
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


@pytest.fixture
def plot_manager(config_and_datastores):  # Renamed from visualizer
    """Create a plot manager instance for testing."""
    _, datastore, datastore_boundary = config_and_datastores
    return PlotDataManager(
        interior_datastore=datastore,
        boundary_datastore=datastore_boundary,
        boundary_var_map={"u100m": "u_component_of_wind1000hPa"},
    )


def test_plot_data_manager(config_and_datastores):
    """Test PlotDataManager initialization and validation."""
    _, datastore, _ = config_and_datastores

    # Valid initialization with datastore
    plot_manager = PlotDataManager(interior_datastore=datastore)
    assert plot_manager.metadata_interior is not None
    assert "dims" in plot_manager.metadata_interior

    # Invalid datastore type
    with pytest.raises(TypeError):
        PlotDataManager(interior_datastore="not_a_datastore")

    # Invalid boundary var map type
    with pytest.raises(TypeError):
        PlotDataManager(
            interior_datastore=datastore, boundary_var_map="invalid"
        )


def test_visualizer_initialization(config_and_datastores):
    """Test Visualizer initialization and metadata extraction."""
    _, datastore, datastore_boundary = config_and_datastores

    # Valid initialization
    vis = PlotDataManager(interior_datastore=datastore)
    assert vis.metadata_interior is not None
    assert "grid_index" in vis.metadata_interior
    assert "feature_names" in vis.metadata_interior

    # Test with boundary datastore
    vis = PlotDataManager(
        interior_datastore=datastore,
        boundary_datastore=datastore_boundary,
        boundary_var_map={"u100m": "u_component_of_wind1000hPa"},
    )
    assert vis.metadata_boundary is not None


def test_visualizer_tensor_conversion(plot_manager):
    """Test tensor to DataArray conversion."""
    # Get correct grid size from interior datastore
    grid_points = plot_manager._interior_datastore.num_grid_points
    n_features = len(plot_manager._interior_datastore.get_vars_names("state"))

    # Test 2D tensor - needs expansion to match expected 3D format
    tensor_2d = torch.randn(grid_points, n_features)
    tensor = tensor_2d.unsqueeze(
        0
    )  # Add time dimension: (1, grid_points, n_features)
    times = torch.tensor([np.datetime64("2023-01-01").astype(np.int64)])

    # Valid conversion
    da = plot_manager.tensor_to_dataarray(tensor, times, "state")
    assert isinstance(da, xr.DataArray)
    assert "grid_index" in da.dims
    assert "state_feature" in da.dims
    assert "time" in da.dims

    # Test 3D tensor with multiple timesteps
    tensor_3d = torch.randn(2, grid_points, n_features)  # (time, grid, feature)
    times_3d = torch.tensor(
        [
            np.datetime64("2023-01-01").astype(np.int64),
            np.datetime64("2023-01-02").astype(np.int64),
        ]
    )
    da_3d = plot_manager.tensor_to_dataarray(tensor_3d, times_3d, "state")
    assert isinstance(da_3d, xr.DataArray)
    assert "time" in da_3d.dims
    assert len(da_3d.time) == 2
    assert "grid_index" in da_3d.dims
    assert "state_feature" in da_3d.dims

    # Invalid category
    with pytest.raises(ValueError):
        plot_manager.tensor_to_dataarray(tensor, times, "invalid")

    # Invalid tensor type
    with pytest.raises(TypeError):
        plot_manager.tensor_to_dataarray(
            np.zeros((1, grid_points, n_features)), times, "state"
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
