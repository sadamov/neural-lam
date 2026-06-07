# Third-party
import pytest

# First-party
from neural_lam.datastore.mdp import MDPDatastore
from tests.conftest import init_datastore_example
from tests.test_training import run_simple_training

STATE_ONLY_CONFIG = (
    "tests/datastore_examples/mdp/danra_100m_winds/state_only.datastore.yaml"
)


def test_state_only_datastore_emits_warning():
    """Test that a state-only datastore (no static features) emits a
    UserWarning instead of raising an error."""
    with pytest.warns(UserWarning, match="No static features"):
        datastore = MDPDatastore(config_path=STATE_ONLY_CONFIG)

    assert datastore is not None


def test_state_only_datastore_static_returns_empty():
    """Test that get_vars_names returns [] for static in a state-only
    datastore, consistent with how missing forcing is handled."""
    with pytest.warns(UserWarning):
        datastore = MDPDatastore(config_path=STATE_ONLY_CONFIG)

    # Static should be empty - this is the core behaviour being fixed
    assert datastore.get_vars_names("static") == []

    # State variables should still be present and non-empty
    assert len(datastore.get_vars_names("state")) > 0


def test_state_only_datastore_forcing_returns_empty():
    """Test that get_vars_names also returns [] for forcing in a
    state-only datastore (no forcing in config)."""
    with pytest.warns(UserWarning):
        datastore = MDPDatastore(config_path=STATE_ONLY_CONFIG)

    assert datastore.get_vars_names("forcing") == []


@pytest.mark.parametrize("missing_category", ["static", "forcing"])
def test_state_only_datastore_all_entry_points_handle_missing(missing_category):
    """All MDPDatastore entry points that take a category argument should
    treat missing optional categories (static, forcing) consistently:
    list accessors return [], get_dataarray returns None. Catches the case
    where only get_vars_names was patched in the original PR."""
    with pytest.warns(UserWarning):
        datastore = MDPDatastore(config_path=STATE_ONLY_CONFIG)

    assert datastore.get_vars_names(missing_category) == []
    assert datastore.get_vars_units(missing_category) == []
    assert datastore.get_vars_long_names(missing_category) == []
    assert datastore.get_num_data_vars(missing_category) == 0
    assert datastore.get_dataarray(missing_category, split="train") is None


def test_state_only_datastore_training_setup_runs():
    """Run the shared small training setup against the MDP datastore."""
    datastore = init_datastore_example(MDPDatastore.SHORT_NAME)
    run_simple_training(datastore, set_output_std=False)
