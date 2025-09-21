import numpy as np
import pytest

from unittest.mock import patch, MagicMock
from data_management.map_descent_dataset import MapDescentDataset
from data_management.data_transfer_objects.split_data import SplitData

@pytest.fixture(scope="module")
def test_split_data():
    labels = np.array([0, 1])
    features = np.random.rand(2, 2)

    return SplitData(
        labels_train=labels,
        labels_test=labels,
        features_train=features,
        features_test=features
    )

@patch("data_management.map_descent_dataset.file_helpers.get_most_recent_dataset_filename")
def test_stored_data_does_exist(mock_getfile):
    mock_file = MagicMock()
    mock_file.name = "test_data.npz"

    # set return value for patch
    mock_getfile.return_value = mock_file

    dataset = MapDescentDataset.__new__(MapDescentDataset)  # bypass all initialization logic in constructor
    dataset.logger = MagicMock()
    dataset.stored_data_filename = ""

    result = dataset._MapDescentDataset__check_if_stored_data_exists()

    assert result is True
    assert dataset.stored_data_filename == "test_data.npz"

@patch("data_management.map_descent_dataset.file_helpers.get_most_recent_dataset_filename")
def test_stored_data_does_not_exist(mock_getfile):
    mock_getfile.return_value = None # seet patch return value

    dataset = MapDescentDataset.__new__(MapDescentDataset) # bypass all initialization logic in constructor
    dataset.logger = MagicMock()
    dataset.stored_data_filename = "test_data.npz"

    result = dataset._MapDescentDataset__check_if_stored_data_exists()

    assert result is False
    assert dataset.stored_data_filename == "test_data.npz"

@patch("data_management.map_descent_dataset.file_helpers.get_most_recent_dataset_filename")
@patch("data_management.map_descent_dataset.DataManager.load_stored_data")
def test_try_fetch_stored_data_success(mock_load, mock_getfile, test_split_data):
    
    # Mock stored data file
    mock_file = MagicMock()
    mock_file.name = "test_data.npz"

    # set patch return values
    mock_getfile.return_value = mock_file
    mock_load.return_value = test_split_data

    dataset = MapDescentDataset.__new__(MapDescentDataset) # bypass all initialization logic in constructor

    # mock logger and initialize data members
    dataset.logger = MagicMock()
    dataset.labels_train = None
    dataset.labels_test = None
    dataset.features_train = None
    dataset.features_test = None

    # Mock DataManager with split data fixture
    dataset.data_manager = MagicMock()
    dataset.data_manager.load_stored_data.return_value = test_split_data

    result = dataset._MapDescentDataset__try_fetch_stored_data()

    assert result is True
    assert dataset.labels_train is not None
    assert dataset.features_train is not None

@patch("data_management.map_descent_dataset.file_helpers.get_most_recent_dataset_filename")
@patch("data_management.map_descent_dataset.DataManager.load_stored_data")
def test_try_fetch_stored_data_fail(mock_load, mock_getfile):
    
    # mock null stored data file
    mock_file = MagicMock()
    mock_file.name = "test_data.npz"
    mock_getfile.return_value = mock_file
    mock_load.return_value = None

    dataset = MapDescentDataset.__new__(MapDescentDataset) # bypass all initialization logic in constructor

    # mock logger and data manager with failed load method
    dataset.logger = MagicMock()
    dataset.data_manager = MagicMock()
    dataset.data_manager.load_stored_data.return_value = None

    with pytest.raises(ValueError):
        dataset._MapDescentDataset__try_fetch_stored_data()

@patch("data_management.map_descent_dataset.ImageProcessor")
@patch("data_management.map_descent_dataset.MapDescentDataset._MapDescentDataset__try_fetch_stored_data")
def test_initialize_data_calls_preprocess(mock_try_fetch, mock_image_processor):
    # failed __try_fetch_stored_data method patch
    mock_try_fetch.return_value = False

    # mock image preprocessor class
    mock_img_processor_instance = MagicMock()
    mock_img_processor_instance.preprocess_image_data = MagicMock()

    mock_image_processor.return_value = mock_img_processor_instance

    dataset = MapDescentDataset.__new__(MapDescentDataset) # bypass all initialization logic in constructor
    dataset.logger = MagicMock()
    dataset.data_manager = MagicMock()

    # bypass Python name mangling to call / mock necessarry methods
    dataset._MapDescentDataset__preprocess_and_fetch = MagicMock()
    dataset._MapDescentDataset__initialize_data()
    dataset._MapDescentDataset__preprocess_and_fetch.assert_called_once()