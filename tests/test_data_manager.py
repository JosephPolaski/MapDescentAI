import numpy as np
import pytest

from unittest.mock import MagicMock, patch
from pathlib import Path
from data_management.data_manager import DataManager
from data_management.data_transfer_objects.split_data import SplitData
from data_management.enums.stored_data_type import StoredDataType
from utilities import constants

@pytest.fixture(scope="module")
def data_manager():
    return DataManager()

def make_test_directory(dir_name, files):
    directory_mock = MagicMock(spec=Path)
    directory_mock.return_value = True
    directory_mock.name = dir_name
    directory_mock.iterdir.return_value = files
    return directory_mock

def make_test_file(filename):
    file_mock = MagicMock(spec=Path)
    file_mock.is_dir.return_value = False
    file_mock.suffix = constants.JPEG_EXT
    file_mock.name = filename
    return file_mock

@patch("utilities.paths.DATA_DIR")
def test_get_image_paths_with_labels(mock_data_dir, data_manager):
    mock_file_1 = make_test_file("test_imgage1.jpg")
    mock_file_2 = make_test_file("test_imgage1.jpg")
    mock_file_3 = make_test_file("test_imgage1.jpg")

    test_files = [mock_file_1, mock_file_2, mock_file_3]

    mock_subdirectory = make_test_directory("TestClass", test_files)

    # set iterdir return value for patched in directory
    mock_data_dir.iterdir.return_value = [mock_subdirectory]

    result = data_manager.get_image_paths_with_labels()

    assert "TestClass" in result
    assert len(result["TestClass"]) == 3
    assert result["TestClass"][0].name == "test_imgage1.jpg"

def test_split_data(data_manager):

    labels = np.array([1,1,1,0,0,0]) # 6 total samples, 3 of each class                    
    features = np.random.rand(6, 2)  # 6 rows, 2 feature columns

    split_data = data_manager.split_dataset(labels, features)

    # 80% for training set (with stratification enabled)
    assert split_data.labels_train.shape[0] == 4
    assert split_data.features_train.shape[0] == 4 # 2 of each class 1 and class 0

    # 20% for test set (with stratification enabled)
    assert split_data.labels_test.shape[0] == 2
    assert split_data.features_test.shape[0] == 2 # 1 of each class 1 and class 0

@patch("utilities.file_helpers.try_create_directory")
@patch("numpy.savez_compressed")
def test_store_data_locally(mock_savez, mock_create, data_manager, tmp_path):   
    labels = np.array([0, 1])
    features = np.random.rand(2, 2)
    split = SplitData(labels, labels, features, features)

    # save with temp directory patched in (no real files written)
    with patch("utilities.paths.STORED_DATA_DIR", tmp_path):
        is_stored = data_manager.store_data_locally(split, StoredDataType.DATASET)

    assert is_stored is True
    mock_create.assert_called_once()
    assert mock_savez.called

@patch("utilities.file_helpers.get_most_recent_dataset_filename")
@patch("numpy.load")
def test_load_stored_data_dataset(mock_load, mock_getfile, data_manager, tmp_path):
    test_npz = {
        "labels_train": np.array([0]),
        "labels_test": np.array([1]),
        "features_train": np.array([[1,2]]),
        "features_test": np.array([[3,4]]),
    }

    # set return values for patches
    mock_load.return_value = test_npz
    mock_getfile.return_value = tmp_path / "test_stored_data.npz"

    result = data_manager.load_stored_data(StoredDataType.DATASET)

    assert result.labels_train.shape == (1,)
    assert result.features_train.shape == (1, 2)