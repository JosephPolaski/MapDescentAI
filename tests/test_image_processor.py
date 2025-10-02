import pytest
import numpy as np

from data_management.image_processor import ImageProcessor

@pytest.fixture(scope="module")
def image_processor():
    processor = ImageProcessor()
    processor.preprocess_image_data()
    return processor

def test_preprocess_image_shape(image_processor):
    features_shape = image_processor.features.shape
    assert features_shape == (27000, 64, 64, 3)

def test_preprocess_labels_shape(image_processor):
    labels_shape = image_processor.labels.shape
    assert labels_shape == (27000, 10)

def test_label_feature_length_match(image_processor):
    assert image_processor.labels.shape[0] == image_processor.features.shape[0]

def test_one_hot_label_correctness(image_processor):
    equals_one_truth_values = [sum(label) == 1 for label in image_processor.labels]
    assert all(equals_one_truth_values)

def test_data_type_features(image_processor):
    assert image_processor.features.dtype in [np.uint8, np.float32]

def test_data_type_labels(image_processor):
    assert image_processor.labels.dtype in [np.uint8]
