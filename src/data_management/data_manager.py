import cv2
import numpy as np
import utilities.paths as paths
import utilities.file_helpers as file_helpers

from data_transfer_objects import SplitData
from sklearn.model_selection import train_test_split
from datetime import datetime
from utilities import MPLog
from utilities import constants
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, List

class DataManager:

    def __init__(self, label_vector, feature_matrix):
        self.JPEG_EXT = ".jpg"         
        self.logger = MPLog()
        self.split_data : SplitData = self.__split_dataset(label_vector, feature_matrix)

    def __split_dataset(self, label_vector, feature_matrix):
        self.logger.info("Randomly splitting data into training (80%) and testing (20%) sets")

        labels_train, labels_test, features_train, features_test = train_test_split(
            label_vector,
            feature_matrix,
            test_size=0.2,
            random_state=constants.RANDOM_SEED
        )

        self.split_data = SplitData(
            labels_train,
            labels_test,
            features_train,
            features_test
        )        

    def store_image_dataset_locally(self):
        try:
            file_helpers.try_create_directory(paths.STORED_DATA_DIR) 

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = paths.STORED_DATA_DIR / (f"MapDescentAI_dataset_" + timestamp + ".npz")
            np.savez_compressed(
                filename, 
                labels_train = self.split_data.labels_train, 
                labels_test = self.split_data.labels_test,
                features_train = self.split_data.features_train,
                features_test = self.split_data.features_test)

            self.logger.info(f"Successfully stored dataset {paths.STORED_DATA_DIR}{filename}")
        except Exception as ex:
            self.logger.error(f"Failed to store data locally: \n\n {ex} \n\n")