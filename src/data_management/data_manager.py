import numpy as np
import sys
import utilities.paths as paths
import utilities.file_helpers as file_helpers

from data_management.data_transfer_objects.split_data import SplitData
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List
from utilities import constants
from utilities import MDLog

class DataManager:

    def __init__(self):           
        self.logger = MDLog()
        self.split_data : SplitData = None
        self.image_count : int = 0  

    def get_image_paths_with_labels(self) -> Dict[str, List[Path]]:      
        try:
            self.logger.method_entry()
            images_with_labels : Dict[str, List[Path]] = {}

            for subdirectory in paths.DATA_DIR.iterdir():

                if not subdirectory.is_dir():
                    continue

                label = subdirectory.name             
                label_exists = label in images_with_labels.keys()
                
                image_paths = [path for path in subdirectory.iterdir() if path.suffix.lower() == constants.JPEG_EXT]                    
                self.image_count += len(image_paths)

                if(label_exists):
                    images_with_labels[label] += image_paths
                else:
                    images_with_labels[label] = image_paths

            return images_with_labels

        except Exception as ex:
            self.logger(f"Failed to get image paths with labels: \n\n {ex} \n\n")      

    def split_dataset(self, label_vector, feature_matrix):
        self.logger.info("Randomly splitting data into training (80%) and testing (20%) sets")

        train_labels, test_labels, train_features, test_features = train_test_split(
            label_vector,
            feature_matrix,
            test_size=0.2,
            random_state=constants.RANDOM_SEED
        )

        self.split_data = SplitData() 
        self.split_data.labels_train = train_labels,
        self.split_data.labels_test = test_labels,
        self.split_data.features_train = train_features,
        self.split_data.features_test = test_features       

    def store_dataset_locally(self):
        try:
            file_helpers.try_create_directory(paths.STORED_DATA_DIR) 

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = paths.STORED_DATA_DIR / ("MapDescentAI_dataset_" + timestamp + ".npz")
            np.savez_compressed(
                filepath, 
                labels_train = self.split_data.labels_train, 
                labels_test = self.split_data.labels_test,
                features_train = self.split_data.features_train,
                features_test = self.split_data.features_test)

            self.logger.info(f"Successfully stored dataset {paths.STORED_DATA_DIR}{filepath}")            
        except Exception as ex:
            self.logger.error(f"Failed to store data locally: \n\n {ex} \n\n")

    def load_stored_data(self) -> SplitData | None:
        try:
            most_recent_dataset : Path = file_helpers.get_most_recent_dataset_filename()

            if most_recent_dataset is None:
                return None

            stored_data = np.load(most_recent_dataset)

            return SplitData(
                labels_test = stored_data["labels_test"],
                labels_train = stored_data["labels_train"],
                features_test = stored_data["features_test"],
                features_train = stored_data["features_train"]
            )

        except Exception as ex:
            self.logger.error(f"Failed to load stored data: \n\n {ex} \n\n")

if __name__=="__main__" :
    print("This module is not meant to be run as a standalone script...exiting..")
    sys.exit(0)