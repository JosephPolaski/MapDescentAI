import numpy as np
import sys
import utilities.paths as paths
import utilities.file_helpers as file_helpers

from data_management.data_transfer_objects.split_data import SplitData
from data_management.data_transfer_objects.model_parameters import ModelParameters
from data_management.enums.stored_data_type import StoredDataType
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List
from utilities import constants
from utilities.md_log import MDLog

logger = MDLog()
  
class DataManager:    

    @staticmethod 
    def get_image_paths_with_labels() -> Dict[str, List[Path]]:      
        try:
            logger.method_entry()
            images_with_labels : Dict[str, List[Path]] = {}

            for subdirectory in paths.DATA_DIR.iterdir():

                if not subdirectory.is_dir():
                    continue                       
               
                image_paths = [path for path in subdirectory.iterdir() if path.suffix.lower() == constants.JPEG_EXT]                
                images_with_labels.setdefault(subdirectory.name, []).extend(image_paths)                                

            return images_with_labels

        except Exception as ex:
            logger(f"Failed to get image paths with labels: \n\n {ex} \n\n")      

    @staticmethod
    def split_dataset(labels, features) -> SplitData:
        logger.info("Randomly splitting data into training (80%) and testing (20%) sets")              

        train_labels, test_labels, train_features, test_features = train_test_split(
            labels, 
            features,
            test_size=0.2,
            random_state=constants.RANDOM_SEED,
            stratify=labels
        )

        split_data = SplitData(
            labels_train = train_labels,
            labels_test = test_labels,
            features_train = train_features,
            features_test = test_features
        ) 

        return split_data       

    @staticmethod
    def store_data_locally(split_data : SplitData, data_type = StoredDataType.DATASET , parameters : ModelParameters = None) -> bool:
        try:
            file_helpers.try_create_directory(paths.STORED_DATA_DIR) 

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = paths.STORED_DATA_DIR / (f"MapDescentAI_{data_type.value}_" + timestamp + ".npz")
            
            DataManager.__save_stored_data(filepath, split_data, data_type, parameters)

            logger.info(f"Successfully stored {paths.STORED_DATA_DIR}{filepath}")
            return True 
                   
        except Exception as ex:
            logger.error(f"Failed to store data locally: \n\n {ex} \n\n")
            return False

    @staticmethod
    def load_stored_data(data_type = StoredDataType.DATASET) -> SplitData | ModelParameters | None:
        try:
            type_name = "dataset" if data_type == StoredDataType.DATASET else "parameters"
            most_recent_data : Path = file_helpers.get_most_recent_dataset_filename(data_type)

            if most_recent_data is None:
                return None

            stored_data = np.load(most_recent_data)
            logger.info(f"Loaded {type_name} from {most_recent_data}")

            return DataManager.__unpack_stored_data(data_type, stored_data)

        except Exception as ex:
            logger.error(f"Failed to load stored data: \n\n {ex} \n\n")    

    @staticmethod
    def __save_stored_data(filepath, split_data, data_type, parameters):
        if(data_type == StoredDataType.DATASET):            
                np.savez_compressed(
                    filepath, 
                    labels_train = split_data.labels_train, 
                    labels_test = split_data.labels_test,
                    features_train = split_data.features_train,
                    features_test = split_data.features_test)
                
        if(data_type == StoredDataType.PARAMETERS):
            np.savez_compressed(
                filepath, 
                weights = parameters.weights, 
                bias = parameters.bias,
                loss_history = parameters.loss_history,
                number_of_classes = parameters.number_of_classes)

    @staticmethod
    def __unpack_stored_data(data_type, stored_data) -> SplitData | ModelParameters:

        if(data_type == StoredDataType.DATASET):               
                
            return SplitData(
                labels_test = stored_data["labels_test"],
                labels_train = stored_data["labels_train"],
                features_test = stored_data["features_test"],
                features_train = stored_data["features_train"]
            )           
                
        if(data_type == StoredDataType.PARAMETERS):            

            return ModelParameters(
                weights = stored_data["weights"],
                bias = stored_data["bias"],
                loss_history = stored_data["loss_history"],
                number_of_classes = stored_data["number_of_classes"]
            )

if __name__=="__main__" :
    print("This module is not meant to be run as a standalone script...exiting..")
    sys.exit(0)