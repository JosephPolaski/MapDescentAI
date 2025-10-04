import torch
import numpy as np
import sys
import utilities.paths as paths
import utilities.file_helpers as file_helpers

from torch import Tensor
from data_management.data_transfer_objects.split_data import SplitData
from data_management.data_transfer_objects.model_parameters import ModelParameters
from data_management.enums.stored_data_type import StoredDataType
from datetime import datetime
from pathlib import Path
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
    def split_dataset(features : Tensor, labels : Tensor, test_size=0.2, seed=None) -> SplitData:
        logger.info("Randomly splitting data into training (80%) and testing (20%) sets")              

        if seed is not None:
            torch.manual_seed(seed)

        feature_count = features.shape[0]
        indexes = torch.randperm(feature_count)
        split_index = int(feature_count * (1 - test_size))

        train_index = indexes[:split_index]
        test_index = indexes[split_index:]       

        split_data = SplitData(
            labels_train = labels[train_index],
            labels_test = labels[test_index],
            features_train = features[train_index],
            features_test = features[test_index]
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
    def __save_stored_data(filepath : Path, split_data : SplitData, data_type: StoredDataType, parameters : ModelParameters):
        if(data_type == StoredDataType.DATASET):            
                np.savez_compressed(
                    filepath, 
                    labels_train = split_data.labels_train.cpu().numpy(), 
                    labels_test = split_data.labels_test.cpu().numpy(),
                    features_train = split_data.features_train.cpu().numpy(),
                    features_test = split_data.features_test.cpu().numpy())
                
        if(data_type == StoredDataType.PARAMETERS):
            np.savez_compressed(
                filepath, 
                weights = parameters.weights, 
                bias = parameters.bias,
                loss_history = parameters.loss_history,
                number_of_classes = parameters.number_of_classes)

    @staticmethod
    def __unpack_stored_data(data_type : StoredDataType, stored_data) -> SplitData | ModelParameters:

        if(data_type == StoredDataType.DATASET):  
            # load data to gpu if available             
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

            return SplitData(
                labels_test = torch.from_numpy(stored_data["labels_test"]).to(device),
                labels_train = torch.from_numpy(stored_data["labels_train"]).to(device),
                features_test = torch.from_numpy(stored_data["features_test"]).to(device),
                features_train = torch.from_numpy(stored_data["features_train"]).to(device)
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