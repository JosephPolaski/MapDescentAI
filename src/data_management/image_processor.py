import cv2
import numpy as np
import sys
import torch

from torch import Tensor
from data_management.data_transfer_objects.feature_label_info import FeatureLabelInfo
from data_management.data_manager import DataManager
from data_management.enums.stored_data_type import StoredDataType
from pathlib import Path
from typing import Dict
from utilities import constants
from utilities.md_log import MDLog

class ImageProcessor:      

    def __init__(self):         
        self.logger = MDLog()
        
        self.features_tensor : Tensor = None
        self.labels_tensor : Tensor = None     

    def preprocess_image_data(self):
        try:
            self.logger.method_entry()

            images_with_labels : Dict[str, Path] = DataManager.get_image_paths_with_labels()             
           
            data : FeatureLabelInfo = self.__get_feature_label_data(images_with_labels)      

            self.__set_tensor_data(data)      

            self.__split_and_store_data()
        except Exception as ex:
            self.logger.error(f"Failed to preprocess data: \n\n {ex} \n\n")    

    def __get_feature_label_data(self, images_with_labels) -> FeatureLabelInfo:
        try:
            result = FeatureLabelInfo()

            for label, image_path_list in images_with_labels.items():               
                for image_path in image_path_list: 
                        
                    normalized_image = cv2.imread(image_path).astype(np.float32) / 255.0

                    # Convert shape to match pytorch (Channels, Height, Width)
                    normalized_image = np.transpose(normalized_image, (2,0,1))

                    result.features_raw.append(normalized_image)
                    result.labels.append(constants.LABEL_TO_INDEX_MAP[label])

            return result
        except Exception as ex:
            self.logger.error(f"Failed getting feature and label data: \n\n {ex} \n\n") 
    
    def __set_tensor_data(self, data : FeatureLabelInfo) -> None:
        features = np.stack(data.features_raw, axis=0)
        self.features_tensor = torch.tensor(features)

        labels = np.array(data.labels, dtype=np.int64)
        self.labels_tensor = torch.tensor(labels, dtype=torch.long) 

    def __split_and_store_data(self):
        split_data = DataManager.split_dataset(self.labels, self.feature)
        DataManager.store_data_locally(split_data, StoredDataType.DATASET)

if __name__=="__main__" :
    print("This module is not meant to be run as a standalone script...exiting..")
    sys.exit(0)

        

                    
                