import cv2
import numpy as np
import sys

from data_management.data_manager import DataManager
from data_management.enums.stored_data_type import StoredDataType
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict
from utilities import constants
from utilities.md_log import MDLog

class ImageProcessor:      

    def __init__(self):
        self.data_manager = DataManager()           
        self.logger = MDLog()
        
        self.features : NDArray = None
        self.labels : None 

    def preprocess_image_data(self):
        try:
            self.logger.method_entry()

            images_with_labels : Dict[str, Path] = self.data_manager.get_image_paths_with_labels()  

            number_of_classes = len(images_with_labels)
            image_data = []
            labels_as_integers = []

            for label, image_path_list in images_with_labels.items():               
                for image_path in image_path_list: 
                      
                      normalized_image = cv2.imread(image_path).astype(np.float32) / 255.0
                      image_data.append(normalized_image)
                      labels_as_integers.append(constants.LABEL_TO_INDEX_MAP[label])

            self.features = np.stack(image_data, axis=0)
            self.labels =  np.eye(number_of_classes, dtype=np.uint8)[labels_as_integers]                   
            self.__split_and_store_data()
        except Exception as ex:
            self.logger.error(f"Failed to preprocess data: \n\n {ex} \n\n")    

    def __split_and_store_data(self):
        split_data = self.data_manager.split_dataset(self.labels, self.feature)
        self.data_manager.store_data_locally(split_data, StoredDataType.DATASET)

if __name__=="__main__" :
    print("This module is not meant to be run as a standalone script...exiting..")
    sys.exit(0)

        

                    
                