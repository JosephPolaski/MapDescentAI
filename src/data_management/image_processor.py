import cv2
import numpy as np
import sys

from data_management.data_manager import DataManager
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict
from utilities import constants
from utilities.md_log import MDLog

class ImageProcessor:      

    def __init__(self):
        self.data_manager = DataManager()           
        self.logger = MDLog()

        self.feature_count = 0
        self.feature_matrix : NDArray = None
        self.label_vector : None
        self.is_preallocated = False

    def __flatten_and_normalize_image_data(self, images_with_labels : Dict[str, Path]):
        total_count = 0

        for label, image_path_list in images_with_labels.items():               
            for image_path in image_path_list:                 

                image = cv2.imread(image_path)
                self.__preallocate_shapes(image.shape)                   

                self.label_vector[total_count] = constants.LABEL_TO_INDEX_MAP[label]
                self.feature_matrix[total_count, :] = image.reshape(-1)

                total_count += 1
                if total_count % 1000 == 0:
                    self.logger.info(f"Completed flattening of {total_count} images.")

        self.feature_matrix /= 255
        self.logger.info("Normalized image BGR values")       
        self.logger.info("Image flattening complete...") 

    def __preallocate_shapes(self, image_shape):      
        if not self.is_preallocated:
            self.label_vector = np.zeros((self.data_manager.image_count,), dtype=int)
            self.feature_count = np.prod(image_shape)
            self.feature_matrix = np.zeros((self.data_manager.image_count, self.feature_count))    
            self.is_preallocated = True  

    def build_flattened_image_data(self) -> None:       
        try:
            self.logger.method_entry()

            images_with_labels : Dict[str, Path] = self.data_manager.get_image_paths_with_labels()           

            self.__flatten_and_normalize_image_data(images_with_labels)                
            self.data_manager.split_dataset(self.label_vector, self.feature_matrix)
            self.data_manager.store_data_locally()       

        except Exception as ex:
            self.logger.error(f"Failed to flatten image data: \n\n {ex} \n\n")

if __name__=="__main__" :
    print("This module is not meant to be run as a standalone script...exiting..")
    sys.exit(0)

        

                    
                