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
        
        self.features : NDArray = None
        self.labels : None 

    def build_image_data(self):
        try:
            self.logger.method_entry()

            images_with_labels : Dict[str, Path] = self.data_manager.get_image_paths_with_labels()  

            number_of_classes = len(images_with_labels)
            image_data = []
            labels_as_integers = []

            for label, image_path_list in images_with_labels.items():               
                for image_path in image_path_list: 
                      
                      image = cv2.imread(image_path)
                      image_data.append(image)
                      labels_as_integers.append(constants.LABEL_TO_INDEX_MAP[label])

            self.features = np.stack(image_data, axis=0)
            self.labels =  np.eye(number_of_classes, dtype=np.uint8)[labels_as_integers]                     
            print()
        except Exception as ex:
            self.logger.error(f"Failed to preprocess data: \n\n {ex} \n\n")    

if __name__=="__main__" :
    print("This module is not meant to be run as a standalone script...exiting..")
    sys.exit(0)

        

                    
                