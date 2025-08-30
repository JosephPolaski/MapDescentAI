import cv2
import numpy as np
import utilities.paths as paths

from sklearn.model_selection import train_test_split
from utilities import MPLog
from utilities import constants
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, List, Tuple

class ImageProcessor:      

    def __init__(self):            
        self.logger = MPLog()
        self.image_count = 0
        self.feature_count = 0
        self.feature_matrix : NDArray = None
        self.label_vector : None

    def __preallocate_shapes(self, image_shape):
        self.label_vector = np.zeros((self.image_count, 1), dtype=int)
        if self.feature_count == 0:
            self.feature_count = np.prod(image_shape)
            self.feature_matrix = np.zeros((self.image_count, self.feature_count))      

    def __get_image_paths_with_labels(self) -> Dict[str, List[Path]]:
        self.logger.method_entry()

        try:

            images_with_labels : Dict[str, List[Path]] = {}

            for subdirectory in paths.DATA_DIR.iterdir():
                if subdirectory.is_dir():

                    label = subdirectory.name             
                    label_exists = label in images_with_labels.keys()

                    image_paths = [path for path in subdirectory.iterdir() if path.suffix.lower() == constants.JPEG_EXT]
                    self.image_count += len(image_paths)

                    if(label_exists):
                        images_with_labels[label] += image_paths
                        continue

                    images_with_labels[label] = image_paths

            return images_with_labels

        except Exception as ex:

            self.logger(f"Failed to get image paths with lablels: \n\n {ex} \n\n")

    def get_flattened_image_data(self) -> Tuple[NDArray, NDArray]:
        self.logger.method_entry()

        try:
            images_with_labels : Dict[str, Path] = self.__get_image_paths_with_labels

            total_count = 0
            for label, image_path_list in images_with_labels.items():               
                for i, image_path in enumerate(image_path_list):                 

                    image = cv2.imread(image_path)
                    self.__preallocate_shapes(image.shape)                   

                    self.label_vector[total_count] = constants.LABEL_TO_INDEX_MAP[label]
                    self.feature_matrix[total_count, :] = image.reshape(-1)

                    total_count += 1
                    if total_count % 1000 == 0:
                        self.logger.info(f"Completed flattening of {total_count} images.")                    
                    
            self.logger.info("Normalizing image rgb values")
            self.feature_matrix /= 255

            self.logger.info(f"Image flattening complete...")  
            return self.label_vector, self.feature_matrix          

        except Exception as ex:
            self.logger.error(f"Failed to flatten image data: \n\n {ex} \n\n")

        

                    
                