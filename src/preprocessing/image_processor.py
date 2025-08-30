import cv2
import numpy as np
import utilities.paths as paths

from utilities import MPLog
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, List

class ImageProcessor:      

    def __init__(self):
        self.JPEG_EXT = ".jpg" 
        self.feature_matrix : NDArray = None
        self.label_vector : NDArray = np.empty((0,1))
        self.logger = MPLog()
        self.image_count = 0
        self.feature_count = 0

    def __preallocate_feature_matrix_shape(self, image_shape):
        if self.feature_count == 0:
            self.feature_count = np.prod(image_shape)
            self.feature_matrix = np.zeros((self.image_count, self.feature_count))

    def flatten_image_data(self, images_with_labels : Dict[str, Path]):
        self.logger.method_entry()

        try:
            for label, image_path_list in images_with_labels.items():               
                for i, image_path in enumerate(image_path_list):

                    label_row = np.array([[label]])
                    self.label_vector = np.vstack((self.label_vector, label_row))

                    image = cv2.imread(image_path)
                    self.__preallocate_feature_matrix_shape(image.shape)                   

                    self.feature_matrix[i, :] = image.reshape(-1)

        except Exception as ex:
            self.logger.error(f"Failed to flatten image data: \n\n {ex} \n\n")
        

    def get_image_paths_with_labels(self) -> Dict[str, List[Path]]:
        self.logger.method_entry()
        
        images_with_labels : Dict[str, List[Path]] = {}

        for subdirectory in paths.DATA_DIR.iterdir():
            if subdirectory.is_dir():

                label = subdirectory.name             
                label_exists = label in images_with_labels.keys()

                image_paths = [path for path in subdirectory.iterdir() if path.suffix.lower() == self.JPEG_EXT]
                self.image_count += len(image_paths)

                if(label_exists):
                    images_with_labels[label] += image_paths
                    continue

                images_with_labels[label] = image_paths

        return images_with_labels

                    
                