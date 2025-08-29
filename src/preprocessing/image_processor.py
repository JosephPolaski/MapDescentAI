import cv2
import numpy as np
import src.mapdescentai_paths as paths

from numpy.typing import NDArray
from pathlib import Path
from typing import Dict

class ImageProcessor:      

    def __init__(self):
        self.JPEG_EXT = ".jpg" 
        self.feature_matrix : NDArray = None
        self.label_vector : NDArray = None

    def flatten_image_data(self):
        pass

    def get_image_paths_with_labels(self) -> Dict[str, Path]:
        images_with_labels = {}

        for subdirectory in paths.DATA_DIR.iterdir():
            if subdirectory.is_dir():

                label = subdirectory.name             
                label_exists = label in images_with_labels.keys()

                image_paths = [path for path in subdirectory.iterdir() if path.suffix.lower() == self.JPEG_EXT]

                if(label_exists):
                    images_with_labels[label] = image_paths
                    continue

                images_with_labels[label] += image_paths

        return images_with_labels

                    
                