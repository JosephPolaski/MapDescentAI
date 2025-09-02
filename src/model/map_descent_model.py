
import numpy as np
import utilities.paths as paths

from data_management.data_manager import DataManager
from data_management.dataset import MapDescentDataset
from data_management.data_transfer_objects.model_parameters import ModelParameters
from data_management.enums.stored_data_type import StoredDataType
from numpy.typing import NDArray
from utilities import constants
from utilities import MDLog

class MapDescentModel:

    def __init__(self):
        # Encapsulated Utilities
        self.logger = MDLog()
        self.dataset : MapDescentDataset = MapDescentDataset()
        self.data_manager : DataManager = DataManager()

        # Data Members
        self.epsilon = 1e-15
        self.parameters : ModelParameters = ModelParameters()        

        self.__try_load_parameters()
        
    def __save_parameters(self):
        self.data_manager.store_data_locally(StoredDataType.PARAMETERS, self.parameters)

    def __try_load_parameters(self):
        stored_parameters = self.data_manager.load_stored_data(StoredDataType.PARAMETERS)  

        if stored_parameters is None:
            self.logger.info("No stored parameters have been established yet, starting with defaults")
            self.parameters.weights = np.random.randn(self.dataset.number_of_features, self.dataset.number_of_classes) * 0.01
            self.parameters.bias = np.zeros((self.dataset.number_of_classes, 1))  
            return

        self.parameters = stored_parameters

    def forward_pass(self) -> NDArray:
        
        self.logger.info("Compute land use class probabilities for each class in each image using softmax")

        self.logger.info("Creating linear combination of features and weights (raw scores per class per image)")
        raw_class_scores = np.dot(self.dataset.features_train, self.parameters.weights) 
        raw_class_scores += self.parameters.bias

        self.logger.info("Get maximum per row to prevent large exponentials in softmax")
        max_class_score_per_row = np.max(raw_class_scores, axis=1, keepdims=True)
        
        self.logger.info("Convert raw scores to normalized class probabilities that sum to 1 per image")
        score_exponentials = np.exp(raw_class_scores - max_class_score_per_row)
        probabilities = score_exponentials / np.sum(score_exponentials, axis=1, keepdims=True)
        return probabilities
    
    def calclate_cross_entropy_loss(self, probabilities: np.ndarray) -> float:
        self.logger.info("Calculating cross entropy loss.")

        number_of_samples = self.dataset.labels_train.shape[0]
        cross_entropy_loss = -np.sum(self.dataset.labels_train * np.log(probabilities + self.epsilon)) /   number_of_samples

        self.logger.info("Adding calculated loss to loss history parameter")
        self.parameters.loss_history.append(cross_entropy_loss)
        return cross_entropy_loss

    def train_model(self):
        pass

    def predict_class_probability(self):
        pass

    def probabilities_to_class_labels(self):
        pass

    def compute_loss(self):
        pass

    def evaluate_performance(self):
        pass

   


