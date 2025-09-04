import numpy as np

from data_management.data_manager import DataManager
from data_management.dataset import MapDescentDataset
from data_management.data_transfer_objects.model_parameters import ModelParameters
from data_management.enums.stored_data_type import StoredDataType
from numpy.typing import NDArray
from utilities.md_log import MDLog

class MapDescentModel:

    def __init__(self):
        # Encapsulated Utilities
        self.logger = MDLog()
        self.dataset : MapDescentDataset = MapDescentDataset()
        self.data_manager : DataManager = DataManager()

        # Data Members
        self.training_loss_history = []
        self.epsilon = 1e-15
        self.parameters : ModelParameters = ModelParameters()        

        self.__try_load_parameters()      
   
    def __try_load_parameters(self):
        stored_parameters = self.data_manager.load_stored_data(StoredDataType.PARAMETERS)  

        if stored_parameters is None:
            self.logger.info("No stored parameters have been established yet, starting with defaults")
            self.parameters.weights = np.random.randn(self.dataset.number_of_features, self.dataset.number_of_classes) * 0.01
            self.parameters.bias = np.zeros((1, self.dataset.number_of_classes))  
            return

        self.parameters = stored_parameters
        self.logger.info("Successfully loaded saved training parameters")

    def forward_pass(self) -> NDArray:        
        """Compute land use class probabilities for each class in each image using softmax"""

        # Creating linear combination of features and weights (raw scores per class per image)
        raw_class_scores = np.dot(self.dataset.features_train, self.parameters.weights) 
        raw_class_scores += self.parameters.bias

        # Get maximum per row to prevent large exponentials in softmax 
        max_class_score_per_row = np.max(raw_class_scores, axis=1, keepdims=True)
        
        # Convert raw scores to normalized class probabilities that sum to 1 per image
        score_exponentials = np.exp(raw_class_scores - max_class_score_per_row)
        probabilities = score_exponentials / np.sum(score_exponentials, axis=1, keepdims=True)
        return probabilities
    
    def calclate_cross_entropy_loss(self, probabilities: np.ndarray) -> float: 
        # 1. Ensure probabilities don't hit exactly 0 or 1
        bounded_probabilities = np.clip(probabilities, self.epsilon, 1 - self.epsilon)

        # 2. Extract the probability assignd to the correct classes
        class_probabilities = bounded_probabilities[np.arange(len(self.dataset.labels_train)), self.dataset.labels_train]

        # 3. Cross-entropy = negative average log probability of correct classes
        cross_entropy_loss = -np.mean(np.log(class_probabilities))
        
        self.training_loss_history.append(float(cross_entropy_loss))
        return cross_entropy_loss

    def backward_pass(self, probabilities: np.ndarray):
        """ Calculating loss gradients with respect to each weight and bias """

        number_of_samples = self.dataset.features_train.shape[0]

        gradient_of_loss_logits = probabilities.copy()
        gradient_of_loss_logits[range(number_of_samples), self.dataset.labels_train] -= 1
        gradient_of_loss_logits /= number_of_samples

        gradient_weights = np.dot(self.dataset.features_train.T, gradient_of_loss_logits)
        gradient_bias = np.sum(gradient_of_loss_logits, axis=0, keepdims=True)

        # Updating weights and bias
        self.parameters.weights -= self.parameters.learning_rate * gradient_weights
        self.parameters.bias -=  self.parameters.learning_rate * gradient_bias

    def save_parameters(self):
        training_loss_array = np.array(self.training_loss_history, dtype=np.float32)

        if(self.parameters.loss_history is None):
            self.parameters.loss_history = training_loss_array
        else:
            self.parameters.loss_history = np.concatenate([self.parameters.loss_history, training_loss_array])
            
        self.data_manager.store_data_locally(StoredDataType.PARAMETERS, self.parameters)

    def train_model(self):
        self.logger.info(f"Training MapDescentAI Model with Learning Rate: {self.parameters.learning_rate} and Epochs: {self.parameters.epochs}")

        for epoch in range(self.parameters.epochs):

            probabilities = self.forward_pass()

            loss = self.calclate_cross_entropy_loss(probabilities)
            self.backward_pass(probabilities)

            self.logger.info(f"Epoch {epoch} completed with a loss value of {loss}")
                

        

            
       
        

   


