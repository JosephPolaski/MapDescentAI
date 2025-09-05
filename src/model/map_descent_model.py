import numpy as np
import utilities.constants as constants

from data_management.data_manager import DataManager
from data_management.dataset import MapDescentDataset
from data_management.data_transfer_objects.model_parameters import ModelParameters
from data_management.enums.stored_data_type import StoredDataType
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, classification_report
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

    def evaluate_performance(self, is_test : bool = False):
        """Evaluate model performance by running model on test dataset"""

        labels = self.dataset.labels_test if is_test else self.dataset.labels_train
        dataset_name = 'Test' if is_test else 'Train'

        # Run forward pass on test dataset to get probabilities
        test_probabilities = self.forward_pass(is_test=is_test)
        predicted_classes = np.argmax(test_probabilities, axis=1)
        
        calculated_accuracy = np.mean(predicted_classes == labels)
        calculated_loss = self.calclate_cross_entropy_loss(test_probabilities, labels)

        matrix_confusion = confusion_matrix(labels, predicted_classes)
        report = classification_report(labels, predicted_classes, target_names= [constants.INDEX_TO_LABEL_MAP[i] for i in np.unique(labels)])

        self.logger.info("\nEvaluation Report \n" + 
                         "============================\n" +
                         f'{dataset_name} Accuracy: {calculated_accuracy:.4f}\n' + 
                         f'{dataset_name} Loss: {calculated_loss:.4f} \n\n' +
                         f'Confusion Matrix: \n{matrix_confusion}\n\n' +
                         f'Classification Report: \n{report}\n' +
                         "============================\n\n")
        
        self.logger.info(f"True label counts: {np.bincount(labels, minlength=self.dataset.number_of_classes)}")
        self.logger.info(f"Predicted counts: {np.bincount(predicted_classes, minlength=self.dataset.number_of_classes)}")

    def forward_pass(self, is_test : bool = False) -> NDArray:        
        """Compute land use class probabilities for each class in each image using softmax"""

        dataset = self.dataset.features_test if is_test else self.dataset.features_train      
            
        # Creating linear combination of features and weights (raw scores per class per image)
        raw_class_scores = np.dot(dataset, self.parameters.weights) 
        raw_class_scores += self.parameters.bias

        # Get maximum per row to prevent large exponentials in softmax 
        max_class_score_per_row = np.max(raw_class_scores, axis=1, keepdims=True)
        
        # Convert raw scores to normalized class probabilities that sum to 1 per image
        score_exponentials = np.exp(raw_class_scores - max_class_score_per_row)
        probabilities = score_exponentials / np.sum(score_exponentials, axis=1, keepdims=True)
        return probabilities
    
    def calclate_cross_entropy_loss(self, probabilities: np.ndarray, labels : np.ndarray) -> float: 
        # 1. Ensure probabilities don't hit exactly 0 or 1
        bounded_probabilities = np.clip(probabilities, self.epsilon, 1 - self.epsilon)

        # 2. Extract the probability assignd to the correct classes
        class_probabilities = bounded_probabilities[np.arange(len(labels)), labels]

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

        shuffled_indexes = np.arange(self.dataset.features_train.shape[0])
        self.dataset.features_train = self.dataset.features_train[shuffled_indexes]
        self.dataset.labels_train = self.dataset.labels_train[shuffled_indexes]

        shuffled_indexes_test = np.arange(self.dataset.features_test.shape[0])
        self.dataset.features_test = self.dataset.features_test[shuffled_indexes_test]
        self.dataset.labels_test = self.dataset.labels_test[shuffled_indexes_test]

        for epoch in range(self.parameters.epochs):

            probabilities = self.forward_pass()

            loss = self.calclate_cross_entropy_loss(probabilities, self.dataset.labels_train)
            self.backward_pass(probabilities)

            self.logger.info(f"Epoch {epoch} completed with a loss value of {loss}")
                

        

            
       
        

   


