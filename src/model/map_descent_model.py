
import numpy as np
import utilities.paths as paths

from data_management.data_manager import DataManager
from data_management.dataset import MapDescentDataset
from data_management.data_transfer_objects.model_parameters import ModelParameters
from data_management.enums.stored_data_type import StoredDataType
from utilities import constants
from utilities import MDLog

class MapDescentModel:

    def __init__(self):
        # Encapsulated Utilities
        self.logger = MDLog()
        self.dataset : MapDescentDataset = MapDescentDataset()
        self.data_manager : DataManager = DataManager()

        # Data Members
        self.parameters : ModelParameters = ModelParameters()
        self.learning_rate = 0
        self.epochs = 0

        self.__try_load_parameters()
        
    def __try_load_parameters(self):
        stored_parameters = self.data_manager.load_stored_data(StoredDataType.PARAMETERS)  

        if stored_parameters is None:
            self.logger.info("No stored parameters have been established yet, starting with default")  
            return

        self.parameters = stored_parameters


