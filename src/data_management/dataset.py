import utilities.file_helpers as file_helpers

from data_management.data_manager import DataManager
from data_management.data_transfer_objects.split_data import SplitData
from data_management.image_processor import ImageProcessor
from utilities.md_log import MDLog


class MapDescentDataset:

    def __init__(self):
        self.logger = MDLog()
        self.data_manager = DataManager()       

        self.stored_data_filename = ""
        self.number_of_classes = self.data_manager.label_count
        self.number_of_features = self.data_manager.training_feature_count

        self.labels_train = None
        self.labels_test = None
        self.features_train = None
        self.features_test = None

        self.__initialize_data()

    def __check_if_stored_data_exists(self) -> bool:
        try:
            stored_file_path = file_helpers.get_most_recent_dataset_filename()

            if stored_file_path is None:
                return False 

            self.stored_data_filename = stored_file_path.name
            return True       
            
        except Exception as ex:
            self.logger.error(f"Check for stored data failed: \n\n {ex} \n\n")
            return False
        
    def __try_fetch_stored_data(self) -> bool:
        stored_data_exists = self.__check_if_stored_data_exists()

        if not stored_data_exists:
            return False
        
        stored_data : SplitData | None = self.data_manager.load_stored_data() 

        if stored_data is None:
            raise ValueError("Stored Map Descent dataset was None")

        self.labels_test = stored_data.labels_test
        self.labels_train = stored_data.labels_train
        self.features_test = stored_data.features_test
        self.features_train = stored_data.features_train
        return True

    def __preprocess_and_fetch(self):
        image_processor = ImageProcessor()
        image_processor.build_flattened_image_data()

        isStoredDataSuccess = self.__try_fetch_stored_data()

        if not isStoredDataSuccess:
            raise FileNotFoundError("Unable to initialize data...")
        
    def __initialize_data(self):        
        try:
            isStoredDataSuccess = self.__try_fetch_stored_data()

            if isStoredDataSuccess:
                return
            
            self.__preprocess_and_fetch()            

        except Exception as ex:
            self.logger.error(f"failed to initialize dataset: \n\n {ex} \n\n")