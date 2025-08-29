import file_helpers
import logging

from logging import Logger
from datetime import datetime
from utilities.paths import Paths
from utilities.constants import Constants
from decorators.lazy_singleton import lazy_singleton

@lazy_singleton
class MPLog:

    def __init__(self):
        self.timestamp = f"{datetime.strftime("%Y%m%d%H%M%S")}"
        self.filename = f"mapdescentlog_" + self.timestamp + ".log"
        self.logger : Logger = None
                        
        self.__initialize_logger()
        
    def __initialize_logger(self):
        isDirSuccess = file_helpers.try_create_directory(Paths.LOGS_DIR)

        if not isDirSuccess:
            print("MapDescentAI logger failed to initialize")
            return

        self.logger = logging.getLogger("MapDescentAiLogger")

        # remove all existing loggers to ensure only file logging
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        formatter = logging.Formatter(
            fmt="[%(asctime)s] | %(module)s:%(lineno)d : %(message)s",
            datefmt="%H:%M:%S"
        )

        file_handler = logging.FileHandler(self.filename)
        file_handler.setLevel(Constants.LOG_LEVEL)    
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def info(self, message:str):
        self.logger.info(message)

    def warning(self, message:str):
        self.logger.warning(message)

    def error(self, message:str):
        self.logger.error(message)

    


        


