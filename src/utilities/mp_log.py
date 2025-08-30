import logging
import utilities.paths as paths
import utilities.constants as constants
import utilities.file_helpers as fh

from logging import Logger
from pathlib import Path
from datetime import datetime
from utilities.decorators.lazy_singleton import lazy_singleton

@lazy_singleton
class MPLog:

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.filename = f"mapdescentlog_" + self.timestamp + ".log"
        self.filepath = paths.LOGS_DIR / self.filename
        self.logger : Logger = None
                        
        self.__initialize_logger()

        self.logger.info("Logger initialized successfully")
        
    def __initialize_logger(self):
        isDirSuccess = fh.try_create_directory(paths.LOGS_DIR)

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

        file_handler = logging.FileHandler(self.filepath)
        file_handler.setLevel(constants.LOG_LEVEL) 
        self.logger.propagate = False   
        file_handler.setFormatter(formatter)

        self.logger.setLevel(constants.LOG_LEVEL)
        self.logger.addHandler(file_handler)

    def info(self, message:str):
        self.logger.info(message)

    def warning(self, message:str):
        self.logger.warning(message)

    def error(self, message:str):
        self.logger.error(message)

    def method_entry(self):
        self.logger.error("Method Entered")

    


        


