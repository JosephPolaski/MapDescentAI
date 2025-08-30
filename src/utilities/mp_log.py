import logging
import inspect
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
            fmt="[%(asctime)s] | %(caller_name)s : %(message)s",
            datefmt="%H:%M:%S"
        )

        file_handler = logging.FileHandler(self.filepath)
        file_handler.setLevel(constants.LOG_LEVEL) 
        self.logger.propagate = False   
        file_handler.setFormatter(formatter)

        self.logger.setLevel(constants.LOG_LEVEL)
        self.logger.addHandler(file_handler)

    def info(self, message:str):
        caller_name = inspect.stack()[1].function
        self.logger.info(message, extra={'caller_name': caller_name})

    def warning(self, message:str):
        caller_name = inspect.stack()[1].function
        self.logger.warning(message, extra={'caller_name': caller_name})

    def error(self, message:str):
        caller_name = inspect.stack()[1].function
        self.logger.error(message, extra={'caller_name': caller_name})

    def method_entry(self):
        caller_name = inspect.stack()[1].function
        self.logger.error("Method Entered", extra={'caller_name': caller_name})

    


        


