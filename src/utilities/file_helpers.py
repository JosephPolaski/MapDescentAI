import os
from pathlib import Path

@staticmethod
def try_create_directory(create_path : Path) -> bool:
    try:
        if not os.path.isdir(create_path):
            os.mkdir(create_path)

        return True    
    except:
        return False   


@staticmethod
def try_create_file(create_path : Path) -> bool:
    try:
        if not os.path.isfile(create_path):
            file_descriptor : int = os.open(create_path, os.O_CREAT)
            os.close(file_descriptor)

        return True    
    except:
        return False