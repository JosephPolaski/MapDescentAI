import os
from pathlib import Path
from utilities import paths

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
    
@staticmethod
def get_most_recent_dataset_filename() -> Path | None:
    try:
        dataset_files = [f for f in paths.STORED_DATA_DIR.iterdir() if f.is_file]
        return max(dataset_files, key=lambda f: f.stat().st_mtime)
    except:
        return None