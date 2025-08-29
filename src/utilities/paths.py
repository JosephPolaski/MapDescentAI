from pathlib import Path

class Paths:
    root = Path(__file__).parent.parent

    SRC_DIR = root / "src"
    DATA_DIR = root / "data"
    LOGS_DIR = root / "logs"
    PREPROCESSING_DIR = SRC_DIR / "preprocessing"