from pathlib import Path

root = Path(__file__).parent.parent.parent

SRC_DIR = root / "src"
DATA_DIR = root / "data/EuroSAT_RGB"
STORED_DATA_DIR = root / "data/stored"
LOGS_DIR = root / "logs"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"