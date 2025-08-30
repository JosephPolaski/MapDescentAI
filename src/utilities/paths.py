from pathlib import Path

root = Path(__file__).parent.parent.parent

SRC_DIR = root / "src"
DATA_DIR = root / "data/EuroSAT_RGB"
LOGS_DIR = root / "logs"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"