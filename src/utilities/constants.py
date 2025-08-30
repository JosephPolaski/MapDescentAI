import logging

LOG_LEVEL = logging.INFO
RGB_MAX = 255
RANDOM_SEED = 88
JPEG_EXT = ".jpg"

LABEL_TO_INDEX_MAP = {
    "AnnualCrop" : 0,
    "Forest" : 1,
    "HerbaceousVegetation" : 2,
    "Highway" : 3,
    "Industrial" : 4,
    "Pasture" : 5,
    "PermanentCrop" : 6,
    "Residential" : 7,
    "River" : 8,
    "SeaLake" : 9
}

INDEX_TO_LABEL_MAP = {
    0 : "AnnualCrop",
    1 : "Forest",
    2 : "HerbaceousVegetation",
    3 : "Highway",
    4 : "Industrial",
    5 : "Pasture",
    6 : "PermanentCrop",
    7 : "Residential",
    8 : "River",
    9 : "SeaLake"
}