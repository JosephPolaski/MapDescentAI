from dataclasses import dataclass
from typing import List

@dataclass
class FeatureLabelInfo:
    features_raw : List | None = None
    labels : List | None = None