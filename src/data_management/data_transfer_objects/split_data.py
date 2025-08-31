from numpy.typing import NDArray
from dataclasses import dataclass

@dataclass
class SplitData:
    labels_train: NDArray = None
    labels_test : NDArray = None
    features_train : NDArray = None
    features_test : NDArray = None