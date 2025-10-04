from torch import Tensor
from dataclasses import dataclass

@dataclass
class SplitData:
    labels_train: Tensor = None
    labels_test : Tensor = None
    features_train : Tensor = None
    features_test : Tensor = None