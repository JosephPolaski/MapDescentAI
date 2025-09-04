from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelParameters:
    weights : NDArray = None
    bias : NDArray = None        
    loss_history : NDArray = None
    number_of_classes : int = 0
    learning_rate : float = 0.01
    epochs : int = 200