from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelParameters:
    weights : NDArray = None
    bias : NDArray = None        
    loss_history : List[float] = field(default_factory=list)
    number_of_classes : int = 0