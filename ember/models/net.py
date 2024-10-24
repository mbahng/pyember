from ember import Tensor
from typing import Any, Optional
from abc import ABC

class Net(ABC): 

    def __init__(self): 
        pass 

    def forward(self, X: Tensor) -> Tensor: 
        pass
