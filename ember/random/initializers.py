from ..tensor_cpp import Tensor 
from .. import tensor_cpp
from typing import List

def gaussian(size: List[int], mean: float = 0.0, stddev: float = 1.0) -> Tensor: 
    return tensor_cpp.gaussian(size, mean, stddev)

def uniform(size: List[int], min: float = 0.0, max: float = 1.0) -> Tensor: 
    return tensor_cpp.uniform(size, min, max)

def ones(size: List[int]) -> Tensor: 
    return tensor_cpp.ones(size)

def zeros(size: List[int]) -> Tensor: 
    return tensor_cpp.ones(size)

