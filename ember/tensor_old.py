from typing import Iterable, List, Union

class Tensor(): 

    def __init__(self, data: List) -> None: 
        self.data = data 
        self.length = len(self.data)
        self.hyperdim = [self.length] 

        self.hyperdims = 0 
        tmp = self.data
        while type(tmp) == List: 
            self.hyperdims += 1 
            
    def __len__(self) -> int: 
        return self.length

    def __getitem__(self, idx) -> int: 
        return self.data[idx]
    
    def __str__(self) -> str: 
        return str(self.data)
    
    def __repr__(self) -> str:
        return str(self.data)

class Scalar(Tensor): 
    
    def __init__(self, data: Union[List,float]): 
        if isinstance(data, float): 
            self.data = [data]
        elif isinstance(data, List): 
            assert len(self.data) == 1
            self.data = data
        super().__init__(self.data) 

class Vector(Tensor): 
    
    def __init__(self, data: List[float]) -> None: 
        super().__init__(data)

    # need to use quotes for typehinting since class isn't fully defined 
    def dot(self, other: 'Vector') -> Scalar: 
        assert len(self) == len(other), \
            f"Lengths of {len(self)} and {len(other)} do not match. "
        
        return Scalar(sum([x * y for x, y in zip(self.data, other.data)]))

class Matrix(Tensor): 
    
    def __init__(self, data: List[List[float]]) -> None: 
        super().__init__(data)

    def matmul(self, other: 'Matrix') -> 'Matrix': 
        return self
