from .tensor import Tensor 

class Dataset(): 

    def __init__(self, X: Tensor, Y: Tensor) -> None: 
        self.index = 0
        self.X = X 
        self.Y = Y 

        assert len(self.X) == len(self.Y), \
            f"Length of input ({len(self.X)}) does not \
              match length of output ({len(self.Y)})."

        self.limit = len(self.X)

    def __iter__(self): 
        return self 

    def __next__(self): 
        if self.index >= self.limit: 
            raise StopIteration
        tmp = (self.X[self.index], self.Y[self.index]) 
        self.index += 1 # self.index is an int so passed by value 
        return tmp
