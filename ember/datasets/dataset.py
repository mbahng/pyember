from .. import Tensor 

class Dataset(): 

    def __init__(self, X: Tensor, Y: Tensor) -> None: 
        self.index = 0
        self.X = X 
        self.Y = Y 

        assert self.X.shape[0] == self.Y.shape[0], \
            f"Length of input ({self.X.shape[0]}) does not \
              match length of output ({self.Y.shape[0]})."

        self.limit = self.X.shape[0]

    def __len__(self): 
        return self.limit

    def __iter__(self): 
        return self 

    def __next__(self): 
        if self.index >= self.limit: 
            raise StopIteration
        tmp = (self.X[self.index], self.Y[self.index]) 
        self.index += 1 # self.index is an int so passed by value 
        return tmp
