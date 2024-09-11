from .. import Tensor

class Loss(): 
    def __init__(self): 
        self.loss = None

    def objective(self, y_pred: Tensor, y_true: Tensor) -> Tensor: 
        pass

class MSELoss(Loss): 

    def __init__(self): 
        super().__init__()

    def objective(self, y_pred: Tensor, y_true: Tensor) -> Tensor: 
        self.loss = ((y_pred - y_true) ** 2).mean() 
        return self.loss
