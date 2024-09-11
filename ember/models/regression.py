from . import Model
from .. import Dataset
from .. import Tensor ,Dataset, Tensor, Loss
from .. import random

class Regression(Model): 

    def __init__(self): 
        super().__init__()
        self.params = {}

class LinearRegression(Regression): 

    def __init__(self, input_dim: int, output_dim: int): 
        super().__init__() 
        self.params = {
            "lin_map" : random.gaussian([output_dim, input_dim]), 
            "bias" : Tensor([[0]])
        }
        self.gradients = {k: None for k in self.params}
        self.computationGraph = {}
        self.Y_pred = None 

    def forward(self, X: Tensor): 
        W = self.params["lin_map"]  # d_out x d_in
        b = self.params["bias"]     # d_out x 1

        # first X.T(0, 1) gives us d_in x N 
        # W.matmul(X.T(0, 1)) is d_out x d_in times d_in x N -> d_out x d_in
        # add that to b which is 1 x 1, which broadcasts 
        self.Y_pred = (W.matmul(X.T(0, 1)) + b).reshape([X.shape[0], b.shape[0]])
        return self.Y_pred

    def backward(self, X: Tensor, Y: Tensor): 
        # just computes gradients 
        dW = ((self.Y_pred - Y).T(0, 1)).matmul(X)
        db = (self.Y_pred - Y).sum()
        return dW, db

    def fit(self, dataset, loss, optimizer): 
        X, Y = dataset.X, dataset.Y

        # if isinstance(optimizer, IterativeOptimizer): 
        #     lr = optimizer.lr
        #     Y_pred = self.forward(X)
        #
        #     # compute loss (optional in this case)
        #     loss.objective(Y_pred, Y)
        #    
        #     # Loss computes gradients 
        #     optimizer.gradient(self, X, Y, Y_pred)
        #
        #     # Optimizer updates the model  
        #     for param in optimizer.grad_params: 
        #         self.params[param] -= lr * optimizer.grad_params[param]

