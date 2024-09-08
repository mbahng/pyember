from . import Model
from .. import Dataset
from .. import Tensor 
from .. import random

class Regression(Model): 

    def __init__(self): 
        super().__init__()
        self.params = {}

class LinearRegression(Regression): 

    def __init__(self, input_dim: int, output_dim: int, compute_least_squares: bool = True): 
        super().__init__() 
        self.params = {
            "lin_map" : random.gaussian([output_dim, input_dim]), 
            "bias" : Tensor([[0]])
        }
        self.compute_least_squares = compute_least_squares

    def fit(self, dataset: Dataset): 
        X, Y = dataset.X, dataset.Y
        # using batch gradient descent
        lr = 0.001
        for _ in range(1000): 
            Y_pred = self.forward(X)
            loss = ((Y_pred - Y) ** 2).mean()
            # X is N x d_in, X.T is d_in x N, Y_pred - Y is N x d_out
            # So dW is d_in x d_out
            dW = (X.T(0, 1).matmul(Y_pred - Y)).T(0, 1)
            db = (Y_pred - Y).sum()
            self.params["lin_map"] -= lr * dW
            self.params["bias"] -= lr * db 

    def forward(self, X: Tensor): 
        W = self.params["lin_map"]  # d_out x d_in
        b = self.params["bias"]     # d_out x 1
        # first X.T(0, 1) gives us d_in x N 
        # W.matmul(X.T(0, 1)) is d_out x d_in times d_in x N -> d_out x d_in
        # add that to b which is 1 x 1, which broadcasts 
        return (W.matmul(X.T(0, 1)) + b).reshape([X.shape[0], b.shape[0]])


