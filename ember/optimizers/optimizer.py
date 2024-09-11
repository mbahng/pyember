from abc import ABC
from functools import singledispatch
# from .. import Tensor, Model, Loss, LinearRegression
#
# class Optimizer(ABC): 
#    
#     def __init__(self, model: Model, loss: Loss): 
#         pass 
#
# class IterativeOptimizer(Optimizer): 
#    
#     def __init__(self, model: Model, loss: Loss, lr: float): 
#         self.model = model 
#         self.loss = loss
#         self.lr = lr
#         self.grad_params = {k : v for k, v in model.params.items()}
#
#     @singledispatch
#     def gradient(self, model: Model, X: Tensor, Y: Tensor): 
#         return None 
#
# class ClosedOptimizer(Optimizer): 
#    
#     def __init__(self): 
#         pass 
#
# class SGDOptimizer(IterativeOptimizer): 
#    
#     def __init__(self, model, loss, lr): 
#         super().__init__(model, loss, lr)
#
#     @singledispatch
#     def gradient(self, model: Model, X: Tensor, Y: Tensor): 
#         return None 
#
#     @gradient.register
#     def _(self, model: LinearRegression, X: Tensor, Y: Tensor, Y_pred: Tensor): 
#         # X is N x d_in, X.T is d_in x N, Y_pred - Y is N x d_out
#         # So dW is d_in x d_out
#         # dW = (X.T(0, 1).matmul(Y_pred - Y)).T(0, 1)
#         dW = ((Y_pred - Y).T(0, 1)).matmul(X)
#         db = (Y_pred - Y).sum()
#         self.grad_params["lin_map"] = dW
#         self.grad_params["bias"] = db
#
#
# class AdamOptimizer(IterativeOptimizer): 
#    
#     def __init__(self, model, loss, lr): 
#         super().__init__(model, loss, lr)
#
# class NesterovOptimizer(IterativeOptimizer): 
#    
#     def __init__(self, model, loss, lr): 
#         super().__init__(model, loss, lr)
#
# class NewtonOptimizer(IterativeOptimizer): 
#    
#     def __init__(self, model, loss, lr): 
#         super().__init__(model, loss, lr)
#
# class LBFGSOptimizer(IterativeOptimizer): 
#    
#     def __init__(self, model, loss, lr): 
#         super().__init__(model, loss, lr)
#
#
