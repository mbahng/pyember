import numpy as np 
import ember
import time  

N = 100

def profile(f): 
  def wrapper(*args, **kwargs): 
    start = time.process_time() 
    _ = f(*args, **kwargs) 
    end = time.process_time() 
    return end - start 
  return wrapper  

x = np.random.randn(N, N)
y = np.random.randn(N, N) 

a = ember.Tensor.gaussian([N, N]) 
b = ember.Tensor.gaussian([N, N])

@profile
def np_constructor(): return np.random.randn(N, N) 
@profile
def em_constructor(): return ember.Tensor.gaussian([N, N]) 

@profile
def np_add(): return x + y 
@profile
def em_add(): return a + b 

@profile
def np_diff(): return x - y 
@profile
def em_diff(): return a - b 

@profile
def np_mul(): return x * y 
@profile
def em_mul(): return a * b 

@profile
def np_matmul(): return x @ y 
@profile
def em_matmul(): return a @ b 

print(f"Constructor     Runtime Multiplier : {em_constructor() / np_constructor()}")  # type: ignore
print(f"Addition        Runtime Multiplier : {em_add() / np_add()}")  # type: ignore
print(f"Subtraction     Runtime Multiplier : {em_diff() / np_diff()}")  # type: ignore
print(f"Multiplication  Runtime Multiplier : {em_mul() / np_mul()}")  # type: ignore
print(f"Mat Multiply    Runtime Multiplier : {em_matmul() / np_matmul()}")  # type: ignore

