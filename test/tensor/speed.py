import unittest
import ember
import numpy as np 
import torch
from time import time

class CompareAddSubtract(unittest.TestCase): 

    def compareAddSubtractSimple(self): 
        pass 

class CompareMatrixMultiplication(unittest.TestCase): 

    def compareMatrixMultiplication(self): 
        now = time() 
        X = ember.random.gaussian([1000, 1000])
        Y = ember.random.gaussian([1000, 1000])
        Z = X @ Y
        print(f"Ember : {time() - now}")

        now = time() 
        X = np.random.randn(1000, 1000)
        Y = np.random.randn(1000, 1000)
        Z = np.matmul(X, Y) 
        print(f"Numpy : {time() - now}")

        now = time()
        X = torch.randn(1000, 1000)
        Y = torch.randn(1000, 1000)
        Z = X @ Y
        print(f"Torch : {time() - now}")

        self.assertTrue("Speed Comparion Tests for 1000x1000")


if __name__ == "__main__": 
    unittest.main()

