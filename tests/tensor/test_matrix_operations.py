import unittest
import numpy as np
from ember import Tensor 


class TestAddTensors(unittest.TestCase): 

    def testAddSimple(self): 

        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = Tensor([[4, 1, 2], [4, 5, 6]])
        sum = Tensor([[5, 3, 5], [8, 10, 12]])

        self.assertEqual(x + y, sum)


if __name__ == "__main__": 
    unittest.main()


