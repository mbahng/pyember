import unittest
from ember import GradTensor

class TestGradTensorAlgebra(unittest.TestCase): 
  # gradtensors with different pivots should not be added, 
  # These are covered in the C++ tests 

  def testGradAdd(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 1)
    truth = GradTensor([2, 4, 6, 8], [2, 2], 1)
    self.assertEqual(x + y, truth)        

  def testGradSub(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 1)
    truth = GradTensor([0, 0, 0, 0], [2, 2], 1)
    self.assertEqual(x - y, truth)

  def testGradMul(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 1)
    truth = GradTensor([1, 4, 9, 16], [2, 2], 1)
    self.assertEqual(x * y, truth)
  
  def testGradMatmul(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 1)
    truth = GradTensor([7, 10, 15, 22], [2, 2], 1)
    self.assertEqual(x @ y, truth)

  def testPivots(self):  
    # after applying chain rule, should keep track of pivot index 
    pass 

if __name__ == "__main__": 
  unittest.main(verbosity=2)

