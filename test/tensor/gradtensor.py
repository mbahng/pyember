import unittest
from ember import GradTensor

class TestGradTensorConstructors(unittest.TestCase): 
  def nullConstructor(self): 
    g = GradTensor() 
    truth = GradTensor([], [], 0) 
    self.assertEqual(g, truth)

  def fullConstructor(self): 
    g = GradTensor([1, 2, 3, 4], [2, 2], 1)
    self.assertEqual(g.data(), [1, 2, 3, 4]) 
    self.assertEqual(g.shape, [2, 2])
    self.assertEqual(g.pivot, 1)

  def defaultConstructor(self):
    g = GradTensor([2, 2], 1)
    self.assertEqual(g.data(), [0, 0, 0, 0])
    self.assertEqual(g.shape, [2, 2])
    self.assertEqual(g.pivot, 1)

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

class TestGradTensorUtil(unittest.TestCase): 

  def testGradEquals(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 1)
    self.assertEqual(x, y)

  def testGradNotEquals(self): 
    w = GradTensor([1, 2, 3, 4], [2, 2], 1)
    x = GradTensor([2, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [4], 1)
    z = GradTensor([1, 2, 3, 4], [2, 2], 0)

    self.assertNotEqual(w, x)
    self.assertNotEqual(w, y)
    self.assertNotEqual(w, z)
    self.assertNotEqual(x, y)
    self.assertNotEqual(x, z)
    self.assertNotEqual(y, z)

if __name__ == "__main__": 
  unittest.main(verbosity=2)

