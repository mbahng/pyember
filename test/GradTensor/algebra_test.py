import unittest
from ember import GradTensor, Tensor, ScalarTensor

# gradtensors with different pivots should not be added, 
# These are covered in the C++ tests 

class TestGradAdd(unittest.TestCase): 

  def testAddGradTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 1)
    truth = GradTensor([2, 4, 6, 8], [2, 2], 1)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)

  def testAddTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = Tensor([1, 2, 3, 4], [2, 2])
    truth = Tensor([2, 4, 6, 8], [2, 2])
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)

  def testAddScalarTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = ScalarTensor(1)
    truth = GradTensor([2, 3, 4, 5], [2, 2], 1)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)

  def testAddFloat(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = 1.
    truth = GradTensor([2, 3, 4, 5], [2, 2], 1)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth) 

class TestGradSub(unittest.TestCase): 

  def testSubGradTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([4, 5, 6, 7], [2, 2], 1)
    truth1 = GradTensor([-3, -3, -3, -3], [2, 2], 1)
    truth2 = GradTensor([3, 3, 3, 3], [2, 2], 1)
    self.assertEqual(x - y, truth1)
    self.assertEqual(y - x, truth2)

  def testSubTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = Tensor([1, 2, 3, 4], [2, 2])
    truth = Tensor([0, 0, 0, 0], [2, 2])
    self.assertEqual(x - y, truth)
    self.assertEqual(y - x, truth)

  def testSubScalarTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = ScalarTensor(1)
    truth1 = GradTensor([0, 1, 2, 3], [2, 2], 1)
    truth2 = GradTensor([0, -1, -2, -3], [2, 2], 1)
    self.assertEqual(x - y, truth1)
    self.assertEqual(y - x, truth2)

  def testSubFloat(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = 1
    truth1 = GradTensor([0, 1, 2, 3], [2, 2], 1)
    truth2 = GradTensor([0, -1, -2, -3], [2, 2], 1)
    self.assertEqual(x - y, truth1)
    self.assertEqual(y - x, truth2)

class TestGradMul(unittest.TestCase): 

  def testMulGradTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 1)
    truth = GradTensor([1, 4, 9, 16], [2, 2], 1)
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)

  def testMulTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = Tensor([1, 2, 3, 4], [2, 2])
    truth = Tensor([1, 4, 9, 16], [2, 2])
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)

  def testMulScalarTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = ScalarTensor(2)
    truth = GradTensor([2, 4, 6, 8], [2, 2], 1)
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)

  def testMulFloat(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = 2
    truth = GradTensor([2, 4, 6, 8], [2, 2], 1)
    self.assertEqual(x * y, truth)
    # self.assertEqual(y * x, truth)

class TestGradMatmul(unittest.TestCase): 

  def testMatmulGradTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 1)
    truth = GradTensor([7, 10, 15, 22], [2, 2], 1)
    self.assertEqual(x @ y, truth)

if __name__ == "__main__": 
  unittest.main(verbosity=2)

