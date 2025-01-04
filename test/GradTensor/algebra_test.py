import unittest
from ember import GradTensor, Tensor


class TestEqNeg(unittest.TestCase):

  def testEqNeq(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1) 
    y = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)  
    z = GradTensor([2, 3, 4, 5], [2, 2], 0, 1)
    self.assertEqual(x, y)
    self.assertNotEqual(x, z) 

# gradtensors with different pivots should not be added, 
# These are covered in the C++ tests 

class TestGradAdd(unittest.TestCase): 

  def testAddGradTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    truth = GradTensor([2, 4, 6, 8], [2, 2], 0, 1)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)
    self.assertEqual(x.__add__(y), truth)
    self.assertEqual(x.__radd__(y), truth)
    self.assertEqual(y.__add__(x), truth)
    self.assertEqual(y.__radd__(x), truth)

  def testAddTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = Tensor([1, 2, 3, 4], [2, 2])
    truth = Tensor([2, 4, 6, 8], [2, 2])
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)
    self.assertEqual(x.__add__(y), truth)
    self.assertEqual(x.__radd__(y), truth)
    self.assertEqual(y.__add__(x), truth)
    self.assertEqual(y.__radd__(x), truth)

  def testAddFloat(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = 1.
    truth = GradTensor([2, 3, 4, 5], [2, 2], 0, 1)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth) 
    self.assertEqual(x.__add__(y), truth)
    self.assertEqual(x.__radd__(y), truth)

class TestGradSub(unittest.TestCase): 

  def testSubGradTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = GradTensor([4, 5, 6, 7], [2, 2], 0, 1)
    truth1 = GradTensor([-3, -3, -3, -3], [2, 2], 0, 1)
    truth2 = GradTensor([3, 3, 3, 3], [2, 2], 0, 1)
    self.assertEqual(x - y, truth1)
    self.assertEqual(y - x, truth2)
    self.assertEqual(x.__sub__(y), truth1)
    self.assertEqual(x.__rsub__(y), truth2)
    self.assertEqual(y.__sub__(x), truth2)
    self.assertEqual(y.__rsub__(x), truth1)

  def testSubTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = Tensor([1, 2, 3, 4], [2, 2])
    truth = Tensor([0, 0, 0, 0], [2, 2])
    self.assertEqual(x - y, truth)
    self.assertEqual(y - x, truth)
    self.assertEqual(x.__sub__(y), truth)
    self.assertEqual(x.__rsub__(y), truth)
    self.assertEqual(y.__sub__(x), truth)
    self.assertEqual(y.__rsub__(x), truth)

  def testSubFloat(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = 1
    truth1 = GradTensor([0, 1, 2, 3], [2, 2], 0, 1)
    self.assertEqual(x - y, truth1)
    self.assertEqual(x.__sub__(y), truth1)

class TestGradMul(unittest.TestCase): 

  def testMulGradTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    truth = GradTensor([1, 4, 9, 16], [2, 2], 0, 1)
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)
    self.assertEqual(x.__mul__(y), truth)
    self.assertEqual(x.__rmul__(y), truth)
    self.assertEqual(y.__mul__(x), truth)
    self.assertEqual(y.__rmul__(x), truth)

  def testMulTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = Tensor([1, 2, 3, 4], [2, 2])
    truth = Tensor([1, 4, 9, 16], [2, 2])
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)
    self.assertEqual(x.__mul__(y), truth)
    self.assertEqual(x.__rmul__(y), truth)
    self.assertEqual(y.__mul__(x), truth)
    self.assertEqual(y.__rmul__(x), truth)

  def testMulFloat(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = 2
    truth = GradTensor([2, 4, 6, 8], [2, 2], 0, 1)
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)
    self.assertEqual(x.__mul__(y), truth)
    self.assertEqual(x.__rmul__(y), truth)

class TestGradMatmul(unittest.TestCase): 

  def testMatmulGradTensor(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    truth = GradTensor([7, 10, 15, 22], [2, 2], 0, 1)
    self.assertEqual(x @ y, truth)

if __name__ == "__main__": 
  unittest.main(verbosity=2)

