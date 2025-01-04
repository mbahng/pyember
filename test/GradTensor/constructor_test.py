import unittest
from ember import GradTensor

class TestGradTensorConstructors(unittest.TestCase): 

  def testFullConstructor(self): 
    g = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    self.assertEqual(g.storage, [1, 2, 3, 4]) 
    self.assertEqual(g.shape, [2, 2])
    self.assertEqual(g.bidx, 0)
    self.assertEqual(g.pidx, 1)

  def testDefaultConstructor(self):
    g = GradTensor([2, 2], 0, 1)
    self.assertEqual(g.storage, [0, 0, 0, 0])
    self.assertEqual(g.shape, [2, 2])
    self.assertEqual(g.bidx, 0)
    self.assertEqual(g.pidx, 1)

  def testScalarConstructor(self): 
    g = GradTensor(3.12) 
    self.assertEqual(g.storage, [3.12]) 
    self.assertEqual(g.item(), 3.12) 
    self.assertEqual(g.bidx, 0)
    self.assertEqual(g.pidx, 0)

class TestGradTensorPivot(unittest.TestCase): 

  def testPivot(self): 
    x1 = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    x2 = GradTensor([1, 2, 3, 4], [1, 2, 2], 0, 2)
    x3 = GradTensor([1, 2, 3, 4], [1, 1, 2, 2], 0, 3)
    x4 = GradTensor([1, 2, 3, 4], [2, 1, 2, 1], 0, 2)
    x5 = GradTensor([2, 2], 0, 1)
    x6 = GradTensor([1, 2, 2], 0, 2)
    x7 = GradTensor([1, 1, 2, 2], 0, 3)
    x8 = GradTensor([2, 1, 2, 1], 0, 2)

    self.assertEqual(x1.pidx, 1)
    self.assertEqual(x2.pidx, 2)
    self.assertEqual(x3.pidx, 3)
    self.assertEqual(x4.pidx, 2)
    self.assertEqual(x5.pidx, 1)
    self.assertEqual(x6.pidx, 2)
    self.assertEqual(x7.pidx, 3)
    self.assertEqual(x8.pidx, 2)

class TestGradTensorType(unittest.TestCase): 

  def testType(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1) 
    self.assertEqual(x.type, "GradTensor")

  def testDtype(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1) 
    self.assertEqual(x.dtype, "double")

if __name__ == "__main__": 
  unittest.main(verbosity=2)

