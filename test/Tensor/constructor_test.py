import unittest
from ember import Tensor

class TestTensorConstructors(unittest.TestCase): 

  def testDefaultConstructor(self): 
    t = Tensor([1, 2, 3, 4], [2, 2]) 
    self.assertEqual(t.storage, [1, 2, 3, 4]) 
    self.assertEqual(t.shape, [2, 2])

  def test1DConstructor(self): 
    t = Tensor([1, 2, 3, 4])
    self.assertEqual(t.storage, [1, 2, 3, 4]) 
    self.assertEqual(t.shape, [4])

  def test2DConstructor(self): 
    t = Tensor([[1, 2], [3, 4]])
    self.assertEqual(t.storage, [1, 2, 3, 4]) 
    self.assertEqual(t.shape, [2, 2])

  def test3DConstructor(self): 
    t = Tensor([
      [[1, 2], [3, 4]], 
      [[5, 6], [7, 8]]
      ])
    self.assertEqual(t.storage, [1, 2, 3, 4, 5, 6, 7, 8]) 
    self.assertEqual(t.shape, [2, 2, 2])

  def testArangeConstructor(self): 
    t = Tensor.arange(0, 10, 2)
    self.assertEqual(t.storage, [0, 2, 4, 6, 8])
    self.assertEqual(t.shape, [5])

  def testLinspaceConstructor(self): 
    t = Tensor.linspace(0, 5, 6) 
    self.assertEqual(t.storage, [0, 1, 2, 3, 4, 5])
    self.assertEqual(t.shape, [6])

  def testGaussianConstructor(self): 
    t = Tensor.gaussian([3, 3], 0, 1)
    self.assertEqual(t.shape, [3, 3])

  def testUniformConstructor(self): 
    t = Tensor.uniform([2, 2], 0, 1)
    self.assertEqual(t.shape, [2, 2])

  def testOnesConstructor(self): 
    t = Tensor.ones([2, 3, 4])
    self.assertEqual(t.storage, [1] * 24)
    self.assertEqual(t.shape, [2, 3, 4])

  def testZerosConstructor(self): 
    t = Tensor.zeros([2, 3, 4])
    self.assertEqual(t.storage, [0] * 24)
    self.assertEqual(t.shape, [2, 3, 4])

if __name__ == "__main__": 
  unittest.main(verbosity=2)

