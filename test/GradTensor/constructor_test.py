import unittest
from ember import GradTensor

class TestGradTensorConstructors(unittest.TestCase): 

  def testFullConstructor(self): 
    g = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    self.assertEqual(g.storage, [1, 2, 3, 4]) 
    self.assertEqual(g.shape, [2, 2])
    self.assertEqual(g.pidx, 1)

  def testDefaultConstructor(self):
    g = GradTensor([2, 2], 0, 1)
    self.assertEqual(g.storage, [0, 0, 0, 0])
    self.assertEqual(g.shape, [2, 2])
    self.assertEqual(g.pidx, 1)

  def testEyeConstructor(self):
    g = GradTensor.eye(2, 1)
    self.assertEqual(g.storage, [1, 0, 0, 1])
    self.assertEqual(g.shape, [2, 2])
    self.assertEqual(g.pidx, 1)

  def testCopyConstructor(self): 
    g = GradTensor([2, 2], 0, 1)
    c = g.copy() 
    self.assertEqual(g, c)

if __name__ == "__main__": 
  unittest.main(verbosity=2)

