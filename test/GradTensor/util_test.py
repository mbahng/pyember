import unittest
from ember import GradTensor

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

