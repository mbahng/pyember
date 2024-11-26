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

if __name__ == "__main__": 
  unittest.main(verbosity=2)

