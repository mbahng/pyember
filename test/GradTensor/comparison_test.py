import unittest
from ember import GradTensor

class TestGradTensorEquality(unittest.TestCase): 

  def testGradEquals(self): 
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    self.assertEqual(x, y)

  def testGradNotEquals(self): 
    w = GradTensor([1, 2, 3, 4], [2, 2], 0, 1)
    x = GradTensor([2, 2, 3, 4], [2, 2], 0, 1)
    y = GradTensor([1, 2, 3, 4], [4], 0, 1)
    z = GradTensor([1, 2, 3, 4], [2, 2], 0, 0)

    self.assertNotEqual(w, x)
    self.assertNotEqual(w, y)
    self.assertNotEqual(w, z)
    self.assertNotEqual(x, y)
    self.assertNotEqual(x, z)
    self.assertNotEqual(y, z)

class TestGradTensorLessGreaterThan(unittest.TestCase): 

  def testLeqGeq(self): 
    x = GradTensor(3.12) 
    y = GradTensor(2.13)  
    z = GradTensor(3.12)
    self.assertTrue(x > y) 
    self.assertTrue(x >= y) 
    self.assertTrue(x >= z) 
    self.assertFalse(x < y) 
    self.assertFalse(x <= y) 
    self.assertTrue(x <= z) 

  def testLeqGeqVector(self): 
    x = GradTensor([8, 3, 1, -1], [2, 2], 0, 1)
    y = GradTensor([1, 2, 3, 4], [2, 2], 0, 1) 
    z = GradTensor([0, 1, 2, 3], [2, 2], 0, 1) 
    self.assertFalse(x < y)
    self.assertFalse(x <= y)
    self.assertFalse(x > y)
    self.assertFalse(x >= y)
    self.assertTrue(z < y)
    self.assertTrue(z <= y)
    self.assertFalse(z > y)
    self.assertFalse(z >= y)

if __name__ == "__main__": 
  unittest.main(verbosity=2)

