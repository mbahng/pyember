import unittest
from ember import ScalarTensor 

class TestTensorConstructors(unittest.TestCase): 

  def testNullConstructor(self): 
    s = ScalarTensor() 
    self.assertEqual(s.data(), [0])
    self.assertEqual(s.shape, [1])

  def testScalarConstructor(self): 
    s = ScalarTensor(2) 
    self.assertEqual(s.data(), [2])
    self.assertEqual(s.shape, [1])

  def testVectorConstructor(self): 
    s = ScalarTensor([2]) 
    self.assertEqual(s.data(), [2])
    self.assertEqual(s.shape, [1])

if __name__ == "__main__": 
  unittest.main(verbosity=2)


