import unittest
from ember import ScalarTensor

class TestTensorConstructors(unittest.TestCase): 
  def nullConstructor(self): 
    s = ScalarTensor() 
    self.assertEqual(s.data(), [0])
    self.assertEqual(s.shape(), [])

  def scalarConstructor(self): 
    s = ScalarTensor(2) 
    self.assertEqual(s.data(), [2])
    self.assertEqual(s.shape(), [])

  def vectorConstructor(self): 
    s = ScalarTensor([2]) 
    self.assertEqual(s.data(), [2])
    self.assertEqual(s.shape(), [])
