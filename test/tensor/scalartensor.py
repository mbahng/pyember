import unittest
from ember import ScalarTensor, Tensor

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


class TestScalarTensorMath(unittest.TestCase): 

  def testSumScalarVector(self): 
    c = ScalarTensor(2)
    v = Tensor.arange(0, 5, 1)
    truth = Tensor.arange(2, 7, 1) 
    sum1 = v + c
    sum2 = c + v 
    self.assertTrue(sum1 == sum2 == truth) 

  def testSumScalarMatrix(self): 
    c = ScalarTensor(2)
    v = Tensor.arange(0, 6, 1).reshape([2, 3])
    truth = Tensor.arange(2, 8, 1).reshape([2, 3])
    sum1 = v + c
    sum2 = c + v 
    self.assertTrue(sum1 == sum2 == truth) 

  def testSumScalarTensor(self): 
    c = ScalarTensor(2)
    v = Tensor.arange(0, 24, 1).reshape([2, 3, 4])
    truth = Tensor.arange(2, 26, 1).reshape([2, 3, 4])
    sum1 = v + c
    sum2 = c + v 
    self.assertTrue(sum1 == sum2 == truth) 

if __name__ == "__main__": 
  unittest.main(verbosity=2)

