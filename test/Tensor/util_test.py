import unittest
from ember import Tensor 

class TestProperIO(unittest.TestCase): 

  def testScalar(self): 
    x = Tensor([1], [1]) 
    gt = "Tensor\n  +1.000\nshape = (1), dtype = double\n"
    self.assertEqual(str(x), gt)

  def test_1N_Vector(self): 
    x = Tensor([1, 2, 3], [1, 3]) 
    gt = "Tensor\n  +1.000  +2.000  +3.000\n\nshape = (1, 3), dtype = double\n"
    self.assertEqual(str(x), gt)

  def test_N1_Vector(self): 
    x = Tensor([1, 2, 3], [3, 1]) 
    gt = "Tensor\n  +1.000\n  +2.000\n  +3.000\n\nshape = (3, 1), dtype = double\n"
    self.assertEqual(str(x), gt)

  def test_NM_Tensor(self): 
    x = Tensor([1, 2, 3, 4], [2, 2]) 
    gt = (
      "Tensor\n" 
      "  +1.000  +2.000\n"
      "  +3.000  +4.000\n\n"
      "shape = (2, 2), dtype = double\n"
    )
    self.assertEqual(str(x), gt)

  def test_1NM_Tensor(self): 
    x = Tensor([1, 2, 3, 4], [1, 2, 2]) 
    gt = (
      "Tensor\n[\n" 
      "  +1.000  +2.000\n"
      "  +3.000  +4.000\n"
      "]\n\n"
      "shape = (1, 2, 2), dtype = double\n"
    )
    self.assertEqual(str(x), gt)

  def test_N1M_Tensor(self): 
    x = Tensor([1, 2, 3, 4], [2, 1, 2]) 
    gt = (
      "Tensor\n[\n" 
      "  +1.000  +2.000\n"
      "]\n"
      "[\n" 
      "  +3.000  +4.000\n"
      "]\n\n"
      "shape = (2, 1, 2), dtype = double\n"
    )
    self.assertEqual(str(x), gt)

  def test_NM1_Tensor(self): 
    x = Tensor([1, 2, 3, 4], [2, 2, 1]) 
    gt = (
      "Tensor\n[\n" 
      "  +1.000\n"
      "  +2.000\n"
      "]\n"
      "[\n" 
      "  +3.000\n"
      "  +4.000\n"
      "]\n\n"
      "shape = (2, 2, 1), dtype = double\n"
    )
    self.assertEqual(str(x), gt)

  def test_1LMN_Tensor(self): 
    pass

  def test_L1MN_Tensor(self): 
    pass

  def test_LM1N_Tensor(self): 
    pass

  def test_LMN1_Tensor(self): 
    pass

class TestTensorUtils(unittest.TestCase): 

 def testTranspose(self): 
  t1 = Tensor.arange(0, 6, 1).reshape([2, 3]) 
  t2 = t1.transpose([0, 1])
  truth = Tensor([0, 3, 1, 4, 2, 5], [3, 2]) 

  self.assertTrue(t2 == truth)

if __name__ == "__main__": 
 unittest.main(verbosity=2)

