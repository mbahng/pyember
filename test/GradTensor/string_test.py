import unittest
from ember import GradTensor

class TestProperGradTensorIO(unittest.TestCase):

  def testScalar(self):
    x = GradTensor([1], [1], 0, 0)
    gt = "GradTensor\n  +1.000\nshape = (1), dtype = double, bidx = 0, pidx = 0\n"
    self.assertEqual(str(x), gt)

  def test_1N_Vector(self):
    x = GradTensor([1, 2, 3], [1, 3], 0, 0)
    gt = "GradTensor\n  +1.000  +2.000  +3.000\n\nshape = (1, 3), dtype = double, bidx = 0, pidx = 0\n"
    self.assertEqual(str(x), gt)

  def test_N1_Vector(self): 
    x = GradTensor([1, 2, 3], [3, 1], 0, 1)
    gt = "GradTensor\n  +1.000\n  +2.000\n  +3.000\n\nshape = (3, 1), dtype = double, bidx = 0, pidx = 1\n"
    self.assertEqual(str(x), gt)

  def test_NM_Tensor(self):
    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 0)
    gt = (
        "GradTensor\n"
        "  +1.000  +2.000\n"
        "  +3.000  +4.000\n\n"
        "shape = (2, 2), dtype = double, bidx = 0, pidx = 0\n"
    )
    self.assertEqual(str(x), gt)

  def test_1NM_Tensor(self):
      x = GradTensor([1, 2, 3, 4], [1, 2, 2], 0, 0)
      gt = (
          "GradTensor\n[\n"
          "  +1.000  +2.000\n"
          "  +3.000  +4.000\n"
          "]\n\n"
          "shape = (1, 2, 2), dtype = double, bidx = 0, pidx = 0\n"
      )
      self.assertEqual(str(x), gt)

  def test_N1M_Tensor(self):
    x = GradTensor([1, 2, 3, 4], [2, 1, 2], 0, 2)
    gt = (
        "GradTensor\n[\n"
        "  +1.000  +2.000\n"
        "]\n"
        "[\n"
        "  +3.000  +4.000\n"
        "]\n\n"
        "shape = (2, 1, 2), dtype = double, bidx = 0, pidx = 2\n"
    )
    self.assertEqual(str(x), gt)

  def test_NM1_Tensor(self):
    x = GradTensor([1, 2, 3, 4], [2, 2, 1], 0, 1)
    gt = (
        "GradTensor\n[\n"
        "  +1.000\n"
        "  +2.000\n"
        "]\n"
        "[\n"
        "  +3.000\n"
        "  +4.000\n"
        "]\n\n"
        "shape = (2, 2, 1), dtype = double, bidx = 0, pidx = 1\n"
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

if __name__ == "__main__": 
  unittest.main(verbosity=2)
