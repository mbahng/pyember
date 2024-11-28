import unittest
from ember import Tensor, ScalarTensor

class TestTensorType(unittest.TestCase):

   def testType(self):
       x = Tensor([1])
       y = Tensor([1, 2, 3])
       z = Tensor([[2, 3], [4, 5]])
       self.assertEqual(x.type(), "Tensor")
       self.assertEqual(y.type(), "Tensor")
       self.assertEqual(z.type(), "Tensor")

   def testDtype(self):
       x = Tensor([1])
       y = Tensor([1, 2, 3])
       z = Tensor([[2, 3], [4, 5]])
       self.assertEqual(x.dtype(), "double")
       self.assertEqual(y.dtype(), "double")
       self.assertEqual(z.dtype(), "double")

class TestTensorEquality(unittest.TestCase):

   def testEqualScalar(self):
       x = Tensor([1])
       y = Tensor([1])
       z = Tensor([2])
       self.assertEqual(x == y, True)
       self.assertEqual(x == z, False)
       self.assertEqual(x.__eq__(y), True)
       self.assertEqual(x.__eq__(z), False)

   def testEqualVector(self):
       x = Tensor([1, 2, 3, 4])
       y = Tensor([1, 2, 3, 4])
       z = Tensor([1, 2, 3, 5])
       w = Tensor([1, 2, 3])
       self.assertEqual(x == y, True)
       self.assertEqual(x == z, False) 
       self.assertEqual(x == w, False)
       self.assertEqual(x.__eq__(y), True)
       self.assertEqual(x.__eq__(z), False)
       self.assertEqual(x.__eq__(w), False)

   def testEqualMatrix(self):
       x = Tensor([[1, 2], [3, 4]])
       y = Tensor([[1, 2], [3, 4]])
       z = Tensor([[1, 2], [3, 5]])
       w = Tensor([[1, 2, 3], [4, 5, 6]])
       self.assertEqual(x == y, True)
       self.assertEqual(x == z, False)
       self.assertEqual(x == w, False)
       self.assertEqual(x.__eq__(y), True)
       self.assertEqual(x.__eq__(z), False)
       self.assertEqual(x.__eq__(w), False)

   def testEqualTensor(self):
       x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
       y = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
       z = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 9]]])
       w = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
       self.assertEqual(x == y, True)
       self.assertEqual(x == z, False)
       self.assertEqual(x == w, False)
       self.assertEqual(x.__eq__(y), True)
       self.assertEqual(x.__eq__(z), False)
       self.assertEqual(x.__eq__(w), False)

class TestTensorInequality(unittest.TestCase):

   def testNotEqualScalar(self):
       x = Tensor([1])
       y = Tensor([1])
       z = Tensor([2])
       self.assertEqual(x != y, False)
       self.assertEqual(x != z, True)
       self.assertEqual(x.__ne__(y), False)
       self.assertEqual(x.__ne__(z), True)

   def testNotEqualVector(self):
       x = Tensor([1, 2, 3, 4])
       y = Tensor([1, 2, 3, 4])
       z = Tensor([1, 2, 3, 5])
       w = Tensor([1, 2, 3])
       self.assertEqual(x != y, False)
       self.assertEqual(x != z, True)
       self.assertEqual(x != w, True)
       self.assertEqual(x.__ne__(y), False)
       self.assertEqual(x.__ne__(z), True)
       self.assertEqual(x.__ne__(w), True)

   def testNotEqualMatrix(self):
       x = Tensor([[1, 2], [3, 4]])
       y = Tensor([[1, 2], [3, 4]])
       z = Tensor([[1, 2], [3, 5]])
       w = Tensor([[1, 2, 3], [4, 5, 6]])
       self.assertEqual(x != y, False)
       self.assertEqual(x != z, True)
       self.assertEqual(x != w, True)
       self.assertEqual(x.__ne__(y), False)
       self.assertEqual(x.__ne__(z), True)
       self.assertEqual(x.__ne__(w), True)

   def testNotEqualTensor(self):
       x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
       y = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
       z = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 9]]])
       w = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
       self.assertEqual(x != y, False)
       self.assertEqual(x != z, True)
       self.assertEqual(x != w, True)
       self.assertEqual(x.__ne__(y), False)
       self.assertEqual(x.__ne__(z), True)
       self.assertEqual(x.__ne__(w), True)

class TestTensorGetterSetter(unittest.TestCase):

   def testGetScalar(self):
       x = Tensor([1])
       self.assertEqual(x[0], ScalarTensor(1))

   def testGetVector(self):
       x = Tensor([1, 2, 3, 4])
       self.assertEqual(x[0], ScalarTensor(1))
       self.assertEqual(x[3], ScalarTensor(4))

   def testGetMatrix(self):
       x = Tensor([[1, 2], [3, 4]])
       self.assertEqual(x[0,0], ScalarTensor(1))
       self.assertEqual(x[0,1], ScalarTensor(2))
       self.assertEqual(x[1,0], ScalarTensor(3))
       self.assertEqual(x[1,1], ScalarTensor(4))

   def testGet3DTensor(self):
       x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
       self.assertEqual(x[0,0,0], ScalarTensor(1))
       self.assertEqual(x[1,1,1], ScalarTensor(8))

class TestTensorSetItem(unittest.TestCase):

   def testSetScalar(self):
       x = Tensor([1])
       x[0] = 5
       self.assertEqual(x[0], ScalarTensor(5))

   def testSetVector(self):
       x = Tensor([1, 2, 3, 4])
       x[0] = 5
       x[3] = 8
       self.assertEqual(x[0], ScalarTensor(5))
       self.assertEqual(x[3], ScalarTensor(8))

   def testSetMatrix(self):
       x = Tensor([[1, 2], [3, 4]])
       x[0,0] = 5
       x[1,1] = 8
       self.assertEqual(x[0,0], ScalarTensor(5))
       self.assertEqual(x[1,1], ScalarTensor(8))

   def testSet3DTensor(self):
       x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
       x[0,0,0] = 9
       x[1,1,1] = 10
       self.assertEqual(x[0,0,0], ScalarTensor(9))
       self.assertEqual(x[1,1,1], ScalarTensor(10))

class TestTensorSlice(unittest.TestCase):

    def testSliceVector(self):
        x = Tensor([1, 2, 3, 4])
        self.assertEqual(x[:], Tensor([1, 2, 3, 4]))
        self.assertEqual(x[1:3], Tensor([2, 3]))
        self.assertEqual(x[::2], Tensor([1, 3]))

    def testSliceMatrix(self):
        x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(x[:, 1], Tensor([2, 5, 8], [3, 1]))  # Keep the first dimension
        self.assertEqual(x[1:, :2], Tensor([[4, 5], [7, 8]]))
        self.assertEqual(x[::2, ::2], Tensor([[1, 3], [7, 9]]))

    def test3DTensorSlice(self):
        x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(x[0], Tensor([[[1, 2], [3, 4]]]))  # Should be 2D
        self.assertEqual(x[:, 1], Tensor([[[3, 4]], [[7, 8]]]))  # Should be 2D
        self.assertEqual(x[:, 1,:], Tensor([[[3, 4]], [[7, 8]]]))  # Should be 2D
        self.assertEqual(x[:, :, 1], Tensor([[[2], [4]], [[6], [8]]]))  # Should be 2D
        self.assertEqual(x[0, 1], Tensor([3, 4], [1, 1, 2]))  # Should be 1D
        self.assertEqual(x[0, 1, :], Tensor([3, 4], [1, 1, 2]))  # Should be 1D
        self.assertEqual(x[0, 0, 1], ScalarTensor(2))  # Should be scalar

    def testSliceEdgeCases(self):
        x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(x[:1], Tensor([[[1, 2], [3, 4]]]))  # Keep 3D with size 1
        self.assertEqual(x[:, :1], Tensor([[[1, 2]], [[5, 6]]]))  # Keep 3D with size 1

class TestProperTensorIO(unittest.TestCase): 

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
    t2 = t1.transpose()
    truth = Tensor([0, 3, 1, 4, 2, 5], [3, 2]) 

    self.assertEqual(t2, truth)

if __name__ == "__main__": 
 unittest.main(verbosity=2)

