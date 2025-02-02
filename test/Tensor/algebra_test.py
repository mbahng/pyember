import unittest
from itertools import product
from ember import Tensor, GradTensor

class TestNeg(unittest.TestCase):

  def testNegate(self): 
    x = Tensor([1]) 
    y = Tensor([1, 2, 3, 4]) 
    z = Tensor([[1, 2], [3, 4]]) 
    xt = Tensor([-1]) 
    yt = Tensor([-1, -2, -3, -4]) 
    zt = Tensor([[-1, -2], [-3, -4]]) 
    self.assertEqual(-x, xt)
    self.assertEqual(-y, yt)
    self.assertEqual(-z, zt)
    
class TestGradTensorAdd(unittest.TestCase):

  def testAddScalarGradTensor(self):
    x = Tensor([1])
    y = GradTensor([2], [1], 0, 1)
    truth = Tensor([3], [1], 0)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)
    self.assertEqual(x.__add__(y), truth)
    self.assertEqual(x.__radd__(y), truth)
    self.assertEqual(y.__add__(x), truth)
    self.assertEqual(y.__radd__(x), truth)

  def testAddVectorGradTensor(self):
    x = Tensor([1, 2, 3, 4])
    y = GradTensor([5, 6, 7, 8], [4], 0, 1)
    truth = Tensor([6, 8, 10, 12], [4], 0)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)
    self.assertEqual(x.__add__(y), truth)
    self.assertEqual(x.__radd__(y), truth)
    self.assertEqual(y.__add__(x), truth)
    self.assertEqual(y.__radd__(x), truth)

  def testAddMatrixGradTensor(self):
    x = Tensor([[1, 2], [3, 4]])
    y = GradTensor([5, 6, 7, 8], [2, 2], 0, 1)
    truth = Tensor([6, 8, 10, 12], [2, 2], 0)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)
    self.assertEqual(x.__add__(y), truth)
    self.assertEqual(x.__radd__(y), truth)
    self.assertEqual(y.__add__(x), truth)
    self.assertEqual(y.__radd__(x), truth)

  def testAddTensorGradTensor(self):
    x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = GradTensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2], 0, 1)
    truth = Tensor([2, 4, 6, 8, 10, 12, 14, 16], [2, 2, 2], 0)
    self.assertEqual(x + y, truth)
    self.assertEqual(y + x, truth)
    self.assertEqual(x.__add__(y), truth)
    self.assertEqual(x.__radd__(y), truth)
    self.assertEqual(y.__add__(x), truth)
    self.assertEqual(y.__radd__(x), truth)

class TestTensorAdd(unittest.TestCase): 

  def testAddScalarTensor(self):
     x = Tensor([1])
     y = Tensor([2])
     truth = Tensor([3])
     self.assertEqual(x + y, truth)
     self.assertEqual(y + x, truth)
     self.assertEqual(x.__add__(y), truth)
     self.assertEqual(x.__radd__(y), truth)
     self.assertEqual(y.__add__(x), truth)
     self.assertEqual(y.__radd__(x), truth)

  def testAddVectorTensor(self):
     x = Tensor([1, 2, 3, 4])
     y = Tensor([1, 2, 3, 4])
     truth = Tensor([2, 4, 6, 8])
     self.assertEqual(x + y, truth)
     self.assertEqual(y + x, truth)
     self.assertEqual(x.__add__(y), truth)
     self.assertEqual(x.__radd__(y), truth)
     self.assertEqual(y.__add__(x), truth)
     self.assertEqual(y.__radd__(x), truth)

  def testAddMatrixTensor(self):
     x = Tensor([[1, 2], [3, 4]])
     y = Tensor([[1, 2], [3, 4]])
     truth = Tensor([[2, 4], [6, 8]])
     self.assertEqual(x + y, truth)
     self.assertEqual(y + x, truth)
     self.assertEqual(x.__add__(y), truth)
     self.assertEqual(x.__radd__(y), truth)
     self.assertEqual(y.__add__(x), truth)
     self.assertEqual(y.__radd__(x), truth)

  def testAddTensorTensor(self):
     x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
     y = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
     truth = Tensor([[[2, 4], [6, 8]], [[10, 12], [14, 16]]])
     self.assertEqual(x + y, truth)
     self.assertEqual(y + x, truth)
     self.assertEqual(x.__add__(y), truth)
     self.assertEqual(x.__radd__(y), truth)
     self.assertEqual(y.__add__(x), truth)
     self.assertEqual(y.__radd__(x), truth)

class TestTensorSub(unittest.TestCase): 

  def testSubScalarTensor(self):
     x = Tensor([1])
     y = Tensor([2])
     truth = Tensor([-1])
     self.assertEqual(x - y, truth)
     self.assertEqual(y - x, Tensor([1]))
     self.assertEqual(x.__sub__(y), truth)
     self.assertEqual(x.__rsub__(y), Tensor([1]))
     self.assertEqual(y.__sub__(x), Tensor([1]))
     self.assertEqual(y.__rsub__(x), truth)

  def testSubVectorTensor(self):
     x = Tensor([1, 2, 3, 4])
     y = Tensor([1, 2, 3, 4])
     truth = Tensor([0, 0, 0, 0])
     self.assertEqual(x - y, truth)
     self.assertEqual(y - x, truth)
     self.assertEqual(x.__sub__(y), truth)
     self.assertEqual(x.__rsub__(y), truth)
     self.assertEqual(y.__sub__(x), truth)
     self.assertEqual(y.__rsub__(x), truth)

  def testSubMatrixTensor(self):
     x = Tensor([[1, 2], [3, 4]])
     y = Tensor([[1, 2], [3, 4]])
     truth = Tensor([[0, 0], [0, 0]])
     self.assertEqual(x - y, truth)
     self.assertEqual(y - x, truth)
     self.assertEqual(x.__sub__(y), truth)
     self.assertEqual(x.__rsub__(y), truth)
     self.assertEqual(y.__sub__(x), truth)
     self.assertEqual(y.__rsub__(x), truth)

  def testSubTensorTensor(self):
     x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
     y = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
     truth = Tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
     self.assertEqual(x - y, truth)
     self.assertEqual(y - x, truth)
     self.assertEqual(x.__sub__(y), truth)
     self.assertEqual(x.__rsub__(y), truth)
     self.assertEqual(y.__sub__(x), truth)
     self.assertEqual(y.__rsub__(x), truth)

class TestTensorMul(unittest.TestCase): 

  def testMulScalarTensor(self):
    x = Tensor([2])
    y = Tensor([3])
    truth = Tensor([6])
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)
    self.assertEqual(x.__mul__(y), truth)
    self.assertEqual(x.__rmul__(y), truth)
    self.assertEqual(y.__mul__(x), truth)
    self.assertEqual(y.__rmul__(x), truth)

  def testMulVectorTensor(self):
    x = Tensor([1, 2, 3, 4])
    y = Tensor([1, 2, 3, 4])
    truth = Tensor([1, 4, 9, 16])
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)
    self.assertEqual(x.__mul__(y), truth)
    self.assertEqual(x.__rmul__(y), truth)
    self.assertEqual(y.__mul__(x), truth)
    self.assertEqual(y.__rmul__(x), truth)

  def testMulMatrixTensor(self):
    x = Tensor([[1, 2], [3, 4]])
    y = Tensor([[1, 2], [3, 4]])
    truth = Tensor([[1, 4], [9, 16]])
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)
    self.assertEqual(x.__mul__(y), truth)
    self.assertEqual(x.__rmul__(y), truth)
    self.assertEqual(y.__mul__(x), truth)
    self.assertEqual(y.__rmul__(x), truth)

  def testMulTensorTensor(self):
    x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    truth = Tensor([[[1, 4], [9, 16]], [[25, 36], [49, 64]]])
    self.assertEqual(x * y, truth)
    self.assertEqual(y * x, truth)
    self.assertEqual(x.__mul__(y), truth)
    self.assertEqual(x.__rmul__(y), truth)
    self.assertEqual(y.__mul__(x), truth)
    self.assertEqual(y.__rmul__(x), truth)

class TestTensorMatmul(unittest.TestCase):

  def testMatmulTensor(self):
    x = Tensor([1, 2, 3, 4]).reshape([1, 4])
    y = Tensor([1, 2, 3, 4]).reshape([4, 1])
    truth = Tensor([30], [1, 1])
    self.assertEqual(x @ y, truth)
    self.assertEqual(x.__matmul__(y), truth)

    # Matrix * Vector 
    x = Tensor([1, 2, 3, 4]).reshape([2, 2])
    y = Tensor([1, 2]).reshape([2, 1])
    truth = Tensor([5, 11], [2, 1])
    self.assertEqual(x @ y, truth)
    self.assertEqual(x.__matmul__(y), truth)

    # Matrix * Matrix
    x = Tensor([1, 2, 3, 4]).reshape([2, 2])
    y = Tensor([1, 2, 3, 4]).reshape([2, 2])
    truth = Tensor([7, 10, 15, 22], [2, 2])
    self.assertEqual(x @ y, truth)
    self.assertEqual(x.__matmul__(y), truth)

    # 3x2 * 2x3 Matrix
    x = Tensor([1, 2, 3, 4, 5, 6]).reshape([3, 2])
    y = Tensor([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    truth = Tensor([9, 12, 15, 19, 26, 33, 29, 40, 51], [3, 3])
    self.assertEqual(x @ y, truth)
    self.assertEqual(x.__matmul__(y), truth)

class TestTensorAddWithGradients(unittest.TestCase): 

  def testAddScalarTensor(self): 
    x = Tensor([1]) 
    y = Tensor([2]) 
    z = x + y 
    z.backprop(True) 

    xgrad_truth = GradTensor([1.], [1, 1], 0, 1)
    ygrad_truth = GradTensor([1.], [1, 1], 0, 1)
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

    x = Tensor([1]) 
    y = Tensor([2]) 
    z = x + y 
    z.backprop(False) 

    xgrad_truth = GradTensor([1.], [1, 1], 0, 1)
    ygrad_truth = GradTensor([1.], [1, 1], 0, 1)

    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)


  def testAddVectorTensor(self): 
    x = Tensor.arange(0, 3, 1)
    y = Tensor.arange(0, 3, 1)
    z = x + y 
    z.backprop(True)
    grad_truth = GradTensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], 0, 1)
    
    self.assertEqual(x.grad, grad_truth)
    self.assertEqual(y.grad, grad_truth)

    x = Tensor.arange(0, 3, 1)
    y = Tensor.arange(0, 3, 1)
    z = x + y 
    z.backprop(False)
    grad_truth = GradTensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], 0, 1)
    self.assertEqual(x.grad, grad_truth)
    self.assertEqual(y.grad, grad_truth)

  def testAddMatrixTensor(self): 
    x = Tensor.arange(0, 6, 1).reshape([2, 3])
    y = Tensor.arange(0, 6, 1).reshape([2, 3])

    z = x + y 
    z.backprop(True)
    grad_truth = GradTensor([2, 3, 2, 3], 0, 2) 
    grad_truth[0, 0, 0, 0] = 1.0
    grad_truth[0, 1, 0, 1] = 1.0
    grad_truth[0, 2, 0, 2] = 1.0
    grad_truth[1, 0, 1, 0] = 1.0
    grad_truth[1, 1, 1, 1] = 1.0
    grad_truth[1, 2, 1, 2] = 1.0

    self.assertEqual(x.grad, grad_truth)
    self.assertEqual(y.grad, grad_truth)

    x = Tensor.arange(0, 6, 1).reshape([2, 3])
    y = Tensor.arange(0, 6, 1).reshape([2, 3])

    z = x + y 
    z.backprop(False)
    grad_truth = GradTensor([2, 3, 2, 3], 0, 2) 
    grad_truth[0, 0, 0, 0] = 1.0
    grad_truth[0, 1, 0, 1] = 1.0
    grad_truth[0, 2, 0, 2] = 1.0
    grad_truth[1, 0, 1, 0] = 1.0
    grad_truth[1, 1, 1, 1] = 1.0
    grad_truth[1, 2, 1, 2] = 1.0

    self.assertEqual(x.grad, grad_truth)
    self.assertEqual(y.grad, grad_truth)

  def testAddTensorTensor(self): 
    x = Tensor.arange(0, 24, 1).reshape([2, 3, 4])
    y = Tensor.arange(0, 24, 1).reshape([2, 3, 4])

    z = x + y 
    z.backprop(True)
    grad_truth = GradTensor([2, 3, 4, 2, 3, 4], 0, 3)
    combos = product(range(2), range(3), range(4)) 
    for i, j, k in combos: 
        grad_truth[i, j, k, i, j, k] = 1.0

    self.assertEqual(x.grad, grad_truth)
    self.assertEqual(y.grad, grad_truth)

    x = Tensor.arange(0, 24, 1).reshape([2, 3, 4])
    y = Tensor.arange(0, 24, 1).reshape([2, 3, 4])

    z = x + y 
    z.backprop(False)
    grad_truth = GradTensor([2, 3, 4, 2, 3, 4], 0, 3)
    combos = product(range(2), range(3), range(4)) 
    for i, j, k in combos: 
        grad_truth[i, j, k, i, j, k] = 1.0

    self.assertEqual(x.grad, grad_truth)
    self.assertEqual(y.grad, grad_truth)

class TestTensorSubWithGradients(unittest.TestCase): 

  def testSubScalarTensor(self): 
    x = Tensor([1]) 
    y = Tensor([2]) 
    z = x - y 
    z.backprop(True) 

    xgrad_truth = GradTensor([1.], [1, 1], 0, 1)
    ygrad_truth = GradTensor([-1.], [1, 1], 0, 1)
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

    x = Tensor([1]) 
    y = Tensor([2]) 
    z = x - y 
    z.backprop(False) 

    xgrad_truth = GradTensor([1.], [1, 1], 0, 1)
    ygrad_truth = GradTensor([-1.], [1, 1], 0, 1)
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

  def testSubVectorTensor(self): 
    x = Tensor.arange(0, 3, 1)
    y = Tensor.arange(0, 3, 1)
    z = x - y 
    z.backprop(True)
    xgrad_truth = GradTensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], 0, 1)
    ygrad_truth = GradTensor([-1, 0, 0, 0, -1, 0, 0, 0, -1], [3, 3], 0, 1)
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

    x = Tensor.arange(0, 3, 1)
    y = Tensor.arange(0, 3, 1)
    z = x - y 
    z.backprop(False)
    xgrad_truth = GradTensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], 0, 1)
    ygrad_truth = GradTensor([-1, 0, 0, 0, -1, 0, 0, 0, -1], [3, 3], 0, 1)
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

class TestTensorMulWithGradients(unittest.TestCase): 

  def testMulScalarTensor(self): 
    x = Tensor([1]) 
    y = Tensor([2]) 
    z = x * y 
    z.backprop(True) 

    xgrad_truth = GradTensor([2.], [1, 1], 0, 1)
    ygrad_truth = GradTensor([1.], [1, 1], 0, 1)
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

    x = Tensor([1]) 
    y = Tensor([2]) 
    z = x * y 
    z.backprop(False) 

    xgrad_truth = GradTensor([2.], [1, 1], 0, 1)
    ygrad_truth = GradTensor([1.], [1, 1], 0, 1)
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

  def testMulVectorTensor(self): 
    x = Tensor.uniform([4], 0, 1) 
    y = Tensor.uniform([4], 0, 1) 
    z = x * y
    z.backprop(True)

    xgrad_truth = GradTensor([4, 4], 0, 1)
    ygrad_truth = GradTensor([4, 4], 0, 1) 
    for i in range(4): 
        xgrad_truth[i, i] = y[i].storage[0]
        ygrad_truth[i, i] = x[i].storage[0]
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

    x = Tensor.uniform([4], 0, 1) 
    y = Tensor.uniform([4], 0, 1) 
    z = x * y
    z.backprop(False)

    xgrad_truth = GradTensor([4, 4], 0, 1)
    ygrad_truth = GradTensor([4, 4], 0, 1) 
    for i in range(4): 
        xgrad_truth[i, i] = y[i].storage[0]
        ygrad_truth[i, i] = x[i].storage[0]
    
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

  def testMulMatrixTensor(self): 
    x = Tensor.uniform([2, 3], 0, 1) 
    y = Tensor.uniform([2, 3], 0, 1) 
    z = x * y
    z.backprop(True)

    xgrad_truth = GradTensor([2, 3, 2, 3], 0, 2)
    ygrad_truth = GradTensor([2, 3, 2, 3], 0, 2) 
    for i, j in product(range(2), range(3)): 
        xgrad_truth[i, j, i, j] = y[i, j].storage[0]
        ygrad_truth[i, j, i, j] = x[i, j].storage[0]
   
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

    x = Tensor.uniform([2, 3], 0, 1) 
    y = Tensor.uniform([2, 3], 0, 1) 
    z = x * y
    z.backprop(False)

    xgrad_truth = GradTensor([2, 3, 2, 3], 0, 2)
    ygrad_truth = GradTensor([2, 3, 2, 3], 0, 2) 
    for i, j in product(range(2), range(3)): 
        xgrad_truth[i, j, i, j] = y[i, j].storage[0]
        ygrad_truth[i, j, i, j] = x[i, j].storage[0]
   
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

  def testMulTensorTensor(self): 
    x = Tensor.uniform([2, 3, 4], 0, 1) 
    y = Tensor.uniform([2, 3, 4], 0, 1) 
    z = x * y
    z.backprop(True)

    xgrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 0, 3)
    ygrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 0, 3) 
    for i, j, k in product(range(2), range(3), range(4)): 
        xgrad_truth[i, j, k, i, j, k] = y[i, j, k].storage[0]
        ygrad_truth[i, j, k, i, j, k] = x[i, j, k].storage[0]
   
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

    x = Tensor.uniform([2, 3, 4], 0, 1) 
    y = Tensor.uniform([2, 3, 4], 0, 1) 
    z = x * y
    z.backprop(False)

    xgrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 0, 3)
    ygrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 0, 3) 
    for i, j, k in product(range(2), range(3), range(4)): 
        xgrad_truth[i, j, k, i, j, k] = y[i, j, k].storage[0]
        ygrad_truth[i, j, k, i, j, k] = x[i, j, k].storage[0]
   
    self.assertEqual(x.grad, xgrad_truth)
    self.assertEqual(y.grad, ygrad_truth)

if __name__ == "__main__": 
  unittest.main(verbosity=2)


