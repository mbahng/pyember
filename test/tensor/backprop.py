import unittest
from itertools import product
from ember import Tensor, GradTensor

class TestAutogradAdd(unittest.TestCase): 
    
    def testSumScalarsIntermediate(self): 
        x = Tensor([1]) 
        y = Tensor([2]) 
        z = x + y 
        z.backprop(True) 

        xgrad_truth = GradTensor([1.], [1, 1], 1)
        ygrad_truth = GradTensor([1.], [1, 1], 1)
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)
    
    def testSumScalarsNoIntermediate(self): 
        x = Tensor([1]) 
        y = Tensor([2]) 
        z = x + y 
        z.backprop(False) 

        xgrad_truth = GradTensor([1.], [1, 1], 1)
        ygrad_truth = GradTensor([1.], [1, 1], 1)
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testSumVectorsIntermediate(self): 
        x = Tensor.arange(0, 3, 1)
        y = Tensor.arange(0, 3, 1)
        z = x + y 
        z.backprop(True)
        grad_truth = GradTensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], 1)
        
        self.assertEqual(x.grad, grad_truth)
        self.assertEqual(y.grad, grad_truth)

    def testSumVectorsNoIntermediate(self): 
        x = Tensor.arange(0, 3, 1)
        y = Tensor.arange(0, 3, 1)
        z = x + y 
        z.backprop(False)
        grad_truth = GradTensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], 1)
        self.assertEqual(x.grad, grad_truth)
        self.assertEqual(y.grad, grad_truth)

    def testSumMatricesIntermediate(self): 
        x = Tensor.arange(0, 6, 1).reshape([2, 3])
        y = Tensor.arange(0, 6, 1).reshape([2, 3])

        z = x + y 
        z.backprop(True)
        grad_truth = GradTensor([2, 3, 2, 3], 2) 
        grad_truth[0, 0, 0, 0] = 1.0
        grad_truth[0, 1, 0, 1] = 1.0
        grad_truth[0, 2, 0, 2] = 1.0
        grad_truth[1, 0, 1, 0] = 1.0
        grad_truth[1, 1, 1, 1] = 1.0
        grad_truth[1, 2, 1, 2] = 1.0

        self.assertEqual(x.grad, grad_truth)
        self.assertEqual(y.grad, grad_truth)

    def testSumMatricesNoIntermediate(self): 
        x = Tensor.arange(0, 6, 1).reshape([2, 3])
        y = Tensor.arange(0, 6, 1).reshape([2, 3])

        z = x + y 
        z.backprop(False)
        grad_truth = GradTensor([2, 3, 2, 3], 2) 
        grad_truth[0, 0, 0, 0] = 1.0
        grad_truth[0, 1, 0, 1] = 1.0
        grad_truth[0, 2, 0, 2] = 1.0
        grad_truth[1, 0, 1, 0] = 1.0
        grad_truth[1, 1, 1, 1] = 1.0
        grad_truth[1, 2, 1, 2] = 1.0

        self.assertEqual(x.grad, grad_truth)
        self.assertEqual(y.grad, grad_truth)

    def testSumTensorsIntermediate(self): 
        x = Tensor.arange(0, 24, 1).reshape([2, 3, 4])
        y = Tensor.arange(0, 24, 1).reshape([2, 3, 4])

        z = x + y 
        z.backprop(True)
        grad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        combos = product(range(2), range(3), range(4)) 
        for i, j, k in combos: 
            grad_truth[i, j, k, i, j, k] = 1.0

        self.assertEqual(x.grad, grad_truth)
        self.assertEqual(y.grad, grad_truth)

    def testSumTensorsNoIntermediate(self): 
        x = Tensor.arange(0, 24, 1).reshape([2, 3, 4])
        y = Tensor.arange(0, 24, 1).reshape([2, 3, 4])

        z = x + y 
        z.backprop(False)
        grad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        combos = product(range(2), range(3), range(4)) 
        for i, j, k in combos: 
            grad_truth[i, j, k, i, j, k] = 1.0

        self.assertEqual(x.grad, grad_truth)
        self.assertEqual(y.grad, grad_truth)

class TestAutogradSub(unittest.TestCase): 
    
    def testSubScalarsIntermediate(self): 
        x = Tensor([1]) 
        y = Tensor([2]) 
        z = x - y 
        z.backprop(True) 

        xgrad_truth = GradTensor([1.], [1, 1], 1)
        ygrad_truth = GradTensor([-1.], [1, 1], 1)
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)
    
    def testSubScalarsNoIntermediate(self): 
        x = Tensor([1]) 
        y = Tensor([2]) 
        z = x - y 
        z.backprop(False) 

        xgrad_truth = GradTensor([1.], [1, 1], 1)
        ygrad_truth = GradTensor([-1.], [1, 1], 1)
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testSubVectorsIntermediate(self): 
        x = Tensor.arange(0, 3, 1)
        y = Tensor.arange(0, 3, 1)
        z = x - y 
        z.backprop(True)
        xgrad_truth = GradTensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], 1)
        ygrad_truth = GradTensor([-1, 0, 0, 0, -1, 0, 0, 0, -1], [3, 3], 1)
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testSubVectorsNoIntermediate(self): 
        x = Tensor.arange(0, 3, 1)
        y = Tensor.arange(0, 3, 1)
        z = x - y 
        z.backprop(False)
        xgrad_truth = GradTensor([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3], 1)
        ygrad_truth = GradTensor([-1, 0, 0, 0, -1, 0, 0, 0, -1], [3, 3], 1)
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testSubMatricesIntermediate(self): 
        x = Tensor.arange(0, 6, 1).reshape([2, 3])
        y = Tensor.arange(0, 6, 1).reshape([2, 3])

        z = x - y 
        z.backprop(True)
        xgrad_truth = GradTensor([2, 3, 2, 3], 2) 
        ygrad_truth = GradTensor([2, 3, 2, 3], 2) 
        combos = product(range(2), range(3)) 
        for i, j in combos: 
            xgrad_truth[i, j, i, j] = 1.0
            ygrad_truth[i, j, i, j] = -1.0

        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testSubMatricesNoIntermediate(self): 
        x = Tensor.arange(0, 6, 1).reshape([2, 3])
        y = Tensor.arange(0, 6, 1).reshape([2, 3])

        z = x - y 
        z.backprop(False)
        xgrad_truth = GradTensor([2, 3, 2, 3], 2) 
        ygrad_truth = GradTensor([2, 3, 2, 3], 2) 
        combos = product(range(2), range(3)) 
        for i, j in combos: 
            xgrad_truth[i, j, i, j] = 1.0
            ygrad_truth[i, j, i, j] = -1.0

        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testSubTensorsIntermediate(self): 
        x = Tensor.arange(0, 24, 1).reshape([2, 3, 4])
        y = Tensor.arange(0, 24, 1).reshape([2, 3, 4])

        z = x - y 
        z.backprop(True)
        xgrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        ygrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        combos = product(range(2), range(3), range(4)) 
        for i, j, k in combos: 
            xgrad_truth[i, j, k, i, j, k] = 1.0
            ygrad_truth[i, j, k, i, j, k] = -1.0

        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testSubTensorsNoIntermediate(self): 
        x = Tensor.arange(0, 24, 1).reshape([2, 3, 4])
        y = Tensor.arange(0, 24, 1).reshape([2, 3, 4])

        z = x - y 
        z.backprop(False)
        xgrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        ygrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        combos = product(range(2), range(3), range(4)) 
        for i, j, k in combos: 
            xgrad_truth[i, j, k, i, j, k] = 1.0
            ygrad_truth[i, j, k, i, j, k] = -1.0

        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

class TestAutogradMul(unittest.TestCase): 
    
    def testMulScalarsIntermediate(self): 
        x = Tensor([1]) 
        y = Tensor([2]) 
        z = x * y 
        z.backprop(True) 

        xgrad_truth = GradTensor([2.], [1, 1], 1)
        ygrad_truth = GradTensor([1.], [1, 1], 1)
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testMulScalarsNoIntermediate(self):
        x = Tensor([1]) 
        y = Tensor([2]) 
        z = x * y 
        z.backprop(False) 

        xgrad_truth = GradTensor([2.], [1, 1], 1)
        ygrad_truth = GradTensor([1.], [1, 1], 1)
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testMulVectorsIntermediate(self): 
        x = Tensor.uniform([4], 0, 1) 
        y = Tensor.uniform([4], 0, 1) 
        z = x * y
        z.backprop(True)

        xgrad_truth = GradTensor([4, 4], 1)
        ygrad_truth = GradTensor([4, 4], 1) 
        for i in range(4): 
            xgrad_truth[i, i] = y[i].data()[0]
            ygrad_truth[i, i] = x[i].data()[0]
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testMulVectorsNoIntermediate(self):
        x = Tensor.uniform([4], 0, 1) 
        y = Tensor.uniform([4], 0, 1) 
        z = x * y
        z.backprop(False)

        xgrad_truth = GradTensor([4, 4], 1)
        ygrad_truth = GradTensor([4, 4], 1) 
        for i in range(4): 
            xgrad_truth[i, i] = y[i].data()[0]
            ygrad_truth[i, i] = x[i].data()[0]
        
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testMulMatricesIntermediate(self):
        x = Tensor.uniform([2, 3], 0, 1) 
        y = Tensor.uniform([2, 3], 0, 1) 
        z = x * y
        z.backprop(True)

        xgrad_truth = GradTensor([2, 3, 2, 3], 2)
        ygrad_truth = GradTensor([2, 3, 2, 3], 2) 
        for i, j in product(range(2), range(3)): 
            xgrad_truth[i, j, i, j] = y[i, j].data()[0]
            ygrad_truth[i, j, i, j] = x[i, j].data()[0]
       
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testMulMatricesNoIntermediate(self):
        x = Tensor.uniform([2, 3], 0, 1) 
        y = Tensor.uniform([2, 3], 0, 1) 
        z = x * y
        z.backprop(False)

        xgrad_truth = GradTensor([2, 3, 2, 3], 2)
        ygrad_truth = GradTensor([2, 3, 2, 3], 2) 
        for i, j in product(range(2), range(3)): 
            xgrad_truth[i, j, i, j] = y[i, j].data()[0]
            ygrad_truth[i, j, i, j] = x[i, j].data()[0]
       
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testMulTensorsIntermediate(self):
        x = Tensor.uniform([2, 3, 4], 0, 1) 
        y = Tensor.uniform([2, 3, 4], 0, 1) 
        z = x * y
        z.backprop(True)

        xgrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        ygrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3) 
        for i, j, k in product(range(2), range(3), range(4)): 
            xgrad_truth[i, j, k, i, j, k] = y[i, j, k].data()[0]
            ygrad_truth[i, j, k, i, j, k] = x[i, j, k].data()[0]
       
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

    def testMulTensorsNoIntermediate(self):
        x = Tensor.uniform([2, 3, 4], 0, 1) 
        y = Tensor.uniform([2, 3, 4], 0, 1) 
        z = x * y
        z.backprop(False)

        xgrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        ygrad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3) 
        for i, j, k in product(range(2), range(3), range(4)): 
            xgrad_truth[i, j, k, i, j, k] = y[i, j, k].data()[0]
            ygrad_truth[i, j, k, i, j, k] = x[i, j, k].data()[0]
       
        self.assertEqual(x.grad, xgrad_truth)
        self.assertEqual(y.grad, ygrad_truth)

class TestAutogradMatmul(unittest.TestCase): 
    
    def testMatmulMatricesIntermediate(self): 
        x = Tensor.arange(0, 4, 1).reshape([2, 2])
        y = Tensor.arange(4, 8, 1).reshape([2, 2])
        z = x @ y  
        z.backprop(True)

        xgrad_block00_truth = GradTensor([4, 6, 0, 0], [1, 1, 2, 2], 2)
        xgrad_block01_truth = GradTensor([5, 7, 0, 0], [1, 1, 2, 2], 2)
        xgrad_block10_truth = GradTensor([0, 0, 4, 6], [1, 1, 2, 2], 2)
        xgrad_block11_truth = GradTensor([0, 0, 5, 7], [1, 1, 2, 2], 2)

        self.assertEqual(x.grad[0,0,:,:], xgrad_block00_truth) 
        self.assertEqual(x.grad[0,1,:,:], xgrad_block01_truth) 
        self.assertEqual(x.grad[1,0,:,:], xgrad_block10_truth) 
        self.assertEqual(x.grad[1,1,:,:], xgrad_block11_truth) 

    def testMatmulMatricesNoIntermediate(self): 
        x = Tensor.arange(0, 4, 1).reshape([2, 2])
        y = Tensor.arange(4, 8, 1).reshape([2, 2])
        z = x @ y  
        z.backprop(False)

        xgrad_block00_truth = GradTensor([4, 6, 0, 0], [1, 1, 2, 2], 2)
        xgrad_block01_truth = GradTensor([5, 7, 0, 0], [1, 1, 2, 2], 2)
        xgrad_block10_truth = GradTensor([0, 0, 4, 6], [1, 1, 2, 2], 2)
        xgrad_block11_truth = GradTensor([0, 0, 5, 7], [1, 1, 2, 2], 2)

        self.assertEqual(x.grad[0,0,:,:], xgrad_block00_truth) 
        self.assertEqual(x.grad[0,1,:,:], xgrad_block01_truth) 
        self.assertEqual(x.grad[1,0,:,:], xgrad_block10_truth) 
        self.assertEqual(x.grad[1,1,:,:], xgrad_block11_truth) 

if __name__ == "__main__": 
  unittest.main(verbosity=2)

