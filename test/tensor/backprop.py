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
        z.backprop(True) 

        grad_truth = GradTensor([1.], [1, 1], 1)
        
        self.assertEqual(x.grad, grad_truth)
        self.assertEqual(y.grad, grad_truth)

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
        z.backprop(True)
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

        self.assertEqual(z.grad, grad_truth)

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

        self.assertEqual(z.grad, grad_truth)

    def testSumTensorsIntermediate(self): 
        x = Tensor.arange(0, 24, 1).reshape([2, 3, 4])
        y = Tensor.arange(0, 24, 1).reshape([2, 3, 4])

        z = x + y 
        z.backprop(False)
        grad_truth = GradTensor([2, 3, 4, 2, 3, 4], 3)
        combos = product(range(2), range(3), range(4)) 
        for i, j, k in combos: 
            grad_truth[i, j, k, i, j, k] = 1.0

        self.assertEqual(z.grad, grad_truth)

class TestAutogradSub(unittest.TestCase): 
    
    def sumScalars(self): 
        pass

    def sumVectors(self): 
        pass

    def sumMatrices(self): 
        pass

    def sumTensors(self): 
        pass

class TestAutogradMul(unittest.TestCase): 
    
    def sumScalars(self): 
        pass

    def sumVectors(self): 
        pass

    def sumMatrices(self): 
        pass

    def sumTensors(self): 
        pass

class TestAutogradMatmul(unittest.TestCase): 
    
    def sumScalars(self): 
        pass

    def sumVectors(self): 
        pass

    def sumMatrices(self): 
        pass

    def sumTensors(self): 
        pass

if __name__ == "__main__": 
  unittest.main(verbosity=2)

