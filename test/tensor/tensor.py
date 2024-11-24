import unittest
from ember import Tensor, GradTensor

class TestTensorConstructors(unittest.TestCase): 

    def testDefaultConstructor(self): 
        t = Tensor([1, 2, 3, 4], [2, 2]) 
        self.assertEqual(t.data(), [1, 2, 3, 4]) 
        self.assertEqual(t.shape, [2, 2])

    def test1DConstructor(self): 
        t = Tensor([1, 2, 3, 4])
        self.assertEqual(t.data(), [1, 2, 3, 4]) 
        self.assertEqual(t.shape, [4])

    def test2DConstructor(self): 
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t.data(), [1, 2, 3, 4]) 
        self.assertEqual(t.shape, [2, 2])

    def test3DConstructor(self): 
        t = Tensor([
            [[1, 2], [3, 4]], 
            [[5, 6], [7, 8]]
            ])
        self.assertEqual(t.data(), [1, 2, 3, 4, 5, 6, 7, 8]) 
        self.assertEqual(t.shape, [2, 2, 2])

    def testArangeConstructor(self): 
        t = Tensor.arange(0, 10, 2)
        self.assertEqual(t.data(), [0, 2, 4, 6, 8])
        self.assertEqual(t.shape, [5])

    def testLinspaceConstructor(self): 
        t = Tensor.linspace(0, 5, 6) 
        self.assertEqual(t.data(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(t.shape, [6])

    def testGaussianConstructor(self): 
        t = Tensor.gaussian([3, 3], 0, 1)
        self.assertEqual(t.shape, [3, 3])

    def testUniformConstructor(self): 
        t = Tensor.uniform([2, 2], 0, 1)
        self.assertEqual(t.shape, [2, 2])

    def testOnesConstructor(self): 
        t = Tensor.ones([2, 3, 4])
        self.assertEqual(t.data(), [1] * 24)
        self.assertEqual(t.shape, [2, 3, 4])

    def testZerosConstructor(self): 
        t = Tensor.zeros([2, 3, 4])
        self.assertEqual(t.data(), [0] * 24)
        self.assertEqual(t.shape, [2, 3, 4])

class TestTensorAlgebra(unittest.TestCase): 

    def testAddWithTensor(self): 
        x = Tensor([1, 2, 3, 4]) 
        y = Tensor([1, 2, 3, 4]) 
        z = x + y 
        
        self.assertTrue(z == Tensor([2, 4, 6, 8]))

    def testAddWithGradTensor(self): 
        x = Tensor([1, 2, 3, 4]) 
        y = GradTensor([1, 2, 3, 4], [4], 0) 
        z = x + y 
        
        self.assertTrue(z == Tensor([2, 4, 6, 8]))

    def testSubWithTensor(self): 
        x = Tensor([1, 2, 3, 4]) 
        y = Tensor([1, 2, 3, 4])
        z = x - y 
        
        self.assertTrue(z == Tensor([0, 0, 0, 0])) 

    def testSubWithGradTensor(self): 
        x = Tensor([1, 2, 3, 4]) 
        y = GradTensor([1, 2, 3, 4], [4], 0) 
        z = x - y 
        self.assertTrue(z == Tensor([0, 0, 0, 0]))

    def testMulWithTensor(self): 
        x = Tensor([1, 2, 3, 4]) 
        y = Tensor([1, 2, 3, 4]) 
        z = x * y 
        self.assertTrue(z == Tensor([1, 4, 9, 16]))
    
    def testMulWithGradTensor(self): 
        x = Tensor([1, 2, 3, 4]) 
        y = GradTensor([1, 2, 3, 4], [4], 0) 
        z = x * y 
        
        self.assertTrue(z == Tensor([1, 4, 9, 16]))

    def testMatMulWithTensor(self): 
        x = Tensor([1, 2, 3, 4]).reshape([1, 4])
        y = Tensor([1, 2, 3, 4]).reshape([4, 1])
        z = x @ y 
        self.assertTrue(z == Tensor([30], [1, 1]))

class TestTensorUtils(unittest.TestCase): 

    def testTranspose(self): 
        t1 = Tensor.arange(0, 6, 1).reshape([2, 3]) 
        t2 = t1.transpose([0, 1])
        truth = Tensor([0, 3, 1, 4, 2, 5], [3, 2]) 

        self.assertTrue(t2 == truth)

if __name__ == "__main__": 
  unittest.main()

