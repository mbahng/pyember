import unittest
from ember import Tensor 

class TestAddSubtract(unittest.TestCase): 

    def testAddSubtractSimple(self): 

        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = Tensor([[4, 1, 2], [4, 5, 6]])
        sum = Tensor([[5, 3, 5], [8, 10, 12]])
        diff = Tensor([[-3, 1, 1], [0, 0, 0]])

        self.assertEqual(x + y, sum)
        self.assertEqual(x - y, diff)
        self.assertEqual(y - x, -1 * diff)

    def testAddSubtractLarge(self): 
        pass 

    def testAddSubtractBroadcasting(self): 
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = Tensor([1, 2, 3])
        sum = Tensor([[2, 4, 6], [5, 7, 9]])
        diff = Tensor([[0, 0, 0], [3, 3, 3]])
        self.assertEqual(x + y, sum)
        self.assertEqual(x - y, diff)

    def testAddSubtractScalar(self): 
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        c = 2
        sum = Tensor([[3, 4, 5], [6, 7, 8]])
        diff = Tensor([[-1, 0, 1], [2, 3, 4]])
        self.assertEqual(x + c, sum)
        self.assertEqual(x - c, diff)

class TestMultiplication(unittest.TestCase): 

    def testMatrixMult(self): 
        x = Tensor([[1, 2], [3, 4]])
        y = Tensor([[5, 6], [7, 8]])
        result = Tensor([[19, 22], [43, 50]])
        self.assertEqual(x @ y, result)

        # Test with non-square matrices
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([[7, 8], [9, 10], [11, 12]])
        result = Tensor([[58, 64], [139, 154]])
        self.assertEqual(a @ b, result)

if __name__ == "__main__": 
    unittest.main()


