import unittest
from ember import GradTensor

class TestGradTensorSlice(unittest.TestCase): 

  def testGetterVector(self): 
    x = GradTensor([1, 2, 3, 4], [4], 0, 1)  
    self.assertEqual(x[:], x) 
    self.assertEqual(x[0:2], GradTensor([1, 2], [2], 0, 1))
    self.assertEqual(x[1:3], GradTensor([2, 3], [2], 0, 1))

  def testGetterMatrix(self): 
    x = GradTensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], 0, 1)  
    self.assertEqual(x[0,:], GradTensor([1, 2, 3], [1, 3], 0, 1))
    self.assertEqual(x[1,:], GradTensor([4, 5, 6], [1, 3], 0, 1))
    self.assertEqual(x[2,:], GradTensor([7, 8, 9], [1, 3], 0, 1))
    self.assertEqual(x[:,0], GradTensor([1, 4, 7], [3, 1], 0, 1))
    self.assertEqual(x[:,1], GradTensor([2, 5, 8], [3, 1], 0, 1))
    self.assertEqual(x[:,2], GradTensor([3, 6, 9], [3, 1], 0, 1))
    self.assertEqual(x[:2,:2], GradTensor([1, 2, 4 ,5], [2, 2], 0, 1))

if __name__ == "__main__": 
  unittest.main(verbosity=2)
