import unittest
from ember import GradTensor

class TestGradTensorCopy(unittest.TestCase): 

  def testShallowCopy(self): 

    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1) 
    y = x.copy() 
    z = x.shallowcopy() 
    
    self.assertEqual(x, y)
    self.assertEqual(x, z) 
    x[0, 0] = 5 # underlying data is shared 
    pass 
    # self.assertEqual(y[0, 0], 5)
    # self.assertEqual(z[0, 0], 5) 
    # y.bidx = 1 
    # self.assertEqual(x.bidx, 1)
    # self.assertEqual(z.bidx, 1)

  def testDeepCopy(self): 

    x = GradTensor([1, 2, 3, 4], [2, 2], 0, 1) 
    y = x.deepcopy()  
    self.assertEqual(x, y) 
    x[0, 0] = 5 # underlying data is not shared 
    self.assertEqual(y[0, 0].item(), 1)
    y.bidx = 1 
    self.assertEqual(x.bidx, 0)

class TestGradTensorReshape(unittest.TestCase): 
  
  def testVectorReshape(self): 
    x = GradTensor([1, 2, 3, 4, 5, 6], [6], 0, 1) 
    y = x.reshape([3, 2]) 
    self.assertEqual(y.shape, [3, 2]) 
    self.assertEqual(x.storage, y.storage)

  def testMatrixReshape(self): 
    x = GradTensor([1, 2, 3, 4, 5, 6], [2, 3], 0, 1) 
    y = x.reshape([3, 2]) 
    self.assertEqual(y.shape, [3, 2]) 
    self.assertEqual(x.storage, y.storage)

  def testTensorReshape(self): 
    x = GradTensor([1, 2, 3, 4, 5, 6], [2, 1, 3], 0, 1) 
    y = x.reshape([3, 2, 1]) 
    self.assertEqual(y.shape, [3, 2, 1]) 
    self.assertEqual(x.storage, y.storage)

class TestGradTensorTranspose(unittest.TestCase): 

  def testTranposeVector(self): 
    x = GradTensor([1, 2, 3, 4, 5, 6], [6], 0, 1) 
    y = x.transpose() 
    self.assertEqual(y, GradTensor([1, 2, 3, 4, 5, 6], [6, 1], 0, 1))

  def testTranposeMatrix(self): 
    x = GradTensor([1, 2, 3, 4, 5, 6], [2, 3], 0, 1) 
    y = x.transpose() 
    z = x.transpose(0, 1) 

    self.assertEqual(y.shape, [3, 2])
    self.assertEqual(z.shape, [3, 2]) 
    self.assertEqual(y.storage, [1, 4, 2, 5, 3, 6])
    self.assertEqual(z.storage, [1, 4, 2, 5, 3, 6])
    self.assertEqual(y, z)


  def testTransposeTensor(self): 
    x = GradTensor(list(range(12)), [2, 2, 3], 1, 2) 
    y = x.transpose() 
    gt = GradTensor(
      [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11], 
      [2, 3, 2], 1, 2
    )
    self.assertEqual(y, gt)

if __name__ == "__main__": 
  unittest.main(verbosity=2)

