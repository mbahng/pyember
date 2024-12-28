#include "../../src/Tensor.h"
#include <vector>

TEST(TensorInitialization, StandardInitialization) {
  std::vector<size_t> shape = {2, 5}; 
  std::vector<double> storage = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}; 
  Tensor t1 = Tensor(storage, shape); 
  EXPECT_EQ(t1.storage(), storage);
  EXPECT_EQ(t1.shape(), shape);
}

TEST(TensorInitialization, ArrayInitialization) {
  std::vector<double> storage = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  std::vector<size_t> shape = {10}; 
  Tensor t1 = Tensor(storage); 
  EXPECT_EQ(t1.storage(), storage);
  EXPECT_EQ(t1.shape(), shape);
}

TEST(TensorInitialization, MatrixInitialization) {
  std::vector<std::vector<double>> storage1 = {{1., 2., 3., 4., 5.}, {6., 7., 8., 9., 10.}}; 
  std::vector<double> storage1_flattened = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.}; 
  std::vector<size_t> shape1 = {2, 5}; 
  Tensor t1 = Tensor(storage1); 
  EXPECT_EQ(t1.storage(), storage1_flattened);
  EXPECT_EQ(t1.shape(), shape1);

  std::vector<std::vector<double>> storage2 = {{2., 3.}, {4., 5.}, {7., 8.}, {9., 10}}; 
  std::vector<double> storage2_flattened = {2., 3., 4., 5., 7., 8., 9., 10}; 
  std::vector<size_t> shape2 = {4, 2}; 
  Tensor t2 = Tensor(storage2); 
  EXPECT_EQ(t2.storage(), storage2_flattened);
  EXPECT_EQ(t2.shape(), shape2);
}

TEST(TensorInitialization, TensorInitialization) {
  // 3-tensor initialization
  std::vector<std::vector<std::vector<double>>> storage1 = {
    {{1., 2., 3., 4., 5.}, {6., 7., 8., 9., 10.}}, 
    {{1., 2., 3., 4., 5.}, {6., 7., 8., 9., 10.}}
  };
  std::vector<double> storage1_flattened = {
    1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
    1., 2., 3., 4., 5., 6., 7., 8., 9., 10.
  }; 
  std::vector<size_t> shape1 = {2, 2, 5}; 
  Tensor t1 = Tensor(storage1); 
  EXPECT_EQ(t1.storage(), storage1_flattened);
  EXPECT_EQ(t1.shape(), shape1);

  std::vector<std::vector<std::vector<double>>> storage2 = {
    {{2., 3.}, {4., 5.}, {7., 8.}, {9., 10.}}, 
    {{2., 3.}, {4., 5.}, {7., 8.}, {9., 10.}}
  };
  std::vector<double> storage2_flattened = {
    2., 3., 4., 5., 7., 8., 9., 10.,
    2., 3., 4., 5., 7., 8., 9., 10.
  }; 
  std::vector<size_t> shape2 = {2, 4, 2}; 
  Tensor t2 = Tensor(storage2); 
  EXPECT_EQ(t2.storage(), storage2_flattened);
  EXPECT_EQ(t2.shape(), shape2);
}

TEST(TensorInitialization, ArangeInitialization) {
  Tensor* t1 = Tensor::arange(10, 20); 
  Tensor* t2 = Tensor::arange(10, 20, 1); 
  ASSERT_TRUE(t1 == t2);
  EXPECT_EQ(t1->storage()[0], 10);
  EXPECT_EQ(t1->storage()[t1->storage().size() - 1], 19);
  
  Tensor* t3 = Tensor::arange(0, 100, 3);
  EXPECT_EQ(t3->storage()[0], 0);
  EXPECT_EQ(t3->storage()[t3->storage().size() - 1], 99);
  
  Tensor* t4 = Tensor::arange(109, 1095, 7); 
  EXPECT_EQ(t4->storage()[0], 109);
  EXPECT_EQ(t4->storage()[t4->storage().size() - 1], 1089);

  delete t1;
  delete t2;
  delete t3;
  delete t4;
}

TEST(TensorInitialization, LinSpaceInitialization) {
  std::vector<size_t> shape = {5, 5}; 
  // Empty test
}

TEST(TensorInitialization, GaussianInitialization) {
  std::vector<size_t> shape = {5, 5}; 
  Tensor* g1 = Tensor::gaussian(shape); 
  Tensor* g2 = Tensor::gaussian(shape, 1.0, 2.0); 
  Tensor* g3 = Tensor::gaussian(shape, -1.9); 
  
  EXPECT_EQ(g1->shape(), shape);
  EXPECT_EQ(g2->shape(), shape);
  EXPECT_EQ(g3->shape(), shape);

  delete g1;
  delete g2;
  delete g3;
}

TEST(TensorInitialization, UniformInitialization) {
  std::vector<size_t> shape = {5, 5}; 
  Tensor* u1 = Tensor::uniform(shape); 
  Tensor* u2 = Tensor::uniform(shape); 
  Tensor* u3 = Tensor::uniform(shape); 
  
  EXPECT_EQ(u1->shape(), shape);
  EXPECT_EQ(u2->shape(), shape);
  EXPECT_EQ(u3->shape(), shape);

  delete u1;
  delete u2;
  delete u3;
}

TEST(TensorInitialization, OnesInitialization) {
  std::vector<size_t> shape = {3, 4}; 
  Tensor* o1 = Tensor::ones(shape);
  
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) { 
      EXPECT_EQ(o1->at({i, j}), 1.0); 
    }
  }

  delete o1;
}

TEST(TensorInitialization, ZerosInitialization) {
  std::vector<size_t> shape = {3, 4}; 
  Tensor* o1 = Tensor::zeros(shape);
  
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) { 
      EXPECT_EQ(o1->at({i, j}), 0.0); 
    }
  }

  delete o1;
}
