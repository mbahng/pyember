#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(UtilsTest, Equality) {
  GradTensor t1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 
  GradTensor t2 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 
  GradTensor t3 = GradTensor({1., 2., 3., 4.}, {2, 2}, 0); 
  GradTensor t4 = GradTensor({1., 2., 3., 4., 5., 6.}, {3, 2}, 1); 
  GradTensor t5 = GradTensor({1., 2., 3., 5.}, {2, 2}, 1); 

  ASSERT_TRUE(t1 == t2);
  ASSERT_TRUE(t1 != t3);
  ASSERT_TRUE(t1 != t4);
  ASSERT_TRUE(t1 != t5);
}

TEST(UtilsTest, Copy) {
  GradTensor t1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
  GradTensor* t2 = t1.copy(); 

  // They are copies
  ASSERT_TRUE(t1 == *t2); 
  // But do not live in the same memory address 
  ASSERT_TRUE(&t1 != t2);
}

TEST(UtilsTest, Index) {
  GradTensor t1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 

  ASSERT_TRUE(t1.at({0, 0}) == 1.); 
  ASSERT_TRUE(t1.at({0, 1}) == 2.); 
  ASSERT_TRUE(t1.at({1, 0}) == 3.); 
  ASSERT_TRUE(t1.at({1, 1}) == 4.); 
}

TEST(UtilsTest, Pivot) {
  GradTensor t1 = GradTensor({1., 2., 3., 4.}, {2, 2}, 1); 
  GradTensor t2 = GradTensor({1., 2., 3., 4.}, {1, 2, 2}, 2); 
  GradTensor t3 = GradTensor({1., 2., 3., 4.}, {1, 1, 2, 2}, 3); 
  GradTensor t4 = GradTensor({1., 2., 3., 4.}, {2, 1, 2, 1}, 2); 

  ASSERT_TRUE(t1.pidx() == 1);    
  ASSERT_TRUE(t2.pidx() == 2);    
  ASSERT_TRUE(t3.pidx() == 3);    
  ASSERT_TRUE(t4.pidx() == 2);    
}

TEST(GradTensorTest, Index) {
  GradTensor t = GradTensor({1., 2., 3., 4.}, {2, 2}, 1);

  ASSERT_EQ(t.at({0, 0}), 1.);
  ASSERT_EQ(t.at({0, 1}), 2.);
  ASSERT_EQ(t.at({1, 0}), 3.);
  ASSERT_EQ(t.at({1, 1}), 4.);
}

TEST(GradTensorTest, Slice1D) {
  GradTensor t = GradTensor({1., 2., 3., 4.}, {2, 2}, 1);
  GradTensor r1 = GradTensor({1., 2.}, {1, 2}, 1);
  GradTensor r2 = GradTensor({3., 4.}, {1, 2}, 1);
  std::vector<GradTensor::Slice> r1s = {Tensor::Slice(0, 1, 1), Tensor::Slice()};
  std::vector<GradTensor::Slice> r2s = {Tensor::Slice(1, 2, 1), Tensor::Slice()};

  GradTensor c1 = GradTensor({1., 3.}, {2, 1}, 1);
  GradTensor c2 = GradTensor({2., 4.}, {2, 1}, 1);
  std::vector<GradTensor::Slice> c1s = {Tensor::Slice(), Tensor::Slice(0, 1, 1)}; 
  std::vector<GradTensor::Slice> c2s = {Tensor::Slice(), Tensor::Slice(1, 2, 1)}; 

  ASSERT_TRUE(*t.slice(r1s) == r1);
  ASSERT_TRUE(*t.slice(r2s) == r2);
  ASSERT_TRUE(*t.slice(c1s) == c1);
  ASSERT_TRUE(*t.slice(c2s) == c2);
}

TEST(GradTensorTest, Slice2D) {
  // GradTensor does not and should not support arange constructor
  for (size_t pidx = 1; pidx < 3; pidx++) {
    GradTensor t = GradTensor(
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0},
      {2, 3, 4}, 1);

    GradTensor b1 = GradTensor(
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}, 
      {1, 3, 4}, 1); 

    GradTensor b2 = GradTensor(
      {12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0},
      {1, 3, 4}, 1); 

    std::vector<GradTensor::Slice> b1s = {GradTensor::Slice(0, 1, 1), 
      GradTensor::Slice(), GradTensor::Slice()};
    std::vector<GradTensor::Slice> b2s = {GradTensor::Slice(1, 2, 1), 
      GradTensor::Slice(), GradTensor::Slice()};

    ASSERT_TRUE(*t.slice(b1s) == b1); 
    ASSERT_TRUE(*t.slice(b2s) == b2); 

    GradTensor r1 = GradTensor({0., 1., 2., 3., 12., 13., 14., 15.}, {2, 1, 4}, 1);
    GradTensor r2 = GradTensor({4., 5., 6., 7., 16., 17., 18., 19.}, {2, 1, 4}, 1);
    GradTensor r3 = GradTensor({8., 9., 10., 11., 20., 21., 22., 23.}, {2, 1, 4}, 1);

    std::vector<Tensor::Slice> r1s = {Tensor::Slice(), 
      Tensor::Slice(0, 1, 1), Tensor::Slice()};
    std::vector<Tensor::Slice> r2s = {Tensor::Slice(), 
      Tensor::Slice(1, 2, 1), Tensor::Slice()};
    std::vector<Tensor::Slice> r3s = {Tensor::Slice(), 
      Tensor::Slice(2, 3, 1), Tensor::Slice()};

    ASSERT_TRUE(*t.slice(r1s) == r1);
    ASSERT_TRUE(*t.slice(r2s) == r2);
    ASSERT_TRUE(*t.slice(r3s) == r3);

    GradTensor c1 = GradTensor({0., 4., 8., 12., 16., 20.}, {2, 3, 1}, 1);
    GradTensor c2 = GradTensor({1., 5., 9., 13., 17., 21.}, {2, 3, 1}, 1);
    GradTensor c3 = GradTensor({2., 6., 10., 14., 18., 22.}, {2, 3, 1}, 1);
    GradTensor c4 = GradTensor({3., 7., 11., 15., 19., 23.}, {2, 3, 1}, 1);
    std::vector<GradTensor::Slice> c1s = {GradTensor::Slice(), 
      GradTensor::Slice(), GradTensor::Slice(0, 1, 1)};
    std::vector<GradTensor::Slice> c2s = {GradTensor::Slice(), 
      GradTensor::Slice(), GradTensor::Slice(1, 2, 1)};
    std::vector<GradTensor::Slice> c3s = {GradTensor::Slice(), 
      GradTensor::Slice(), GradTensor::Slice(2, 3, 1)};
    std::vector<GradTensor::Slice> c4s = {GradTensor::Slice(), 
      GradTensor::Slice(), GradTensor::Slice(3, 4, 1)};

    ASSERT_TRUE(*t.slice(c1s) == c1);
    ASSERT_TRUE(*t.slice(c2s) == c2);
    ASSERT_TRUE(*t.slice(c3s) == c3);
    ASSERT_TRUE(*t.slice(c4s) == c4);


  }
}

TEST(GradTensorTest, Slice3D) {

}

TEST(GradTensorTest, Transpose) {
  GradTensor t1 = GradTensor({0., 1., 2., 3., 4., 5.}, {2, 3}, 1);
  GradTensor t2 = t1.transpose();
  GradTensor truth = GradTensor({0., 3., 1., 4., 2., 5.}, {3, 2}, 1);

  ASSERT_TRUE(t2 == truth);
}


