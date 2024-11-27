#include <gtest/gtest.h>
#include "../../src/Tensor.h"

TEST(TensorTest, Index) {
  Tensor t = Tensor({1., 2., 3., 4.}, {2, 2});

  ASSERT_EQ(t.at({0, 0}), 1.);
  ASSERT_EQ(t.at({0, 1}), 2.);
  ASSERT_EQ(t.at({1, 0}), 3.);
  ASSERT_EQ(t.at({1, 1}), 4.);
}

TEST(TensorTest, Slice1D) {
  Tensor t = Tensor({1., 2., 3., 4.}, {2, 2});
  Tensor r1 = Tensor({1., 2.}, {1, 2});
  Tensor r2 = Tensor({3., 4.}, {1, 2});
  std::vector<Tensor::Slice> r1s = {Tensor::Slice(0, 1, 1), Tensor::Slice()};
  std::vector<Tensor::Slice> r2s = {Tensor::Slice(1, 2, 1), Tensor::Slice()};

  Tensor c1 = Tensor({1., 3.}, {2, 1});
  Tensor c2 = Tensor({2., 4.}, {2, 1});
  std::vector<Tensor::Slice> c1s = {Tensor::Slice(), Tensor::Slice(0, 1, 1)}; 
  std::vector<Tensor::Slice> c2s = {Tensor::Slice(), Tensor::Slice(1, 2, 1)}; 

  ASSERT_TRUE(*t.slice(r1s) == r1);
  ASSERT_TRUE(*t.slice(r2s) == r2);
  ASSERT_TRUE(*t.slice(c1s) == c1);
  ASSERT_TRUE(*t.slice(c2s) == c2);
}

TEST(TensorTest, Slice2D) {
  Tensor* t = Tensor::arange(0, 24)->reshape({2, 3, 4}); 
  Tensor* b1 = Tensor::arange(0, 12)->reshape({1, 3, 4});
  Tensor* b2 = Tensor::arange(12, 24)->reshape({1, 3, 4}); 
  std::vector<Tensor::Slice> b1s = {Tensor::Slice(0, 1, 1), 
    Tensor::Slice(), Tensor::Slice()};
  std::vector<Tensor::Slice> b2s = {Tensor::Slice(1, 2, 1), 
    Tensor::Slice(), Tensor::Slice()};

  ASSERT_TRUE(*t->slice(b1s) == *b1);
  ASSERT_TRUE(*t->slice(b2s) == *b2);

  Tensor r1 = Tensor({0., 1., 2., 3., 12., 13., 14., 15.}, {2, 1, 4});
  Tensor r2 = Tensor({4., 5., 6., 7., 16., 17., 18., 19.}, {2, 1, 4});
  Tensor r3 = Tensor({8., 9., 10., 11., 20., 21., 22., 23.}, {2, 1, 4});
  std::vector<Tensor::Slice> r1s = {Tensor::Slice(), 
    Tensor::Slice(0, 1, 1), Tensor::Slice()};
  std::vector<Tensor::Slice> r2s = {Tensor::Slice(), 
    Tensor::Slice(1, 2, 1), Tensor::Slice()};
  std::vector<Tensor::Slice> r3s = {Tensor::Slice(), 
    Tensor::Slice(2, 3, 1), Tensor::Slice()};

  ASSERT_TRUE(*t->slice(r1s) == r1);
  ASSERT_TRUE(*t->slice(r2s) == r2);
  ASSERT_TRUE(*t->slice(r3s) == r3);

  Tensor c1 = Tensor({0., 4., 8., 12., 16., 20.}, {2, 3, 1});
  Tensor c2 = Tensor({1., 5., 9., 13., 17., 21.}, {2, 3, 1});
  Tensor c3 = Tensor({2., 6., 10., 14., 18., 22.}, {2, 3, 1});
  Tensor c4 = Tensor({3., 7., 11., 15., 19., 23.}, {2, 3, 1});
  std::vector<Tensor::Slice> c1s = {Tensor::Slice(), 
    Tensor::Slice(), Tensor::Slice(0, 1, 1)};
  std::vector<Tensor::Slice> c2s = {Tensor::Slice(), 
    Tensor::Slice(), Tensor::Slice(1, 2, 1)};
  std::vector<Tensor::Slice> c3s = {Tensor::Slice(), 
    Tensor::Slice(), Tensor::Slice(2, 3, 1)};
  std::vector<Tensor::Slice> c4s = {Tensor::Slice(), 
    Tensor::Slice(), Tensor::Slice(3, 4, 1)};

  ASSERT_TRUE(*t->slice(c1s) == c1);
  ASSERT_TRUE(*t->slice(c2s) == c2);
  ASSERT_TRUE(*t->slice(c3s) == c3);
  ASSERT_TRUE(*t->slice(c4s) == c4);
}

TEST(TensorTest, Slice3D) {
}

TEST(TensorTest, Transpose) {
  Tensor* t1 = Tensor::arange(0, 6)->reshape({2, 3}); 
  Tensor* t2 = t1->transpose();
  Tensor truth = Tensor({0., 3., 1., 4., 2., 5.}, {3, 2});

  ASSERT_TRUE(*t2 == truth);
}


