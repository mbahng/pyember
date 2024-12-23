#include <gtest/gtest.h> 
#include "../../src/Tensor.h"
#include "../../src/utils.h"
#include <vector>

namespace T_Add_T {

  TEST(TensorAlgebraTest, AddTensorVector) {
    Tensor* t1 = Tensor::arange(0, 5);
    Tensor* t2 = Tensor::arange(0, 5); 
    Tensor* s1 = t1->add(t2);
    Tensor* truth_sum = Tensor::arange(0, 10, 2);
    ASSERT_TRUE(*s1 == *truth_sum); 
    s1->backprop(true);
    GradTensor* truth = GradTensor::eye(5, 0, 1);
    ASSERT_TRUE(*(t1->grad) == *truth); 
    ASSERT_TRUE(*(t2->grad) == *truth); 
    delete t1;
    delete t2;
    delete s1;
    delete truth_sum;
    delete truth;
  }

}

namespace T_Sub_T {

  TEST(TensorAlgebraTest, SubTensor) {
    Tensor* t1 = Tensor::arange(0, 5);
    Tensor* t2 = Tensor::arange(0, 5); 
    Tensor* s1 = t1->sub(t2);
    Tensor* truth_sum = Tensor::zeros({5});
    
    ASSERT_TRUE(*s1 == *truth_sum); 
    
    s1->backprop(true);
    GradTensor* truth1 = GradTensor::eye(5, 0, 1);
    GradTensor* truth2 = GradTensor::eye(5, 0, 1); 
    for (int i = 0; i < truth2->storage_.size(); i++) {
      truth2->storage_[i] *= -1;
    }
    ASSERT_TRUE(*(t1->grad) == *truth1); 
    ASSERT_TRUE(*(t2->grad) == *truth2); 

    delete t1;
    delete t2;
    delete s1;
    delete truth_sum;
    delete truth1;
    delete truth2;
  }

}

namespace T_Mul_T {

  TEST(TensorAlgebraTest, MulTensor) {
    // Empty test placeholder
  }

}

namespace T_Matmul_T {

  TEST(TensorAlgebraTest, NB_matmul_NB) { 
    Tensor* t1 = Tensor::arange(0, 10, 1)->reshape({5, 2}, true); 
    Tensor* t2 = Tensor::arange(0, 6, 1)->reshape({2, 3}, true); 
    Tensor* prod = t1->matmul(t2); 
    Tensor* gt = new Tensor(
      Index::concat(
        CIntegrity::range(3, 6, 1), 
        CIntegrity::range(9, 20, 5), 
        CIntegrity::range(15, 34, 9), 
        CIntegrity::range(21, 48, 13), 
        CIntegrity::range(27, 62, 17)
      ), 
      {5, 3}, true
    ); 

    ASSERT_TRUE(*prod == *gt);
  }

  TEST(TensorAlgebraTest, B_matmul_NB) { 
    Tensor* t1 = new Tensor(
      Index::concat(CIntegrity::range(0, 10, 1), CIntegrity::range(0, 10, 1)), 
      {2, 5, 2}, true
    );
    Tensor* t2 = Tensor::arange(0, 6, 1)->reshape({2, 3}, true); 
    Tensor* prod = t1->matmul(t2); 
    Tensor* gt = new Tensor(
      Index::concat(
        CIntegrity::range(3, 6, 1), CIntegrity::range(9, 20, 5), CIntegrity::range(15, 34, 9), CIntegrity::range(21, 48, 13), CIntegrity::range(27, 62, 17), 
        CIntegrity::range(3, 6, 1), CIntegrity::range(9, 20, 5), CIntegrity::range(15, 34, 9), CIntegrity::range(21, 48, 13), CIntegrity::range(27, 62, 17)
      ), 
      {2, 5, 3}, true
    );
    ASSERT_TRUE(*prod == *gt);
  }
  

}

namespace T_Add_GT {

  TEST(TensorAlgebraTest, AddGradTensorVector) {
    // Empty test placeholder
  }

}

namespace T_Sub_GT {

  TEST(TensorAlgebraTest, SubGradTensor) {
    // Empty test placeholder
  }

}

namespace T_Mul_GT {

  TEST(TensorAlgebraTest, MulGradTensor) {
    // Empty test placeholder
  }

}

