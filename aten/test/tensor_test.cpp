#include <gtest/gtest.h>
#include "../src/Tensor.h"
#include <vector>

TEST(TensorTest, Creation) {
    std::vector<double> data = {2.0};
    std::vector<int64_t> shape = {1};
    Tensor t(data, shape);
    EXPECT_EQ(t.data()[0], 2.0);
}
