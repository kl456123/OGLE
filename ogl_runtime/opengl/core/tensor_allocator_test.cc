#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/core/tensor_allocator.h"


using namespace ::opengl::testing;

namespace opengl{
    TEST(TensorAllocatorTest, SimpleTest){
        auto a = TensorPoolAllocator();
        EXPECT_EQ(a.GetTotalSize(), 0);
        EXPECT_EQ(a.GetFreeSize(), 0);
        IntList shape{1, 3, 320, 320};
        Tensor* t = a.AllocateTensor(shape, Tensor::DEVICE_TEXTURE,
                dlxnet::TensorProto::ANY);
        EXPECT_EQ(a.GetTotalSize(), 1);
        EXPECT_EQ(a.GetFreeSize(), 0);
        a.DeallocateTensor(t);
        EXPECT_EQ(a.GetTotalSize(), 1);
        EXPECT_EQ(a.GetFreeSize(), 1);

        // alloc the same tensor again
        Tensor* t2 = a.AllocateTensor(shape, Tensor::DEVICE_TEXTURE,
                dlxnet::TensorProto::ANY);
        EXPECT_EQ(a.GetTotalSize(), 1);
        EXPECT_EQ(a.GetFreeSize(), 0);
        a.DeallocateTensor(t2);
        EXPECT_EQ(a.GetTotalSize(), 1);
        EXPECT_EQ(a.GetFreeSize(), 1);
    }
} // namespace opengl
