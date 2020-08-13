#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/core/driver.h"
#include "opengl/core/context.h"
#include "opengl/core/tensor_allocator.h"


using namespace ::opengl::testing;
namespace opengl{
    TEST(ContextTest, CopyBufferToImage){
    }

    TEST(ContextTest, CopyImageToBuffer){
    }


    TEST(ContextTest, NHWCAndNHWC4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NHWC4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();

        ctx->CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        ctx->CopyDeviceTensorToCPU(src_gpu_tensor, actual_cpu_tensor);

        CheckSameTensor(src_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(ContextTest, NCHWAndHWN4C4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NCHW));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NCHW));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::HWN4C4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();

        ctx->CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        ctx->CopyDeviceTensorToCPU(src_gpu_tensor, actual_cpu_tensor);

        CheckSameTensor(src_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(ContextTest, ANYAndANY4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();

        ctx->CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        ctx->CopyDeviceTensorToCPU(src_gpu_tensor, actual_cpu_tensor);

        CheckSameTensor(src_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(ContextTest, CPUUsageTest){
        auto ctx = GetContext();
        auto a = TensorPoolAllocator();
        while(true){
            IntList shape{1, 320, 320, 3};
            Tensor* src_cpu_tensor = a.AllocateTensor(shape, Tensor::HOST_MEMORY, dlxnet::TensorProto::ANY);
            Tensor* src_gpu_tensor = a.AllocateTensor(shape, Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY);
            // ctx->CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
            a.DeallocateTensor(src_cpu_tensor);
            a.DeallocateTensor(src_gpu_tensor);
        }
    }

}

