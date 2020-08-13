#include "opengl/nn/kernels/kernel_test_utils.h"


using namespace ::opengl::testing;

namespace opengl{
    TEST(KernelTestUtilsTest, CopyTest){
        auto ctx = GetContext();
        const IntList shape{1, 2, 1, 3};
        // cpu tensor
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto dst_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));

        Tensor*  src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor*  src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor*  dst_cpu_tensor = dst_cpu_tensor_ptr.get();

        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        CopyDeviceTensorToCPU(src_gpu_tensor, dst_cpu_tensor);

        CheckSameTensor(src_cpu_tensor, dst_cpu_tensor);
    }
}
