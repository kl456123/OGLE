#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/core/functor.h"
#include "opengl/core/cpu_functor.h"
#include "opengl/core/driver.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        void NHWCToANYCPU(const Tensor* src_tensor, Tensor* dst_tensor){
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();
            CHECK_EQ(src_tensor->num_elements(), dst_tensor->num_elements());
            const int num_elements = src_tensor->num_elements();
            for(int i=0;i<num_elements;++i){
                dst_data[i] = src_data[i];
            }
        }

        void ConvertTensorHWN4C4ToNCHW(void* src, Tensor* tensor){
            float* nchw_data = tensor->host<float>();
            float* hwn4c4_data = (float*)src;
            const int num_elements = tensor->num_elements();
            const int c = tensor->shape()[1];
            const int h = tensor->shape()[2];
            const int w = tensor->shape()[3];
            const int n = tensor->shape()[0];
            const int up_channel = UP_DIV(c, 4)*4;
            const int n4 = UP_DIV(n, 4);
            const int c4 = UP_DIV(c, 4);

            for(int i=0; i<num_elements; ++i){
                int cur = i;
                const int w_i = cur%w;
                cur/=w;
                const int h_i = cur%h;
                cur/=h;
                const int c_i = cur%c;
                cur/=c;
                const int n_i = cur;
                const int offset = ((((h_i*w+w_i)*n4+n_i/4)*c4+c_i/4)*4+c_i%4)*4+n_i%4;

                nchw_data[i]=hwn4c4_data[offset];
            }
        }

        void ConvertTensorNCHWToNHWC4(const Tensor* cpu_tensor, void** out){

            const int n = cpu_tensor->shape()[0];
            const int c = cpu_tensor->shape()[1];
            const int h = cpu_tensor->shape()[2];
            const int w = cpu_tensor->shape()[3];

            const int num_elements = n*UP_DIV(c, 4)*4*h*w;
            float* data = new float[num_elements];
            memset(data, 0, sizeof(float)*num_elements);
            float* orig_data = cpu_tensor->host<float>();
            for(int i=0;i<cpu_tensor->num_elements();++i){
                int cur = i;
                const int w_i = cur%w;
                cur/=w;
                const int h_i = cur%h;
                cur/=h;
                const int c_i = cur%c;
                cur/=c;
                const int n_i = cur;
                const int offset = (((n_i*h+h_i)*w+w_i)*UP_DIV(c, 4)+c_i/4)*4+c_i%4;
                data[offset] = orig_data[i];
            }
            *out = data;
        }

    }// namespace

    TEST(FunctorTest, NHWC4ToANY4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape{1, 64, 64, 19};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC4));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NHWC4));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
        auto expected_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));

        Tensor* actual_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_tensor =  expected_tensor_ptr.get();

        // any4 to any4 (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any4 to any (device->device)
        functor::ConvertTensorNHWC4ToANY4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_tensor);

        host_functor::ConvertTensorNHWC4ToANY4()(ctx, src_cpu_tensor, expected_tensor);

        CheckSameTensor(expected_tensor, actual_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, ANY4ToANYTest){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape{2, 5};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto expected_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));

        Tensor* actual_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_tensor =  expected_tensor_ptr.get();

        // any4 to any4 (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any4 to any (device->device)
        functor::ConvertTensorANY4ToANY()(ctx, src_gpu_tensor, dst_gpu_tensor);
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_tensor);

        host_functor::ConvertTensorANY4ToANY()(ctx, src_cpu_tensor, expected_tensor);

        CheckSameTensor(expected_tensor, actual_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, ANYToANY4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape{2, 3};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto expected_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_cpu_tensor = expected_cpu_tensor_ptr.get();

        // any to any (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any to any4 (device->device)
        functor::ConvertTensorANYToANY4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        // any to any (device->host)
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);
        host_functor::ConvertTensorANYToANY4()(ctx, src_cpu_tensor, expected_cpu_tensor);
        // Sync();

        CheckSameTensor(expected_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, ANYToNHWC4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape{1, 2, 2, 2};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NHWC4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC4));
        auto expected_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_cpu_tensor = expected_cpu_tensor_ptr.get();

        // any to any (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any to any4 (device->device)
        functor::ConvertTensorANYToNHWC4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        // any to any (device->host)
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);
        host_functor::ConvertTensorANYToNHWC4()(ctx, src_cpu_tensor, expected_cpu_tensor);
        // Sync();

        CheckSameTensor(expected_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, NCHWToHWN4C4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape = {1,2,3,6};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NCHW));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NCHW));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::HWN4C4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::HWN4C4));
        auto expected_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::HWN4C4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_cpu_tensor = expected_cpu_tensor_ptr.get();

        // any to any (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any to any4 (device->device)
        functor::ConvertTensorNCHWToHWN4C4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        // any to any (device->host)
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);
        host_functor::ConvertTensorNCHWToHWN4C4()(ctx, src_cpu_tensor, expected_cpu_tensor);
        // Sync();

        CheckSameTensor(expected_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, DataLayoutTest){
        auto ctx = GetContext();

        // DIFFERENT_SHAPE_LOOP_START;
        IntList shape = {1,2,3,6};
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto dst_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        Tensor* actual_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_cpu_tensor = dst_cpu_tensor_ptr.get();

        functor::ConvertTensorTest()(ctx, src_gpu_tensor, actual_tensor);
        CopyDeviceTensorToCPU(actual_tensor, dst_cpu_tensor);
    }

    TEST(FunctorTest, HWN4C4ToNCHWTest){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape = {1,2,3,6};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::HWN4C4));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::HWN4C4));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NCHW));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NCHW));
        auto expected_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NCHW));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_cpu_tensor = expected_cpu_tensor_ptr.get();

        // any to any (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any to any4 (device->device)
        functor::ConvertTensorHWN4C4ToNCHW()(ctx, src_gpu_tensor, dst_gpu_tensor);
        // any to any (device->host)
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);
        host_functor::ConvertTensorHWN4C4ToNCHW()(ctx, src_cpu_tensor, expected_cpu_tensor);
        // Sync();

        CheckSameTensor(expected_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }


    TEST(FunctorTest, SSDBBoxDecoderTest){
        auto ctx = GetContext();
        const std::vector<float> variances{0.1, 0.1, 0.2, 0.2};
        IntList shape = {1, 100, 4};

        auto cpu_anchors = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto gpu_anchors = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));

        auto cpu_box_preds = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto gpu_box_preds = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));

        auto cpu_decoded_box = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto gpu_decoded_box = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));

        auto actual_cpu_tensor = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto expected_cpu_tensor = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));

        /// gpu version
        CopyCPUTensorToDevice(cpu_anchors.get(), gpu_anchors.get());
        CopyCPUTensorToDevice(cpu_box_preds.get(), gpu_box_preds.get());
        functor::SSDBBoxDecoder()(ctx, gpu_box_preds.get(), gpu_anchors.get(), gpu_decoded_box.get(), variances);
        CopyDeviceTensorToCPU(gpu_decoded_box.get(), actual_cpu_tensor.get());

        // cpu version
        host_functor::SSDBBoxDecoder()(ctx, cpu_box_preds.get(), cpu_anchors.get(), expected_cpu_tensor.get(),
                variances);

        CheckSameTensor(actual_cpu_tensor.get(), expected_cpu_tensor.get());
    }
}// namespace opengl
