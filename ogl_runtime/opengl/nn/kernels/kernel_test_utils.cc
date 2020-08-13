#include "opengl/nn/kernels/kernel_test_utils.h"


namespace opengl{
    namespace testing{
        // create a session for all test
        std::unique_ptr<FBOSession> InitSession(){
            auto session = std::unique_ptr<FBOSession>(new FBOSession);
            return session;
        }

        void InitOGLContext(){
            //TODO(breakpoint) how to init once for all test case
            ::opengl::glfw_init();
            ::opengl::glew_init();
        }

        void CheckSameTensor(const Tensor* cpu_tensor1, const Tensor* cpu_tensor2){
            auto cpu_shape = cpu_tensor1->shape();
            auto ogl_shape = cpu_tensor2->shape();
            CHECK_EQ(cpu_shape.size(), ogl_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                EXPECT_EQ(cpu_shape[i], ogl_shape[i]);
            }

            // ogl data
            const float* ogl_output_data = cpu_tensor2->host<float>();
            // original data
            const float* cpu_output_data = cpu_tensor1->host<float>();
            CHECK_EQ(cpu_tensor1->AllocatedElements(), cpu_tensor2->AllocatedElements());
            const int output_num_elements = cpu_tensor1->AllocatedElements();
            for(int i=0;i<output_num_elements;++i){
                EXPECT_NEAR(cpu_output_data[i], ogl_output_data[i], 1e-5)<<" When index: "<<i;
            }
            // check dtype and dformat
            CHECK_EQ(cpu_tensor1->dformat(), cpu_tensor2->dformat());
            CHECK_EQ(cpu_tensor1->dtype(), cpu_tensor2->dtype());
            CHECK_EQ(cpu_tensor1->mem_type(), cpu_tensor2->mem_type());
        }

        void CheckSameValueTensor(const Tensor* cpu_tensor1, const Tensor* cpu_tensor2){
            const float* output_data1 = cpu_tensor1->host<float>();
            const float* output_data2 = cpu_tensor2->host<float>();
            CHECK_EQ(cpu_tensor1->AllocatedElements(), cpu_tensor2->AllocatedElements());
            const int output_num_elements = cpu_tensor1->AllocatedElements();
            for(int i=0;i<output_num_elements;++i){
                EXPECT_FLOAT_EQ(output_data1[i], output_data2[i])<<" When index: "<<i;
            }
            // check dtype and dformat
            CHECK_EQ(cpu_tensor1->dformat(), cpu_tensor2->dformat());
            CHECK_EQ(cpu_tensor1->dtype(), cpu_tensor2->dtype());
            CHECK_EQ(cpu_tensor1->mem_type(), cpu_tensor2->mem_type());
        }


        void CleanupTensorList(::opengl::TensorList* outputs_tensor){
            for(auto& tensor: *outputs_tensor){
                delete tensor;
            }
        }

        void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
            CHECK(cpu_tensor->is_host());
            CHECK(!device_tensor->is_host());

            // check same bytes
            CHECK_EQ(cpu_tensor->AllocatedSize(), device_tensor->AllocatedSize());

            auto cpu_shape = cpu_tensor->shape();
            auto device_shape = device_tensor->shape();
            CHECK_EQ(cpu_shape.size(), device_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                CHECK_EQ(cpu_shape[i], device_shape[i]);
            }

            auto texture = device_tensor->device<Texture>();
            const int width = texture->width();
            const int height = texture->height();
            GLenum format = texture->format();
            GLenum type = texture->type();
            // TODO(breakpoint) why DMA is slower than non DMA
            CopyHostToTexture(cpu_tensor->host(), width, height, device_tensor->device<Texture>()->id(),
                    format, type);
        }

        void CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor){
            CHECK(cpu_tensor->is_host());
            CHECK(!device_tensor->is_host());

            // check same bytes
            CHECK_EQ(cpu_tensor->AllocatedSize(), device_tensor->AllocatedSize());

            auto cpu_shape = cpu_tensor->shape();
            auto device_shape = device_tensor->shape();
            CHECK_EQ(cpu_shape.size(), device_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                CHECK_EQ(cpu_shape[i], device_shape[i]);
            }

            auto texture = device_tensor->device<Texture>();
            const int width = texture->width();
            const int height = texture->height();
            GLenum format = texture->format();
            GLenum type = texture->type();
            // TODO(breakpoint) why DMA is slower than non DMA
            CopyTextureToHost(cpu_tensor->host(), width, height, device_tensor->device<Texture>()->id(),
                    format, type);
        }


        void CheckSameDeviceTensor(const Tensor* gpu_tensor1, const Tensor* gpu_tensor2){
            CHECK(!gpu_tensor1->is_host());
            CHECK(!gpu_tensor2->is_host());
            CHECK_EQ(gpu_tensor1->AllocatedSize(), gpu_tensor2->AllocatedSize());

            auto cpu_shape = gpu_tensor1->shape();
            auto ogl_shape = gpu_tensor2->shape();
            CHECK_EQ(cpu_shape.size(), ogl_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                EXPECT_EQ(cpu_shape[i], ogl_shape[i]);
            }

            // check dtype and dformat
            CHECK_EQ(gpu_tensor1->dformat(), gpu_tensor2->dformat());
            CHECK_EQ(gpu_tensor1->dtype(), gpu_tensor2->dtype());
            CHECK_EQ(gpu_tensor1->mem_type(), gpu_tensor2->mem_type());

            const int allocated_size = gpu_tensor1->AllocatedSize();

            auto cpu_tensor1_ptr = std::unique_ptr<Tensor>(new Tensor(gpu_tensor1));
            auto cpu_tensor2_ptr = std::unique_ptr<Tensor>(new Tensor(gpu_tensor2));

            Tensor* cpu_tensor1 = cpu_tensor1_ptr.get();
            Tensor* cpu_tensor2 = cpu_tensor2_ptr.get();

            // copy data
            CopyDeviceTensorToCPU(gpu_tensor1, cpu_tensor1);
            CopyDeviceTensorToCPU(gpu_tensor2, cpu_tensor2);

            CheckSameValueTensor(cpu_tensor1, cpu_tensor2);
        }
    }//namespace testing
}//namespace opengl
