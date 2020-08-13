#ifndef OPENGL_CORE_CONTEXT_H_
#define OPENGL_CORE_CONTEXT_H_
#include<vector>
#include <memory>

#include "opengl/core/opengl.h"
#include "opengl/core/tensor.h"
#include "opengl/core/buffer.h"
#include "opengl/nn/all_shaders.h"

namespace opengl{
    class Allocator;
    class Program;
    class TensorPoolAllocator;
    class Context{
        public:
            Context(Allocator* allocator);
            Context():Context(nullptr){}
            virtual ~Context();

            // call it before use session
            void Reset();

            void Compute(std::initializer_list<size_t> dim_sizes);

            void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor);
            void CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor);

            // for now convert it in host memory instead of device memory
            // TODO(breakpoint) add 1d representation in fragment shader to support nhwc format
            // buffer in nhwc layout, texture in (nh, w, c4, 4) layout
            // conversion between nhwc and nhwc4
            void ConvertTensorNHWCToNHWC4(const Tensor* nhwc_tensor, void** nhwc4_data);
            void ConvertTensorNHWCToNHWC4(const Tensor* , Tensor* );
            void ConvertTensorNHWC4ToNHWC(void* out, Tensor* tensor);

            // conversion between nchw and hwn4c4
            void ConvertTensorNCHWToHWN4C4(const Tensor* tensor, void** out);
            void ConvertTensorHWN4C4ToNCHW(void* out, Tensor* tensor);

            // torch dformat to dlxnet dformat
            void ConvertTensorNCHWToNHWC4(const Tensor* tensor, void** out);

            void ConvertTensorHWN4C4ToNHWC(void* src, Tensor* tensor);

            void ConvertTensorToStride4(const Tensor* src_tensor, void** out);
            void ConvertTensorFromStride4(void* src, Tensor* tensor);

            // primitive apis operate buffer and image
            void CopyImageToBuffer(Texture* texture, Buffer* buffer);
            void CopyBufferToImage(Texture* texture, Buffer* buffer);

            void CopyCPUBufferToDevice(Buffer* buffer, void* buffer_cpu);
            void CopyDeviceBufferToCPU(Buffer* buffer, void* buffer_cpu);
            void Finish(){glFlush();}

            Program* CreateProgram(const std::string& kernel_fname,
                    const std::string& build_options="");

            Tensor* AllocateTensor(const IntList& shape,
                    Tensor::MemoryType mem_type, DataFormat dformat);
            void DeallocateTensor(Tensor* ptr);
        private:
            // used to allocator new buffer or texture duration runtime
            Allocator* allocator_;

            void CreateVertexShader();

            // used to copy
            std::unique_ptr<ShaderBuffer> temp_buffer_;// owned

            // common compute shader
            const char* kImage2buffer_name_ = "../opengl/examples/gpgpu/image2buffer.glsl";
            const char* kBuffer2image_name_ = "../opengl/examples/gpgpu/buffer2image.glsl";

            GLuint vertex_shader_;
            // output target in each kernel
            GLuint frame_buffer_;

            std::unique_ptr<TensorPoolAllocator> tensor_pool_allocator_;
            // cache all opengl programs
            // std::unique_ptr<ProgramSet> program_set_;
            std::map<string, Program*> program_set_;
    };

    Context* GetContext();

}//namespace opengl


#endif
