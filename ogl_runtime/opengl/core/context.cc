#include <cstring>
#include <glog/logging.h>

#include "opengl/core/context.h"
#include "opengl/core/program.h"
#include "opengl/utils/macros.h"
#include "opengl/core/driver.h"
#include "opengl/core/tensor_format.h"
#include "opengl/core/functor.h"
#include "opengl/core/tensor_allocator.h"


namespace opengl{
    namespace internal{
        void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
            CHECK(cpu_tensor->is_host());
            CHECK(!device_tensor->is_host());

            auto cpu_shape = cpu_tensor->shape();
            auto device_shape = device_tensor->shape();
            CHECK_EQ(cpu_shape.size(), device_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                CHECK_EQ(cpu_shape[i], device_shape[i]);
            }

            // check same bytes
            CHECK_EQ(cpu_tensor->AllocatedSize(), device_tensor->AllocatedSize());

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

            auto cpu_shape = cpu_tensor->shape();
            auto device_shape = device_tensor->shape();
            CHECK_EQ(cpu_shape.size(), device_shape.size());
            for(int i=0;i<cpu_shape.size();++i){
                CHECK_EQ(cpu_shape[i], device_shape[i]);
            }

            // check same bytes
            CHECK_EQ(cpu_tensor->AllocatedSize(), device_tensor->AllocatedSize());

            auto texture = device_tensor->device<Texture>();
            const int width = texture->width();
            const int height = texture->height();
            GLenum format = texture->format();
            GLenum type = texture->type();
            // TODO(breakpoint) why DMA is slower than non DMA
            CopyTextureToHost(cpu_tensor->host(), width, height, device_tensor->device<Texture>()->id(),
                    format, type);
        }
    }

    namespace{
        const GLenum kDataType=GL_FLOAT;
        GLenum kInternalFormat = GL_RGBA32F;
        GLenum kFormat = GL_RGBA;
        // Don't need to change this.
        // We want to draw 2 giant triangles that cover the whole screen.
        struct Vertex {
            float x, y;
        };

        static constexpr size_t kNumVertices = 6;

        const char *vertex_shader_text = "#version 300 es\n"
            "in vec2 point; // input to vertex shader\n"
            "void main() {\n"
            "  gl_Position = vec4(point, 0.0, 1.0);\n"
            "}\n";

        const Vertex vertices[kNumVertices] = {
            {-1.f, -1.f},
            {1.0f, -1.f},
            {1.0f, 1.0f},
            {-1.f, -1.f},
            {-1.f, 1.0f},
            {1.0f, 1.0f},
        };
    }

    void Context::Compute(std::initializer_list<size_t> dim_sizes){
        auto ptr = dim_sizes.begin();
        glDispatchCompute(ptr[0], ptr[1], ptr[2]);
    }

    void Context::Reset(){
        frame_buffer_ = CreateFrameBuffer();
        CreateVertexShader();
    }

    Context::Context(Allocator* allocator)
        :allocator_(allocator),
        tensor_pool_allocator_(new TensorPoolAllocator){
            // max size allowed when using texture
            LOG(INFO)<<"max group invacations: "<<GetMaxTextureSize();
            // prepare framebuffer and vertex shader first
            // as for fragment shader, it is used as compute kernel
            Reset();
        }

    Tensor* Context::AllocateTensor(const IntList& shape,
            Tensor::MemoryType mem_type, DataFormat dformat){
        return tensor_pool_allocator_->AllocateTensor(
                shape, mem_type, dformat);
    }

    void Context::DeallocateTensor(Tensor* ptr){
        return tensor_pool_allocator_->DeallocateTensor(ptr);
    }


    void Context::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
        // optimize for special case
        if(cpu_tensor->last_stride()%4==0
                && device_tensor->dformat()==dlxnet::TensorProto::ANY4){
            internal::CopyCPUTensorToDevice(cpu_tensor, device_tensor);
            return;
        }

        if(device_tensor->dformat()==cpu_tensor->dformat()){
            internal::CopyCPUTensorToDevice(cpu_tensor, device_tensor);
            return;
        }

        Tensor* src_gpu_tensor =  tensor_pool_allocator_->AllocateTensor(
                cpu_tensor->shape(), Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY);

        internal::CopyCPUTensorToDevice(cpu_tensor, src_gpu_tensor);
        if(device_tensor->dformat()==dlxnet::TensorProto::ANY4){
            // any to any4 (device->device)
            functor::ConvertTensorANYToANY4()(this, src_gpu_tensor, device_tensor);
        }else if(device_tensor->dformat()==dlxnet::TensorProto::NHWC4
                && cpu_tensor->dformat()== dlxnet::TensorProto::NHWC){
            functor::ConvertTensorANYToNHWC4()(this, src_gpu_tensor, device_tensor);
        }else if(device_tensor->dformat()==dlxnet::TensorProto::HWN4C4){
            functor::ConvertTensorNCHWToHWN4C4()(this, src_gpu_tensor, device_tensor);
        }else{
            LOG(FATAL)<<"unsupported conversion from device_dformat: "
                <<device_tensor->dformat()<<" -> cpu_dformat: "
                <<cpu_tensor->dformat();
        }
        tensor_pool_allocator_->DeallocateTensor(src_gpu_tensor);
    }


    void Context::CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor){
        // optimize for special case
        if(cpu_tensor->last_stride()%4==0
                && device_tensor->dformat()==dlxnet::TensorProto::ANY4){
            internal::CopyDeviceTensorToCPU(device_tensor, cpu_tensor);
            return;
        }

        if(device_tensor->dformat()==cpu_tensor->dformat()){
            internal::CopyDeviceTensorToCPU(device_tensor, cpu_tensor);
            return;
        }
        Tensor* dst_gpu_tensor = tensor_pool_allocator_->AllocateTensor(
                device_tensor->shape(), Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY);
        if(device_tensor->dformat()==dlxnet::TensorProto::ANY4
                &&cpu_tensor->dformat()==dlxnet::TensorProto::ANY){
            functor::ConvertTensorANY4ToANY()(this, device_tensor, dst_gpu_tensor);
        }else if(device_tensor->dformat()==dlxnet::TensorProto::NHWC4){
            functor::ConvertTensorNHWC4ToANY()(this, device_tensor, dst_gpu_tensor);
        }else if(device_tensor->dformat()==dlxnet::TensorProto::HWN4C4){
            functor::ConvertTensorHWN4C4ToNCHW()(this, device_tensor, dst_gpu_tensor);
        }else{
            LOG(FATAL)<<"unsupported conversion from device_dformat: "
                <<device_tensor->dformat()<<" -> cpu_dformat: "
                <<cpu_tensor->dformat();
        }
        internal::CopyDeviceTensorToCPU(dst_gpu_tensor, cpu_tensor);
        tensor_pool_allocator_->DeallocateTensor(dst_gpu_tensor);
    }

    void Context::CopyImageToBuffer(Texture* texture, Buffer* buffer){
        // program
        Program program = Program();
        program.AttachFile(kImage2buffer_name_).Link();

        GLuint ray_program = program.program_id();
        // GLuint tex_output = texture->id();
        // GLuint SSBO = buffer->id();

        int tex_w = texture->shape()[0];
        int tex_h = texture->shape()[1];

        program.Activate();

        //set param in program and then dispatch the shaders
        {
            // set input and output
            program.set_vec2i("image_shape", tex_w, tex_h);
            OPENGL_CHECK_ERROR;
            program.set_input_sampler2D(texture->id(), texture->format());
            OPENGL_CHECK_ERROR;
            program.set_buffer(buffer->id(), buffer->target());
            OPENGL_CHECK_ERROR;

            glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
            OPENGL_CHECK_ERROR;
        }

        // sync
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    void Context::CopyBufferToImage(Texture* texture, Buffer* buffer){
        //(TODO cache program if possible)
        // program
        Program program = Program();
        program.AttachFile(kBuffer2image_name_).Link();


        GLuint ray_program = program.program_id();

        int tex_w = texture->shape()[0];
        int tex_h = texture->shape()[1];

        program.Activate();

        //set param in program and then dispatch the shaders
        {
            // set input and output
            program.set_vec2i("image_shape", tex_w, tex_h);
            OPENGL_CHECK_ERROR;
            program.set_output_sampler2D(texture->id(), texture->format());
            OPENGL_CHECK_ERROR;
            program.set_buffer(buffer->id(), buffer->target());
            OPENGL_CHECK_ERROR;

            glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
        }

        // sync
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    void Context::CopyCPUBufferToDevice(Buffer* buffer, void* buffer_cpu){
        // upload
        auto ptr = buffer->Map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
        ::memcpy(ptr, buffer_cpu, buffer->size());
        buffer->UnMap();
    }

    void Context::CopyDeviceBufferToCPU(Buffer* buffer, void* buffer_cpu){
        // download
        auto ptr = buffer->Map(GL_MAP_READ_BIT);
        ::memcpy(buffer_cpu, ptr, buffer->size());
        buffer->UnMap();
    }

    void Context::CreateVertexShader(){
        // We always render the same vertices and triangles.
        GLuint vertex_buffer;
        OPENGL_CALL(glGenBuffers(1, &vertex_buffer));
        OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
        OPENGL_CALL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                    GL_STATIC_DRAW));

        GLuint vertex_array;
        OPENGL_CALL(glGenVertexArrays(1, &vertex_array));
        OPENGL_CALL(glBindVertexArray(vertex_array));
        OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));

        // We always use the same vertex shader.
        vertex_shader_ = CreateShader(GL_VERTEX_SHADER, vertex_shader_text);
    }

    Context::~Context(){
        OPENGL_CALL(glDeleteFramebuffers(1, &frame_buffer_));
        for(auto iter:program_set_){
            delete iter.second;
        }
    }

    Program* Context::CreateProgram(const std::string& kernel_fname,
            const std::string& build_options){
        if(kernel_fname.empty()){
            // no kernel program needed for this op, like const op
            return nullptr;
        }

        string program_key = kernel_fname+build_options;
        if(program_set_.find(program_key)!=program_set_.end()){
            auto program = program_set_[program_key];
            program->Activate();
            return program;
        }
        // set program
        // program_ .reset(new Program);
        auto program = new Program;
        (*program).AttachSource(kernel_fname, GL_FRAGMENT_SHADER, build_options)
            .AttachShader(vertex_shader_);
        program->Link();
        program->Activate();
        // set vertex shader first
        // then you can set fragment shader to do actually computation
        program->SetVertexShader();
        program_set_[program_key] = program;
        return program;
    }


    Context* GetContext(){
        static Context* context = new Context;
        return context;
    }
}//namespace opengl
