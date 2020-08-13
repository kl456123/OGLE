#include "opengl/core/ogl_allocator.h"
#include "opengl/core/driver.h"
#include "opengl/core/texture.h"
#include "opengl/utils/macros.h"
#include  "opengl/core/tensor_format.h"

namespace opengl{
    namespace{
        // note that here alignment is refers to num_elements alignment
        const int kOGLAlignment = 4;
    }
    OGLTextureAllocator::OGLTextureAllocator()
        :kMaxTextureSize_(GetMaxTextureSize()){}

    // void* OGLTextureAllocator::AllocateRaw(const IntList& shape, DataFormat dformat,
            // Tensor::DataType dtype){
        // // h, w
        // const auto texture_shape = CalcAllocatedSize2D(shape, dformat);
        // return new Texture({texture_shape[1], texture_shape[0]}, GL_RGBA32F, GL_TEXTURE_2D, nullptr);
    // }

    void* OGLTextureAllocator::AllocateRaw(size_t alignment, size_t num_bytes){
        CHECK_EQ(alignment, kOGLAlignment);
        // only float type supported now
        const size_t requested_num_elements = num_bytes / sizeof(float);

        const size_t alloc_num_elements = UP_ROUND(requested_num_elements, alignment);
        const int image_height = UP_DIV(alloc_num_elements/alignment, kMaxTextureSize_);
        const int image_width = (image_height==1)? alloc_num_elements/alignment: kMaxTextureSize_;

        void* ptr = new Texture({image_width, image_height}, GL_RGBA32F, GL_TEXTURE_2D, nullptr);
        return ptr;
    }

    void OGLTextureAllocator::DeallocateRaw(void* ptr){
        Texture* texture_ptr = reinterpret_cast<Texture*>(ptr);
        delete texture_ptr;
    }

    // stride allocator implementation

    void* StrideAllocator::Allocate(Allocator* raw_allocator, size_t num_bytes,
            size_t stride, const AllocationAttributes& allocation_attr){
        // check stride is valid first
        int num_elements = num_bytes/sizeof(float);
        // CHECK_EQ(num_elements%stride, 0);
        if(num_elements%stride!=0){
            num_elements = UP_ROUND(num_elements, stride);
            num_bytes = num_elements*sizeof(float);
        }

        const size_t aligned_stride = UP_ROUND(stride, kOGLAlignment);
        const size_t requested_num_bytes = aligned_stride * num_bytes /stride;
        void* ptr = raw_allocator->AllocateRaw(kOGLAlignment, requested_num_bytes);
        return ptr;
    }

    void StrideAllocator::Deallocate(Allocator* raw_allocator, void* ptr){
        if(ptr){
            raw_allocator->DeallocateRaw(ptr);
        }
    }

    Allocator* ogl_texture_allocator(){
        static Allocator* a = new OGLTextureAllocator();
        return a;
    }
}//namespace opengl
