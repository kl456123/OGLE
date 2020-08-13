#ifndef OPENGL_CORE_OGL_ALLOCATOR_H_
#define OPENGL_CORE_OGL_ALLOCATOR_H_
#include "opengl/core/allocator.h"

namespace opengl{
    class OGLTextureAllocator: public Allocator{
        public:
            OGLTextureAllocator();
            virtual ~OGLTextureAllocator()override{}
            std::string Name() override { return "ogl"; }
            void* AllocateRaw(size_t alignment, size_t num_bytes) override;
            // void* AllocateRaw(const IntList& shape, DataFormat dformat,
                    // Tensor::DataType dtype)override;

            void DeallocateRaw(void* ptr) override;
        private:
            const int kMaxTextureSize_;
            std::string name_;
    };

    class StrideAllocator{
        public:
            static void* Allocate(Allocator* raw_allocator, size_t num_elements,
                    size_t stride, const AllocationAttributes& allocation_attr);

            static void Deallocate(Allocator* raw_allocator, void* ptr);
    };


}//namespace opengl


#endif
