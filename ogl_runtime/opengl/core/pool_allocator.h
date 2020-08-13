#ifndef OPENGL_CORE_POOL_ALLOCATOR_H_
#define OPENGL_CORE_POOL_ALLOCATOR_H_
#include "opengl/core/allocator.h"
#include "opengl/core/types.h"

namespace opengl{
    class PoolAllocator:public Allocator{
        public:
            PoolAllocator(Allocator* allocator);

            ~PoolAllocator() override;

            string Name() override { return name_; }

            void* AllocateRaw(size_t alignment, size_t num_bytes) override;

            void DeallocateRaw(void* ptr) override;

        private:
            Allocator* allocator_;
            string name_;
    };
} // namespace opengl


#endif
