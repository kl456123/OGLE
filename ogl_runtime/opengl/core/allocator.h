#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_
/***************************
 * class Allocator is just to allocate memory in device or host, dont care of shape or types,
 * shape is cared when create tensor, types is cared in TypedAllocator. AllocationAttributes and
 * AllocatorStatistic can be added here for debugging and collecting information to log.
 * Note that use bytes insteads of number as the argument names
 */
#include <vector>
#include <string>
#include "opengl/core/types.h"
#include <functional>
#include "opengl/core/opengl.h"

namespace opengl{
    struct AllocationAttributes{
    };

    struct AllocatorStatistic{
    };

    class Allocator{
        public:
            // Return a string identifying this allocator
            virtual std::string Name() = 0;

            // Return an uninitialized block of memory that is "num_bytes" bytes
            // in size.  The returned pointer is guaranteed to be aligned to a
            // multiple of "alignment" bytes.
            // REQUIRES: "alignment" is a power of 2.
            virtual void* AllocateRaw(size_t alignment, size_t num_bytes) = 0;

            // Return an uninitialized block of memory that is "num_bytes" bytes
            // in size with specified allocation attributes.  The returned pointer is
            // guaranteed to be aligned to a multiple of "alignment" bytes.
            // REQUIRES: "alignment" is a power of 2.
            virtual void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) {
                // The default behavior is to use the implementation without any allocation
                // attributes.
                return AllocateRaw(alignment, num_bytes);
            }

            // Deallocate a block of memory pointer to by "ptr"
            // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
            virtual void DeallocateRaw(void* ptr) = 0;

            virtual ~Allocator();

            // check if it is opaque or not(note that device memory is refers to opaque memory)
            virtual bool AllocatesOpaqueHandle()const{return false;}

            virtual int AllocationId()const{return 0;}
    };

    // some common allocators
    Allocator* ogl_texture_allocator();
    Allocator* cpu_allocator();
    void* allocate_texture(const IntList& shape, DataFormat dformat);
    void* allocate_host_mem(const IntList& shape, DataFormat dformat);
    void deallocate_texture(void* ptr);
    void deallocate_host_mem(void* ptr);

    // high level api of allocator, used to construct PoolAllocator and BFCAllocator
    class SubAllocator{
        public:
            // Visitor gets called with a pointer to a memory area and its
            // size in bytes.  The index value will be numa_node for a CPU
            // allocator and GPU id for a GPU allocator.
            typedef std::function<void(void*, int index, size_t)> Visitor;

            SubAllocator(const std::vector<Visitor>& alloc_visitors,
                    const std::vector<Visitor>& free_visitors);

            virtual ~SubAllocator() {}
            virtual void* Alloc(size_t alignment, size_t num_bytes) = 0;
            virtual void Free(void* ptr, size_t num_bytes) = 0;

        protected:
            // Implementation of Alloc() method must call this on newly allocated
            // value.
            void VisitAlloc(void* ptr, int index, size_t num_bytes);

            // Implementation of Free() method must call this on value to be
            // freed immediately before deallocation.
            void VisitFree(void* ptr, int index, size_t num_bytes);

            const std::vector<Visitor> alloc_visitors_;
            const std::vector<Visitor> free_visitors_;

    };

}//namespace opengl
#endif
