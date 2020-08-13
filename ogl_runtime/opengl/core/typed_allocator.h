#ifndef OPENGL_CORE_TYPED_ALLOCATOR_H_
#define OPENGL_CORE_TYPED_ALLOCATOR_H_
/************************
 * TypedAllocator is a static member functions collections, so use it directly
 * like `TypedAllocator::Allocate<T>(num_elements);`.
 * note that use num_elements instead of num_bytes
 * This file just contains some template member functions, so dont need source file
 * with it.
 */
#include <limits>
#include "opengl/core/allocator.h"

namespace opengl{
    class TypedAllocator{
        public:
            TypedAllocator();
            template<typename T>
                static T* Allocate(Allocator* raw_allocator, size_t num_elements
                        const AllocationAttributes& allocation_attr){
                    if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
                        return nullptr;
                    }
                    void* p =
                        raw_allocator->AllocateRaw(sizeof(T) * num_elements, allocation_attr);
                    T* typed_p = reinterpret_cast<T*>(p);
                    if (typed_p) RunCtor<T>(raw_allocator, typed_p, num_elements);
                    return typed_p;
                }

            template<typename T>
                void Deallocate(){
                }

        private:
            template <typename T>
                static void RunDtor(Allocator* raw_allocator, T* p, size_t n) {}
    };

    template <>
        /* static */
        inline void TypedAllocator::RunCtor(Allocator* raw_allocator, ShaderBuffer* p,
                size_t n) {
            if (!raw_allocator->AllocatesOpaqueHandle()) {
                for (size_t i = 0; i < n; ++p, ++i) new (p) ShaderBuffer(n);
            }
        }

    template <>
        /* static */
        inline void TypedAllocator::RunDtor(Allocator* raw_allocator, ShaderBuffer* p,
                size_t n) {
            if (!raw_allocator->AllocatesOpaqueHandle()) {
                for (size_t i = 0; i < n; ++p, ++i) p->~ShaderBuffer();
            }
        }

}

#endif
