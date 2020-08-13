#ifndef OPENGL_UTILS_MEM_H_
#define OPENGL_UTILS_MEM_H_
#include <cstddef>

namespace opengl{
    namespace port{
        void* AlignedMalloc(size_t size, int minimum_alignment);
        void* Malloc(size_t size);
        void AlignedFree(void* aligned_memory);
        void Free(void* ptr);
    }//port
}//namespace opengl


#endif
