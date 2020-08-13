#include "opengl/utils/mem.h"
#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#include <sys/sysinfo.h>
#else
#include <sys/syscall.h>
#endif

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

namespace opengl{
    namespace port{
        void* Malloc(size_t size) { return malloc(size); }

        void* AlignedMalloc(size_t size, int minimum_alignment) {
            void* ptr = nullptr;
            // posix_memalign requires that the requested alignment be at least
            // sizeof(void*). In this case, fall back on malloc which should return
            // memory aligned to at least the size of a pointer.
            const int required_alignment = sizeof(void*);
            if (minimum_alignment < required_alignment) return Malloc(size);
            int err = posix_memalign(&ptr, minimum_alignment, size);
            if (err != 0) {
                return nullptr;
            } else {
                return ptr;
            }
        }

        void AlignedFree(void* aligned_memory) { Free(aligned_memory); }
        void Free(void* ptr) { free(ptr); }

    }
}//namespace opengl
