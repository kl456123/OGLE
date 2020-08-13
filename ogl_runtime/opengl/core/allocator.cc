#include "opengl/core/allocator.h"


namespace opengl{
    Allocator::~Allocator(){}

    SubAllocator::SubAllocator(const std::vector<Visitor>& alloc_visitors,
            const std::vector<Visitor>& free_visitors)
        : alloc_visitors_(alloc_visitors), free_visitors_(free_visitors) {}

    void SubAllocator::VisitAlloc(void* ptr, int index, size_t num_bytes) {
        for (const auto& v : alloc_visitors_) {
            v(ptr, index, num_bytes);
        }
    }

    void SubAllocator::VisitFree(void* ptr, int index, size_t num_bytes) {
        // Although we don't guarantee any order of visitor application, strive
        // to apply free visitors in reverse order of alloc visitors.
        for (int i = free_visitors_.size() - 1; i >= 0; --i) {
            free_visitors_[i](ptr, index, num_bytes);
        }
    }
}//namespace opengl
