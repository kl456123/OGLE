#ifndef OPENGL_CORE_TENSOR_ALLOCATOR_H_
#define OPENGL_CORE_TENSOR_ALLOCATOR_H_
#include <list>

#include "opengl/core/tensor.h"
#include "opengl/core/types.h"

namespace opengl{
    class Tensor;
    class TensorPoolAllocator{
        public:
            TensorPoolAllocator();
            virtual ~TensorPoolAllocator();
            Tensor* AllocateTensor(const IntList& shape,
                    Tensor::MemoryType mem_type, DataFormat dformat);
            void DeallocateTensor(Tensor* ptr);

            uint64 GetFreeSize()const{
                return free_tensors_.size();
            }
            uint64 GetTotalSize()const{
                return total_tensors_.size();
            }
        private:
            std::list<Tensor*> free_tensors_;
            std::list<Tensor*> total_tensors_;

    };
}//namespace opengl



#endif
