#include "opengl/core/tensor_allocator.h"

namespace opengl{
    namespace{
        bool SameShape(const IntList& shape1, const IntList& shape2){
            if(shape1.size()!=shape2.size()){
                return false;
            }
            const int num = shape1.size();
            for(int i=0;i<num;++i){
                if(shape1[i]!=shape2[i]){
                    return false;
                }
            }
            return true;
        }
    }
    TensorPoolAllocator::TensorPoolAllocator(){
    }

    Tensor* TensorPoolAllocator::AllocateTensor(const IntList& shape,
            Tensor::MemoryType mem_type, DataFormat dformat){
        for (auto iter = free_tensors_.begin(); iter != free_tensors_.end(); ++iter) {
            Tensor* tensor = *iter;
            if(SameShape(tensor->shape(), shape)
                    &&mem_type==tensor->mem_type()
                    &&dformat==tensor->dformat()){
                free_tensors_.erase(iter);
                return tensor;
            }
        }
        Tensor* tensor = new Tensor(Tensor::DT_FLOAT,
                shape, mem_type, dformat);
        total_tensors_.push_back(tensor);
        return tensor;
    }

    void TensorPoolAllocator::DeallocateTensor(Tensor* ptr){
        free_tensors_.push_back(ptr);
    }

    TensorPoolAllocator::~TensorPoolAllocator(){
        for(auto tensor:total_tensors_){
            delete tensor;
        }
    }
}//namespace opengl

