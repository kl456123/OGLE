#ifndef OPENGL_NN_KERNELS_POOL_H_
#define OPENGL_NN_KERNELS_POOL_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;
    enum PoolType{
        MaxPool,
        AveragePool,
        GlobalAveragePool
    };

    template<PoolType pool_type>
        class PoolKernel: public Kernel{
            public:
                PoolKernel(Context* context);
                virtual void Compute(TensorList& inputs, TensorList& outputs);
                virtual void InferOutputShape(TensorShapeList& inputs,
                        TensorShapeList& outputs);
                virtual void InferOutputShape(const TensorList& inputs,
                        TensorShapeList& outputs);
                virtual void SetupAttr(const dlxnet::Attribute& attr);
                virtual void SelectKernel(const TensorList& inputs)override;
                virtual ~PoolKernel();
            private:
                int padding_;
                int stride_;
                int kernel_size_;
                PoolType pool_type_=pool_type;
        };
}//namespace opengl

#endif
