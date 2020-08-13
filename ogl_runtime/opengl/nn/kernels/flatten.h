#ifndef OPENGL_NN_KERNELS_FLATTEN_H_
#define OPENGL_NN_KERNELS_FLATTEN_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

        class FlattenKernel: public Kernel{
            public:
                FlattenKernel(Context* context);
                virtual void Compute(TensorList& inputs, TensorList& outputs);
                virtual void InferOutputShape(TensorShapeList& inputs,
                        TensorShapeList& outputs)override;
                virtual void SetupAttr(const dlxnet::Attribute& attr)override;
                virtual void SelectKernel(const TensorList& inputs)override;
                virtual ~FlattenKernel();
            private:
                int axis_;
                Tensor* any4_tensor_=nullptr;
        };
}//namespace opengl

#endif
