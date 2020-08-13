#ifndef OPENGL_NN_KERNELS_TRANSPOSE_H_
#define OPENGL_NN_KERNELS_TRANSPOSE_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

        class TransposeKernel: public Kernel{
            public:
                TransposeKernel(Context* context);
                virtual void Compute(TensorList& inputs, TensorList& outputs)override;
                virtual void InferOutputShape(TensorShapeList& inputs,
                        TensorShapeList& outputs)override;
                virtual void SetupAttr(const dlxnet::Attribute& attr)override;
                virtual void SelectKernel(const TensorList& inputs)override;
                virtual ~TransposeKernel();
            private:
                std::vector<int> perm_;

                // cached any4 tensor
                Tensor* cached_any4_tensor_ = nullptr;

        };
}//namespace opengl

#endif
