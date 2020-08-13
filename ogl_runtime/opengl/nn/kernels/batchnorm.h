#ifndef OPENGL_NN_KERNELS_BATCHNORM_H_
#define OPENGL_NN_KERNELS_BATCHNORM_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

    class BatchNormKernel: public Kernel{
        public:
            BatchNormKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(const TensorList& inputs,
                    TensorShapeList& outputs);
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs);
            virtual void SetupAttr(const dlxnet::Attribute& attr);
            virtual void SelectKernel(const TensorList& inputs)override;
            virtual ~BatchNormKernel();
        private:
            float momentum_;
            float eps_;
    };
}//namespace opengl

#endif
