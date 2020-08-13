#ifndef OPENGL_NN_KERNELS_RELU_H_
#define OPENGL_NN_KERNELS_RELU_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

    class ReluKernel: public Kernel{
        public:
            ReluKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs);
            virtual void SetupAttr(const dlxnet::Attribute& attr);
            virtual ~ReluKernel();
        private:
            float slope_;
    };
}//namespace opengl

#endif
