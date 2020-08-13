#ifndef OPENGL_NN_KERNELS_UNSQUEEZE_H_
#define OPENGL_NN_KERNELS_UNSQUEEZE_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

        class UnsqueezeKernel: public Kernel{
            public:
                UnsqueezeKernel(Context* context);
                virtual void Compute(TensorList& inputs, TensorList& outputs);
                virtual void InferOutputShape(TensorShapeList& inputs,
                        TensorShapeList& outputs);
                virtual void SetupAttr(const dlxnet::Attribute& attr);
                virtual ~UnsqueezeKernel();
            private:
                std::vector<int> axes_;

        };
}//namespace opengl

#endif
