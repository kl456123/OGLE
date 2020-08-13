#ifndef KERNELS_BINARY_H_
#define KERNELS_BINARY_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{

    class Context;
    class BinaryKernel:public Kernel{
        public:
            BinaryKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs);
            virtual void InferOutputShape(const TensorList& inputs,
                    TensorShapeList& outputs);
            virtual void SetupAttr(const dlxnet::Attribute& attr);
            virtual void SelectKernel(const TensorList& inputs)override;
            virtual ~BinaryKernel();
    };
}//namespace opengl


#endif
