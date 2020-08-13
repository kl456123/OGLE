#ifndef OPENGL_NN_KERNELS_CONCAT_H_
#define OPENGL_NN_KERNELS_CONCAT_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

    class ConcatKernel: public Kernel{
        public:
            ConcatKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(const TensorList& inputs,
                    TensorShapeList& outputs)override;
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs)override{}
            virtual void SetupAttr(const dlxnet::Attribute& attr);
            virtual void SelectKernel(const TensorList& inputs)override;
            virtual ~ConcatKernel();
        private:
            int axis_;

    };
}//namespace opengl

#endif
