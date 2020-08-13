#ifndef OPENGL_NN_KERNELS_SHAPE_H_
#define OPENGL_NN_KERNELS_SHAPE_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

        class ShapeKernel: public Kernel{
            public:
                ShapeKernel(Context* context);
                virtual void Compute(TensorList& inputs, TensorList& outputs);
                virtual void InferOutputShape(TensorShapeList& inputs,
                        TensorShapeList& outputs);
                virtual void SetupAttr(const dlxnet::Attribute& attr);
                virtual ~ShapeKernel();
                virtual bool ForceReady()const override{return true;}
            private:
                Tensor* tensor_;

        };
}//namespace opengl

#endif
