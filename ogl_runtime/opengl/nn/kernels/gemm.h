#ifndef OPENGL_NN_KERNELS_GEMM_H_
#define OPENGL_NN_KERNELS_GEMM_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{

    class Context;

    class GemmKernel: public Kernel{
        public:
            GemmKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs)override;
            virtual void SetupAttr(const dlxnet::Attribute& attr);
            virtual ~GemmKernel();
        private:
            float alpha_;
            float beta_;
            int transB_;
    };
}//namespace opengl


#endif
