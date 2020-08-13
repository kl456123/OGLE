#ifndef OPENGL_NN_KERNELS_CONV2D_H_
#define OPENGL_NN_KERNELS_CONV2D_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{

    class Context;

    class Conv2DKernel: public Kernel{
        public:
            Conv2DKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs);
            virtual void SetupAttr(const dlxnet::Attribute& attr);
            virtual void SelectKernel(const TensorList& inputs)override;
            virtual ~Conv2DKernel();
        private:
            int padding_;
            int stride_;
            int kernel_size_;
            int group_size_;
            int dilation_;
            std::string activation_type_;
            float min_;
            float max_;
    };
}//namespace opengl


#endif
