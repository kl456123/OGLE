#ifndef OPENGL_NN_KERNELS_CLIP_H_
#define OPENGL_NN_KERNELS_CLIP_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

    class ClipKernel: public Kernel{
        public:
            ClipKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(const TensorList& inputs,
                    TensorShapeList& output_shapes)override;
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs)override{};
            virtual void SetupAttr(const dlxnet::Attribute& attr)override;
            virtual void SelectKernel(const TensorList& inputs)override;
            virtual ~ClipKernel();
        private:
            float min_;
            float max_;

    };
}//namespace opengl

#endif
