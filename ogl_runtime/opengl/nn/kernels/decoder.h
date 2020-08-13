#ifndef OPENGL_NN_KERNELS_DECODER_H_
#define OPENGL_NN_KERNELS_DECODER_H_
#include <vector>
#include "opengl/core/types.h"
#include "opengl/core/kernel.h"


namespace opengl{
    class Context;

    class DecoderKernel: public Kernel{
        public:
            DecoderKernel(Context* context);
            virtual void Compute(TensorList& inputs, TensorList& outputs);
            virtual void InferOutputShape(const TensorList& inputs,
                    TensorShapeList& outputs)override;
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs)override{}
            virtual void SetupAttr(const dlxnet::Attribute& attr);
            virtual void SelectKernel(const TensorList& inputs)override;
            virtual ~DecoderKernel();
        private:
            float score_threshold_;
            float nms_threshold_;
            std::vector<float> variances_;

    };
}//namespace opengl

#endif
