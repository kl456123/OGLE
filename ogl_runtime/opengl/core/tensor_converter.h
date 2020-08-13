#ifndef OPENGL_CORE_TENSOR_CONVERTER_H_
#define OPENGL_CORE_TENSOR_CONVERTER_H_
#include "opengl/core/tensor.h"

namespace opengl{

    class TensorFormatConverter{
        public:
            static void ConvertTensor(const Tensor* src_tensor, Tensor* dst_tensor);
        private:
            static void ConvertFromNHWCToNHWC4(const Tensor* src_tensor,
                    Tensor* dst_tensor);

            static void ConvertFromNCHWToNHWC4(const Tensor* src_tensor,
                    Tensor* dst_tensor);

            static void ConvertFromNCHWToNHWC4(const Tensor* src_tensor,
                    Tensor* dst_tensor);
    };
}//namespace



#endif
