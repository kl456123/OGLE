#ifndef OPENGL_CORE_HOST_FUNCTOR_H_
#define OPENGL_CORE_HOST_FUNCTOR_H_
#include <vector>

namespace opengl{
    class Tensor;
    class Context;
    namespace host_functor{
        struct ConvertTensorNHWC4ToANY4{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        struct ConvertTensorANYToANY4{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        struct ConvertTensorANY4ToANY{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        struct ConvertTensorANYToNHWC4{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        // only usd for filter
        struct ConvertTensorNCHWToHWN4C4{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        struct ConvertTensorHWN4C4ToNCHW{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        struct SSDBBoxDecoder{
            void operator()(Context* ctx, const Tensor* prediction_tensor,
                    const Tensor* anchor_tensor, Tensor* decoded_tensor,
                    const std::vector<float>& variances);
        };
    } // namespace host_functor
} // namespace opengl

#endif
