#ifndef OPENGL_CORE_FUNCTOR_H_
#define OPENGL_CORE_FUNCTOR_H_
#include <vector>

namespace opengl{
    class Tensor;
    class Context;
    namespace functor{
        /////////////////////////////////
        // all dformat conversion
        /////////////////////////////////

        // some commonly used kernel functions, but we dont consider them as kernel to simpliy
        // logic, just make it as a functor.
        // Note that `Functor` is a struct overrided operator() to be called more easily.
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

        // used for test
        struct ConvertTensorTest{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        struct ConvertTensorNHWC4ToANY{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        struct ConvertTensorHWN4C4ToNCHW{
            void operator()(Context* ctx, const Tensor* src_tensor, Tensor* dst_tensor);
        };

        ////////////////////////////////
        // some common used functor like
        // box encoder and decoder, nms
        // and so on
        struct SSDBBoxDecoder{
            void operator()(Context* ctx, const Tensor* prediction_tensor,
                    const Tensor* anchor_tensor, Tensor* decoded_tensor,
                    const std::vector<float>& variances);
        };

        struct SSDBBoxEncoder{
            void operator()(Context* ctx, const Tensor* gtboxes_tensor,
                    const Tensor* anchor_tensor, Tensor* encoded_tensor);
        };

        struct NMS{
            void operator()(Context* ctx, const Tensor* boxes, Tensor* final_boxes,
                    float nms_threshold);
        };
    }//namespace functor
}// namespace opengl


#endif
