#include "opengl/core/functor.h"
#include "opengl/core/tensor.h"
#include "opengl/core/context.h"
#include "opengl/nn/all_shaders.h"
#include "opengl/core/program.h"
#include <glog/logging.h>

namespace opengl{
    namespace internal{
        IntList AmendShape(const IntList& shape){
            CHECK_LE(shape.size(), 4);
            const int remain_dims = 4-shape.size();
            IntList amended_shape = shape;
            for(int i=0;i<remain_dims;++i){
                amended_shape.insert(amended_shape.begin(), 1);
            }
            return amended_shape;
        }

        void RunOpenGLProgram(const std::string& kernel_fname, Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            // common used for copy from host to device
            CHECK(!src_tensor->is_host());
            CHECK(!dst_tensor->is_host());

            // not own it
            auto program = ctx->CreateProgram(kernel_fname);
            // activate it before use
            program->Activate();

            program->SetRetVal({dst_tensor});

            // set input
            auto src_texture = src_tensor->device<Texture>();
            program->set_vec4i("output_shape", AmendShape(dst_tensor->shape()));
            // input
            {
                program->set_image2D("input_image", src_texture->id(),  0);
                OPENGL_CHECK_ERROR;
            }
            program->Run();
        }
    }//namespace internal
    namespace functor{
        void ConvertTensorNHWC4ToANY4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram(glsl_nhwc4_to_any4_glsl,
                    ctx, src_tensor, dst_tensor);
        }

        void ConvertTensorANYToANY4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram(glsl_any_to_any4_glsl,
                    ctx, src_tensor, dst_tensor);

        }

        void ConvertTensorANY4ToANY::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram(glsl_any4_to_any_glsl,
                    ctx, src_tensor, dst_tensor);
        }

        void ConvertTensorANYToNHWC4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram(glsl_any_to_nhwc4_glsl,
                    ctx, src_tensor, dst_tensor);
        };

        void ConvertTensorNCHWToHWN4C4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram(glsl_nchw_to_hwn4c4_glsl,
                    ctx, src_tensor, dst_tensor);
        }

        void ConvertTensorTest::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram(glsl_layout_test_glsl,
                    ctx, src_tensor, dst_tensor);
        };

        void ConvertTensorNHWC4ToANY::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram(glsl_nhwc4_to_any_glsl,
                    ctx, src_tensor, dst_tensor);
        };

        void ConvertTensorHWN4C4ToNCHW::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            internal::RunOpenGLProgram(glsl_hwn4c4_to_nchw_glsl,
                    ctx, src_tensor, dst_tensor);
        };

        ////////////////////////////////
        // common used functor in detector
        void SSDBBoxDecoder::operator()(Context* ctx, const Tensor* prediction_tensor,
                const Tensor* anchor_tensor, Tensor* decoded_tensor,
                const std::vector<float>& variances){

            // common used for copy from host to device
            // (num_batches, num_samples, 4)
            CHECK(!prediction_tensor->is_host());
            // (num_samples, 4)
            CHECK(!anchor_tensor->is_host());
            // (num_batches, num_sampels, 4)
            CHECK(!decoded_tensor->is_host());

            // check shape
            auto box_preds_shape = prediction_tensor->shape();
            auto anchor_shape = anchor_tensor->shape();
            auto decoded_shape = decoded_tensor->shape();
            CHECK_EQ(box_preds_shape.size(), 3);
            CHECK_EQ(decoded_shape.size(), 3);
            CHECK_EQ(anchor_shape.size(), 2);

            CHECK_EQ(decoded_shape[2], 4);
            CHECK_EQ(box_preds_shape[2], 4);
            CHECK_EQ(anchor_shape[0], box_preds_shape[1]);
            CHECK_EQ(anchor_shape[1], 4);

            auto program = ctx->CreateProgram(glsl_decoder_glsl);
            // activate it before use
            program->Activate();

            program->SetRetVal({decoded_tensor});

            // set input
            auto prediction_texture = prediction_tensor->device<Texture>();
            auto anchors_texture = anchor_tensor->device<Texture>();
            program->set_vec4("variances", variances);
            // prediction
            {
                program->set_image2D("prediction", prediction_texture->id(),  0);
                OPENGL_CHECK_ERROR;
            }

            // prediction
            {
                program->set_image2D("anchors", anchors_texture->id(),  1);
                OPENGL_CHECK_ERROR;
            }
            program->Run();
        }

        void SSDBBoxEncoder::operator()(Context* ctx, const Tensor* gtboxes_tensor,
                const Tensor* anchor_tensor, Tensor* encoded_tensor){
        }

        void NMS::operator()(Context* ctx, const Tensor* boxes, Tensor* final_boxes, float nms_threshold){

            // common used for copy from host to device
            CHECK(!boxes->is_host());
            CHECK(!final_boxes->is_host());
            CHECK_EQ(boxes->shape().size(), 3);

            auto program = ctx->CreateProgram(glsl_nms_glsl);
            // activate it before use
            program->Activate();

            program->SetRetVal({final_boxes});

            // set input
            auto boxes_texture = boxes->device<Texture>();
            program->set_vec3i("input_shape", boxes->shape());
            program->set_float("nms_threshold", nms_threshold);
            program->set_int("topk", 100);
            // prediction
            {
                program->set_image2D("boxes", boxes_texture->id(),  0);
                OPENGL_CHECK_ERROR;
            }

            program->Run();
        }
    }//namespace functor
}//namespace opengl
