#include <cmath>

#include "opengl/core/cpu_functor.h"
#include "opengl/utils/logging.h"
#include "opengl/core/tensor.h"


namespace opengl{
    namespace host_functor{
        void ConvertTensorANYToANY4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            CHECK(src_tensor->is_host());
            CHECK(dst_tensor->is_host());
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();

            const int src_num_elements = src_tensor->num_elements();
            const int src_last_dim = src_tensor->last_stride();
            const int dst_last_dim = UP_ROUND(src_last_dim, 4);

            memset(dst_data, 0, dst_tensor->AllocatedSize());
            // only use data if it is in src
            for(int i=0; i<src_num_elements; ++i){
                const int dst_index = i/src_last_dim*dst_last_dim+i%src_last_dim;
                dst_data[dst_index] = src_data[i];
            }
        }

        void ConvertTensorANY4ToANY::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            CHECK(src_tensor->is_host());
            CHECK(dst_tensor->is_host());
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();

            const int dst_num_elements = dst_tensor->num_elements();
            const int dst_last_dim = dst_tensor->last_stride();
            const int src_last_dim = UP_ROUND(dst_last_dim, 4);
            const int src_num_elements = src_tensor->AllocatedElements();

            memset(dst_data, 0, dst_tensor->AllocatedSize());
            // only use data if it is in src
            for(int i=0; i<dst_num_elements; ++i){
                const int src_index = i/dst_last_dim*src_last_dim+i%dst_last_dim;
                dst_data[i] = src_data[src_index];
            }
        }

        void ConvertTensorANYToNHWC4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            // the same as any to any4
            ConvertTensorANYToANY4()(ctx, src_tensor, dst_tensor);
        }

        void ConvertTensorHWN4C4ToNCHW::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            CHECK(src_tensor->is_host());
            CHECK(dst_tensor->is_host());
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();

            const int num_elements = dst_tensor->num_elements();
            const int c = dst_tensor->shape()[1];
            const int h = dst_tensor->shape()[2];
            const int w = dst_tensor->shape()[3];
            const int n = dst_tensor->shape()[0];
            const int up_channel = UP_DIV(c, 4)*4;
            const int n4 = UP_DIV(n, 4);
            const int c4 = UP_DIV(c, 4);

            for(int i=0; i<num_elements; ++i){
                int cur = i;
                const int w_i = cur%w;
                cur/=w;
                const int h_i = cur%h;
                cur/=h;
                const int c_i = cur%c;
                cur/=c;
                const int n_i = cur;
                const int offset = ((((h_i*w+w_i)*n4+n_i/4)*c4+c_i/4)*4+c_i%4)*4+n_i%4;

                dst_data[i]= src_data[offset];
            }

        }

        void ConvertTensorNCHWToHWN4C4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            CHECK(src_tensor->is_host());
            CHECK(dst_tensor->is_host());
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();
            const int src_num_elements = src_tensor->num_elements();
            const int n_out = src_tensor->shape()[0];
            const int n_in = src_tensor->shape()[1];
            const int h = src_tensor->shape()[2];
            const int w = src_tensor->shape()[3];
            const int in_4 = UP_DIV(n_in, 4);
            const int out_4 = UP_DIV(n_out, 4);
            memset(dst_data, 0, dst_tensor->AllocatedSize());
            for(int i=0; i<src_num_elements; ++i){
                int cur = i;
                const int w_i = cur%w;
                cur /= w;
                const int h_i = cur%h;
                cur/=h;
                const int in_i = cur%n_in;
                cur/=n_in;
                const int out_i = cur;

                const int hw_i = h_i*w+w_i;
                const int io4_i = (out_i/4 *in_4+in_i/4)*4+in_i%4;
                const int dst_index = (hw_i*in_4*out_4*4+io4_i)*4+out_i%4;
                dst_data[dst_index] = src_data[i];
            }
        }

        void ConvertTensorNHWC4ToANY4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            CHECK(src_tensor->is_host());
            CHECK(dst_tensor->is_host());
            CHECK_GE(dst_tensor->AllocatedSize(), src_tensor->AllocatedSize());
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();
            memset(dst_data, 0, dst_tensor->AllocatedSize());
            memcpy(dst_data, src_data, src_tensor->AllocatedSize());
        }

        void SSDBBoxDecoder::operator()(Context* ctx, const Tensor* box_preds,
                const Tensor* anchor_tensor, Tensor* decoded_tensor,
                const std::vector<float>& variances){
            const float* box_preds_data = box_preds->host<float>();
            const float* anchors_dataPtr = anchor_tensor->host<float>();
            const int num_samples = box_preds->shape()[1];
            float* decoded_boxes_data = decoded_tensor->host<float>();
            CHECK_EQ(box_preds->shape()[2], 4);
            const int num_cols = 4;

            for(int i=0;i<num_samples;++i){
                float ycenter =     box_preds_data[i*num_cols + 1] * variances[1]  * anchors_dataPtr[i*4 + 3] + anchors_dataPtr[i*4 + 1];
                float xcenter =     box_preds_data[i*num_cols + 0] * variances[0]  * anchors_dataPtr[i*4 + 2] + anchors_dataPtr[i*4 + 0];
                float h       = std::exp(box_preds_data[i*num_cols + 3] * variances[3]) * anchors_dataPtr[i*4 + 3];
                float w       = std::exp(box_preds_data[i*num_cols + 2] * variances[2]) * anchors_dataPtr[i*4 + 2];

                float ymin    = ( ycenter - h * 0.5 );
                float xmin    = ( xcenter - w * 0.5 );
                float ymax    = ( ycenter + h * 0.5 );
                float xmax    = ( xcenter + w * 0.5 );
                decoded_boxes_data[i*4] = xmin;
                decoded_boxes_data[i*4+1] = ymin;
                decoded_boxes_data[i*4+2] = xmax;
                decoded_boxes_data[i*4+3] = ymax;
            }
        }

    } // namespace host_functor
} // namespace opengl
