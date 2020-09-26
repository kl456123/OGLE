#include "opengl/core/tensor_format.h"
#include "opengl/utils/macros.h"
#include "opengl/core/driver.h"

namespace opengl{
    namespace{
        const uint64 kAlignment = 4;
    }
    int GetChannel(const IntList& shape, DataFormat dformat){
        CHECK_GE(shape.size(), 4);
        if(dformat==::dlxnet::TensorProto::NHWC){
            return shape[3];
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return shape[1];
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return shape[3];
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return shape[3];
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    int GetBatch(const IntList& shape,
            DataFormat dformat){
        CHECK_GE(shape.size(), 4);
        if(dformat==::dlxnet::TensorProto::NHWC){
            return shape[0];
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return shape[0];
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return shape[0];
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return shape[2];
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    int GetWidth(const IntList& shape,
            DataFormat dformat){
        CHECK_GE(shape.size(), 4);
        if(dformat==::dlxnet::TensorProto::NHWC){
            return shape[2];
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return shape[3];
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return shape[2];
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return shape[1];
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    int GetHeight(const IntList& shape,
            DataFormat dformat){
        CHECK_GE(shape.size(), 4);
        if(dformat==::dlxnet::TensorProto::NHWC){
            return shape[1];
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return shape[2];
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return shape[1];
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return shape[0];
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    IntList MakeTensorShape(const int batch, const int height,
            const int width, const int channels, DataFormat dformat){
        if(dformat==::dlxnet::TensorProto::NHWC){
            return {batch, height, width, channels};
        }
        if(dformat==::dlxnet::TensorProto::NCHW){
            return {batch, channels, height, width};
        }

        if(dformat==::dlxnet::TensorProto::NHWC4){
            return {batch, height, width, UP_DIV(channels, 4), 4};
        }

        if(dformat==::dlxnet::TensorProto::HWN4C4){
            return {height, width, UP_DIV(batch, 4), UP_DIV(channels, 4), 4, 4};
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    IntList TensorShapeFromFormat(DataFormat dst_format,
            const IntList& src_shape, DataFormat src_format) {
        if (src_format == dst_format) {
            return src_shape;
        }
        auto channels = GetChannel(src_shape, src_format);
        auto batch = GetBatch(src_shape, src_format);
        auto width = GetWidth(src_shape, src_format);
        auto height = GetHeight(src_shape, src_format);

        // compose them according to dst_format
        return MakeTensorShape(batch, height, width, channels, dst_format);
    }

    IntList MakeTextureShape(const IntList shape, DataFormat dformat){
        // make sure the correct dformat
        CHECK(dformat==::dlxnet::TensorProto::NHWC4
                ||::dlxnet::TensorProto::HWN4C4);
        if(dformat==::dlxnet::TensorProto::NHWC4){
            CHECK_EQ(shape.size(), 5);
            return {shape[2]*shape[3], shape[0]*shape[1], 4};
        }
        if(dformat==::dlxnet::TensorProto::HWN4C4){
            CHECK_EQ(shape.size(), 6);
            return {shape[3]*4, shape[0]*shape[1]*shape[2], 4};
        }
        LOG(FATAL)<<"dformat: "<< dformat<<" is not correct";
    }

    DataFormat StrToFormat(std::string dformat_str){
        DataFormat dformat;
        if(dformat_str=="NHWC"){
            dformat = dlxnet::TensorProto::NHWC;
        }else if(dformat_str=="NCHW"){
            dformat = dlxnet::TensorProto::NCHW;
        }else if(dformat_str=="ANY"){
            dformat = dlxnet::TensorProto::ANY;
        }else if(dformat_str=="NHWC4"){
            dformat = dlxnet::TensorProto::NHWC4;
        }else if(dformat_str=="HWN4C4"){
            dformat = dlxnet::TensorProto::HWN4C4;
        }else{
            LOG(FATAL)<<"unsupported dformat_str: "<<dformat_str;
        }
        return dformat;
    }


    std::string FormatToStr(DataFormat dformat){
        if(dformat==dlxnet::TensorProto::NHWC){
            return "NHWC";
        }else{
            LOG(FATAL)<<"unsupported dformat: "<<dformat;
        }
    }


    DataFormat FormatToStride4(DataFormat dformat){
        if(dformat==dlxnet::TensorProto::NHWC
                ||dformat==dlxnet::TensorProto::NHWC4){
            return dlxnet::TensorProto::NHWC4;
        }
        if(dformat==dlxnet::TensorProto::ANY
                ||dformat==dlxnet::TensorProto::ANY4){
            return dlxnet::TensorProto::ANY4;
        }

        if(dformat==dlxnet::TensorProto::NCHW
                ||dformat==dlxnet::TensorProto::HWN4C4){
            // only used for filter
            return dlxnet::TensorProto::HWN4C4;
        }

        LOG(FATAL)<<"unsupported dformat: "<<dformat;
    }

    DataFormat FormatFromStride4(DataFormat dformat){
        if(dformat==dlxnet::TensorProto::NHWC4
                ||dformat==dlxnet::TensorProto::NHWC){
            return dlxnet::TensorProto::NHWC;
        }
        if(dformat==dlxnet::TensorProto::ANY4
                ||dformat==dlxnet::TensorProto::ANY){
            return dlxnet::TensorProto::ANY;
        }

        if(dformat==dlxnet::TensorProto::HWN4C4
                ||dformat==dlxnet::TensorProto::NCHW){
            // only used for filter
            return dlxnet::TensorProto::NCHW;
        }
        LOG(FATAL)<<"unsupported dformat: "<<dformat;
    }

    bool IsHostDFormat(DataFormat dformat){
    }

    bool IsDeviceDFormat(DataFormat dformat){
    }


    bool IsStrideDFormat(DataFormat dformat){
        if(dformat==dlxnet::TensorProto::NHWC
                ||dformat==dlxnet::TensorProto::NCHW
                ||dformat==dlxnet::TensorProto::ANY){
            return false;
        }
        return true;
    }

    bool CheckIndexValid(const uint64 index, const IntList& shape, DataFormat dformat){
        // index is zero based
        CHECK_GT(shape.size(), 0);
        const int dim_size = shape.size();
        const int last_dim = shape[dim_size-1];

        int num_elements = 1;
        for(auto item: shape){
            num_elements*=item;
        }

        // continugous dformat
        if(dformat==dlxnet::TensorProto::NHWC
                || dformat==dlxnet::TensorProto::NCHW
                || dformat==dlxnet::TensorProto::ANY){
            return index<num_elements;
        }

        // stride dformat
        if(dformat==dlxnet::TensorProto::NHWC4
                || dformat==dlxnet::TensorProto::ANY4){
            return (index<num_elements/last_dim*UP_ROUND(last_dim, 4))
                && (index%UP_ROUND(last_dim, 4)<last_dim);
        }

        if(dformat==dlxnet::TensorProto::HWN4C4){
            // shape, oihw
            const int o4 = UP_ROUND(shape[0], 4);
            const int i4 = UP_ROUND(shape[1], 4);
            LOG(FATAL)<<"unsupported now";
        }
    }


    uint64 CalcAllocatedSize1D(const IntList& shape, DataFormat dformat){
        CHECK_GE(shape.size(), 0);
        const int dim_size = shape.size();
        const int last_dim = dim_size==0 ? 1: shape[dim_size-1] ;
        int num_elements = 1;
        for(auto item: shape){
            num_elements*=item;
        }
        const uint64 max_stride = GetMaxTextureSize()*kAlignment;

        // no stride dformat: NHWC, NCHW, ANY
        // stride dformat: ANY4
        // special case: NHWC4, HWN4C4
        if(!IsStrideDFormat(dformat)){
            uint64 align_num_elements = UP_ROUND(num_elements, kAlignment);
            if(align_num_elements>max_stride){
                return UP_ROUND(align_num_elements, max_stride);
            }
            return align_num_elements;
        }

        if(dformat==dlxnet::TensorProto::ANY4){
            // change the last dim
            uint64 align_dim = UP_ROUND(last_dim, kAlignment);
            uint64 align_num_elements = num_elements/last_dim*align_dim;
            if(align_num_elements>max_stride){
                return UP_ROUND(align_num_elements, max_stride);
            }
            return align_num_elements;
        }

        CHECK_EQ(shape.size(), 4);
        // nh, wc/4, 4
        if(dformat==dlxnet::TensorProto::NHWC4){
            int image_height = shape[0] * shape[1];
            int image_width = UP_DIV(shape[3], kAlignment) * shape[2];
            return image_width*image_height*kAlignment;
        }
        if(dformat==dlxnet::TensorProto::HWN4C4){
            // make sure filter shape should be oihw format
            // oihw
            // hwo/4, i/4*4, 4
            int image_height = shape[3]*shape[2]*UP_DIV(shape[0], kAlignment);
            int image_width = UP_ROUND(shape[1], kAlignment);
            return image_height*image_width*kAlignment;
        }
        LOG(FATAL)<<"unsupported dformat_str: "<<FormatToStr(dformat);
    }

    IntList CalcAllocatedSize2D(const IntList& shape, DataFormat dformat){
        CHECK_GE(shape.size(), 0);
        const int dim_size = shape.size();
        const int last_dim = dim_size==0 ? 1: shape[dim_size-1] ;
        int num_elements = 1;
        for(auto item: shape){
            num_elements*=item;
        }
        const uint64 max_stride = GetMaxTextureSize()*kAlignment;

        // no stride dformat: NHWC, NCHW, ANY
        // stride dformat: ANY4
        // special case: NHWC4, HWN4C4
        if(!IsStrideDFormat(dformat)){
            uint64 align_num_elements = UP_ROUND(num_elements, kAlignment);
            if(align_num_elements>max_stride){
                return {int(UP_DIV(align_num_elements, max_stride)), int(max_stride/kAlignment)};
            }
            return {1, int(align_num_elements/kAlignment)};
        }

        if(dformat==dlxnet::TensorProto::ANY4){
            // change the last dim
            uint64 align_dim = UP_ROUND(last_dim, kAlignment);
            uint64 align_num_elements = num_elements/last_dim*align_dim;
            if(align_num_elements>max_stride){
                return {int(UP_DIV(align_num_elements, max_stride)), int(max_stride/kAlignment)};
            }
            return {1, int(align_num_elements/kAlignment)};
        }

        CHECK_EQ(shape.size(), 4);
        // nh, wc/4, 4
        if(dformat==dlxnet::TensorProto::NHWC4){
            int image_height = shape[0] * shape[1];
            int image_width = UP_DIV(shape[3], kAlignment) * shape[2];
            return {image_height, image_width};
        }
        if(dformat==dlxnet::TensorProto::HWN4C4){
            // make sure filter shape should be oihw format
            // oihw
            // hwo/4, i/4*4, 4
            int image_height = shape[3]*shape[2]*UP_DIV(shape[0], kAlignment);
            int image_width = UP_ROUND(shape[1], kAlignment);
            return {image_height, image_width};
        }
        LOG(FATAL)<<"unsupported dformat_str: "<<FormatToStr(dformat);
    }
}//namespace opengl
