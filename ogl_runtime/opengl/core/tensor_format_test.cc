#include <memory>
#include "opengl/test/test.h"
#include "opengl/core/tensor_format.h"
#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/core/types.h"


namespace opengl{
    namespace{
        struct ShapeAndDFormat{
            IntList shape;
            DataFormat dformat;
            int channel;
            int batch;
            int height;
            int width;
        };
        typedef std::vector<ShapeAndDFormat> ShapeAndDFormatList;
        void GetAllShapeAndDFormat(ShapeAndDFormatList* list){
            // four types of dformat
            // list->resize(4);
            list->emplace_back(ShapeAndDFormat({{10, 300, 224, 3},
                        ::dlxnet::TensorProto::NHWC, 3, 10, 300, 224}));
            list->emplace_back(ShapeAndDFormat({{10, 3, 300,224},
                        ::dlxnet::TensorProto::NCHW, 3, 10, 300, 224}));
            list->emplace_back(ShapeAndDFormat({{10, 300, 224, 2, 4},
                        ::dlxnet::TensorProto::NHWC4, 2, 10, 300, 224}));
            list->emplace_back(ShapeAndDFormat({{5, 5, 16, 1,4,4},
                        ::dlxnet::TensorProto::HWN4C4, 1, 16, 5, 5}));
        }
    }
    TEST(TensorFormat, GetChannel){
        ShapeAndDFormatList list;
        GetAllShapeAndDFormat(&list);
        for(auto& item: list){
            int channel = GetChannel(item.shape, item.dformat);
            EXPECT_EQ(channel, item.channel);
        }
    }

    TEST(TensorFormat, GetBatch){
        ShapeAndDFormatList list;
        GetAllShapeAndDFormat(&list);
        for(auto& item: list){
            int channel = GetBatch(item.shape, item.dformat);
            EXPECT_EQ(channel, item.batch);
        }
    }

    TEST(TensorFormat, CalcAllocatedSize1DTest){
        DIFFERENT_SHAPE_LOOP_START;
        // const IntList& shape = {1, 320, 320, 3};
        DataFormat dformat1 = dlxnet::TensorProto::NHWC;
        DataFormat dformat2 = dlxnet::TensorProto::ANY;
        auto size = CalcAllocatedSize1D(shape, dformat1);
        auto texture_shape = CalcAllocatedSize2D(shape, dformat2);

        EXPECT_EQ(texture_shape[0]*texture_shape[1]*4, size);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(TensorFormat, CalcAllocatedSize2DTest){
    }
}//namespace opengl
