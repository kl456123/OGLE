#include <memory>
#include "opengl/test/test.h"
#include "opengl/core/tensor.h"
#include "opengl/core/types.h"


namespace opengl{
    namespace{
        // some helper functions
        // assert tensor shape equality
        inline void ExpectTensorShapeEqual(const IntList& a, const IntList& b){
            EXPECT_EQ(a.size(), b.size());
            for(int i=0;i<a.size();++i){
                EXPECT_EQ(a[i], b[i]);
            }
        }
        template<typename T>
            inline void ExpectEqual(const T& a, const T& b){
                // int case
                EXPECT_EQ(a, b);
            }
        template<>
            inline void ExpectEqual(const float& a, const float& b){
                EXPECT_FLOAT_EQ(a, b);
            }
        // assert tensor value equality, dont care it shape
        // but should make sure that the num_elements should be
        // the same first.
        template<typename T>
            inline void ExpectTensorValueEqual(const Tensor& a, const Tensor& b){
                // check num_elements first
                EXPECT_EQ(a.num_elements(), b.num_elements());
                EXPECT_EQ(a.dtype(), b.dtype());
                const T* x = a.host<T>();
                const T* y = b.host<T>();
                const auto size = a.num_elements();
                for(int i=0;i<size;++i){
                    // due to interge and float should be deal with
                    // in different cases.
                    ExpectEqual(x[i], y[i]);
                }
            }
        // assert tensor, just assert its value and shape at the same time
        template<typename T>
            inline void ExpectTensorEqual(const Tensor& a, const Tensor& b){
                ExpectTensorShapeEqual(a.shape(), b.shape());
                ExpectTensorValueEqual<T>(a, b);
            }
    }//namespace

    TEST(TensorHostTest, NHWC){
        IntList image_shape = {1, 224, 224, 3};
        auto tensor = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, image_shape));
        ExpectTensorShapeEqual(tensor->shape(), image_shape);
        EXPECT_TRUE(tensor->is_host());
        EXPECT_EQ(tensor->dtype(), Tensor::DT_FLOAT);
        EXPECT_EQ(tensor->mem_type(), Tensor::HOST_MEMORY);
        // default dformat nhwc
        EXPECT_EQ(tensor->dformat(), dlxnet::TensorProto::NHWC);

        // dim helper
        EXPECT_EQ(tensor->channel(), 3);
        EXPECT_EQ(tensor->num(), 1);
        EXPECT_EQ(tensor->width(), 224);
        EXPECT_EQ(tensor->height(), 224);
    }

    TEST(TensorHostTest, NCHW){
        IntList image_shape = {1, 3, 224, 224};
        auto tensor = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, image_shape,
                    dlxnet::TensorProto::NCHW));
        EXPECT_TRUE(tensor->is_host());
        EXPECT_EQ(tensor->dtype(), Tensor::DT_FLOAT);
        EXPECT_EQ(tensor->mem_type(), Tensor::HOST_MEMORY);
        ExpectTensorShapeEqual(tensor->shape(), image_shape);
        EXPECT_EQ(tensor->dformat(), dlxnet::TensorProto::NCHW);

        // dim helper
        EXPECT_EQ(tensor->channel(), 3);
        EXPECT_EQ(tensor->num(), 1);
        EXPECT_EQ(tensor->width(), 224);
        EXPECT_EQ(tensor->height(), 224);
    }

    TEST(TensorTextureTest, NHWC4){
        // used for most common cases
        // input shape in nhwc
        IntList image_shape = {1, 224, 224, 3};
        auto tensor = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, image_shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NHWC4));
        EXPECT_FALSE(tensor->is_host());
        EXPECT_EQ(tensor->dtype(), Tensor::DT_FLOAT);
        EXPECT_EQ(tensor->mem_type(), Tensor::DEVICE_TEXTURE);
        EXPECT_EQ(tensor->dformat(), dlxnet::TensorProto::NHWC4);

        // dim helper
        EXPECT_EQ(tensor->channel(), 3);
        EXPECT_EQ(tensor->num(), 1);
        EXPECT_EQ(tensor->width(), 224);
        EXPECT_EQ(tensor->height(), 224);

        // check texture width and height
        // nhwc4
        auto texture = tensor->device<Texture>();
        EXPECT_EQ(texture->width(), UP_DIV(3, 4)*224);
        EXPECT_EQ(texture->height(), 1*224);
    }

    TEST(TensorTextureTest, HWN4C4){
        // used only for filter
        // input shape in nchw
        IntList filter_shape = {1, 30, 4, 3};
        auto tensor = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, filter_shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::HWN4C4));
        EXPECT_FALSE(tensor->is_host());
        EXPECT_EQ(tensor->dtype(), Tensor::DT_FLOAT);
        EXPECT_EQ(tensor->mem_type(), Tensor::DEVICE_TEXTURE);
        EXPECT_EQ(tensor->dformat(), dlxnet::TensorProto::HWN4C4);

        /// dim helper
        EXPECT_EQ(tensor->channel(), 30);
        EXPECT_EQ(tensor->num(), 1);
        EXPECT_EQ(tensor->width(), 3);
        EXPECT_EQ(tensor->height(), 4);

        // check texture width and height
        // hwn4c4
        auto texture = tensor->device<Texture>();
        EXPECT_EQ(texture->width(), UP_DIV(1, 4)*4*UP_DIV(30, 4));
        EXPECT_EQ(texture->height(), 4*3);
    }

    TEST(TensorTest, FromProto){
    }

    TEST(TensorTest, FromHost){
    }

    TEST(TensorTest, SizeTest){
        // cpu tensor
        IntList image_shape = {1, 3, 30, 30};
        std::vector<DataFormat> tested_dformats = {
            dlxnet::TensorProto::ANY,
            dlxnet::TensorProto::ANY4,
            // dlxnet::TensorProto::NHWC,
            // dlxnet::TensorProto::NCHW,
            dlxnet::TensorProto::NHWC4,
            dlxnet::TensorProto::HWN4C4
        };

        for(auto dformat: tested_dformats){
            auto gpu_tensor = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, image_shape,
                        Tensor::DEVICE_TEXTURE, dformat));
            auto cpu_tensor = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, image_shape,
                        dformat));
            EXPECT_EQ(gpu_tensor->AllocatedSize(), cpu_tensor->AllocatedSize());
        }
        // const int num_elements = tensor->num_elements();
        // EXPECT_EQ(tensor->AllocatedSize(), num_elements );
        // EXPECT_EQ(tensor->RequestedSize(), 1*3*224*224*sizeof(float));
        // EXPECT_EQ(tensor->size(), 1*3*224*224*sizeof(float));
    }
}//namespace opengl
