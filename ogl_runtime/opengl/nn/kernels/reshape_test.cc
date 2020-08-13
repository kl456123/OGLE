#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/reshape.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        const ::dlxnet::ModelProto BuildGraph(const Tensor* cpu_tensor,
                const Tensor* shape_tensor){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const_id = AddConstNode(scope_ptr, "const", cpu_tensor);
            int shape_tensor_id = AddConstNode(scope_ptr, "const", shape_tensor);
            int reshape_id = AddReshapeNode(scope_ptr, "output", {const_id, shape_tensor_id});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }

        void SingleInference(const IntList& src_shape,
                const IntList& dst_shape){
            auto session = InitSession();
            const auto const_tensor = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, src_shape,
                    dlxnet::TensorProto::ANY));
            IntList target_shape;
            target_shape.emplace_back(dst_shape.size());
            // no need to clean up, its ownership moved to tensor
            float* shape_data = new float[dst_shape.size()];
            for(int i=0;i<dst_shape.size();++i){
                shape_data[i] = float(dst_shape[i]);
            }
            const auto shape_tensor = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, target_shape, shape_data,
                    dlxnet::TensorProto::ANY));
            session->LoadGraph(BuildGraph(const_tensor.get(), shape_tensor.get()));

            ::opengl::TensorList outputs_cpu;
            session->Run({});
            ::opengl::TensorNameList output_names({"output"});
            ::opengl::StringList dformats({"ANY"});

            // get cpu outputs from device
            session->GetOutputs(output_names, dformats, &outputs_cpu);
            auto output_shape = outputs_cpu[0]->shape();
            for(int i=0;i<output_shape.size();++i){
                EXPECT_EQ(output_shape[i], dst_shape[i]);
            }

            CheckSameValueTensor(outputs_cpu[0], const_tensor.get());
            CleanupTensorList(&outputs_cpu);
        }
    }

    TEST(ReshapeTest, SameShapeTest){
        const int max_dims = 20;
        const int max_dims_size = 4;
        for(int dims_size=1; dims_size<=max_dims_size;++dims_size){
            IntList shape1, shape2;
            // generate random shape
            for(int i=1;i<=dims_size;++i){
                shape1.emplace_back(random()%max_dims+1);
            }
            shape2 = shape1;
            SingleInference(shape1, shape2);
        }
    }

    TEST(ReshapeTest, OneFirstShapeTest){
        const int max_dims = 20;
        const int max_dims_size = 4;
        for(int dims_size=1; dims_size<=max_dims_size;++dims_size){
            IntList shape1, shape2;
            int num=1;
            // generate random shape
            for(int i=1;i<=dims_size;++i){
                int value = random()%max_dims+1;
                shape1.emplace_back(value);
                num*=value;
            }
            shape2 = {1, num};
            SingleInference(shape1, shape2);
        }
    }

    TEST(ReshapeTest, SpecialShapeTest){
        SingleInference({1, 2, 4, 4}, {1, 32});
    }
}//namespace opengl
