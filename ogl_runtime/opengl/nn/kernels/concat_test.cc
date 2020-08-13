#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/concat.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        void ConcatCPU(const float* input1, const float* input2,
                const int axis, const IntList& input_shape1, const IntList& input_shape2,
                const IntList& output_shape, float* output){
            // get total num first
            int output_num_elements = 1;
            for(auto item:output_shape){
                output_num_elements*=item;
            }

            for(int i=0;i<output_num_elements;++i){
                // get its coords in output
                auto coord = OffsetToCoord(i, output_shape);
                const int index = coord[axis];
                float value;
                if(index<input_shape1[axis]){
                    // use input1
                    const int offset = CoordToOffset(coord, input_shape1);
                    value = input1[offset];
                }else{
                    coord[axis] = index - input_shape1[axis];
                    const int offset = CoordToOffset(coord, input_shape2);
                    value = input2[offset];
                }
                output[i] = value;
            }
        }


        const ::dlxnet::ModelProto BuildGraph(const Tensor* cpu_tensor1,
                const Tensor* cpu_tensor2, const int axis){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const1_id = AddConstNode(scope_ptr, "const1", cpu_tensor1);
            int const2_id = AddConstNode(scope_ptr, "const2", cpu_tensor2);
            AddConcatNode(scope_ptr, "output", {const1_id, const2_id}, {axis});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }

        void SingleInference(const IntList& shape1, const IntList& shape2, const int axis){
            auto session = InitSession();
            const auto const1 = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape1,
                    dlxnet::TensorProto::ANY));
            const auto const2 = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape2,
                    dlxnet::TensorProto::ANY));
            session->LoadGraph(BuildGraph(const1.get(), const2.get(), axis));

            ::opengl::TensorList outputs_cpu;
            // do computation for the graph
            session->Run({});

            ::opengl::TensorNameList output_names({"output", "const1", "const2"});
            ::opengl::StringList dformats({"ANY", "ANY", "ANY"});

            // get cpu outputs from device
            session->GetOutputs(output_names, dformats, &outputs_cpu);
            // check const
            CheckSameTensor(outputs_cpu[1], const1.get());
            CheckSameTensor(outputs_cpu[2], const2.get());

            // check the result
            // check the shape first
            const float* ogl_output_data = outputs_cpu[0]->host<float>();
            const int output_num_elements = outputs_cpu[0]->num_elements();
            const auto output_shape = outputs_cpu[0]->shape();
            for(int i=0;i<output_shape.size();++i){
                if(axis!=i){
                    EXPECT_EQ(output_shape[i], shape1[i]);
                }else{
                    EXPECT_EQ(output_shape[i], shape1[i]+shape2[i]);
                }
            }
            // check the value
            const float* ogl_const1_data = outputs_cpu[1]->host<float>();
            const float* ogl_const2_data = outputs_cpu[2]->host<float>();
            float* cpu_output_data = new float[output_num_elements];
            ConcatCPU(ogl_const1_data, ogl_const2_data, axis, outputs_cpu[1]->shape(),
                    outputs_cpu[2]->shape(), outputs_cpu[0]->shape(), cpu_output_data);
            for(int i=0;i<output_num_elements;++i){
                EXPECT_EQ(ogl_output_data[i], cpu_output_data[i])<<"When index: "<<i;
            }

            CleanupTensorList(&outputs_cpu);
        }
    }// namespace

    TEST(ConcatTest, SimpleTest){
        const IntList shape1{2, 3, 5};
        const IntList shape2{2, 3, 5};
        const int axis = 1;
        SingleInference(shape1, shape2, axis);
    }

    TEST(ConcatTest, DimsTest){
        const IntList shape1{2, 3, 5};
        const IntList shape2{2, 3, 5};
        const int axis = 2;
        SingleInference(shape1, shape2, axis);
    }

    TEST(ConcatTest, DimAndShapeTest){
        // nhwc
        const int max_dims = 10;
        const int max_dims_size = 4;
        for(int dims_size=1; dims_size<=max_dims_size;++dims_size){
            for(int axis=0;axis<dims_size;++axis){
                IntList shape1, shape2;
                // generate random shape
                for(int i=1;i<=dims_size;++i){
                    shape1.emplace_back(random()%max_dims+1);
                }
                shape2 = shape1;
                shape2[axis] = random()%max_dims+1;
                SingleInference(shape1, shape2, axis);
            }
        }
    }
}//namespace opengl
