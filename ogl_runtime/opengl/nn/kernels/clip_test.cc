#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/clip.h"

#include <cmath>


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        void ClipCPU(const float* const_data, float min, float max,
                const int num_elements, float* cpu_output_data){
            for(int i=0;i<num_elements;++i){
                cpu_output_data[i] = std::min(max, const_data[i]);
                cpu_output_data[i] = std::max(min, const_data[i]);
            }
        }

        const ::dlxnet::ModelProto BuildGraph(const Tensor* cpu_tensor,
                float min, float max){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const_id = AddConstNode(scope_ptr, "const", cpu_tensor);
            AddClipNode(scope_ptr, "output", {const_id}, {min, max});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }

        void SingleInference(const IntList& shape, float min, float max){
            auto session = InitSession();
            const auto const_tensor = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                        dlxnet::TensorProto::ANY));

            session->LoadGraph(BuildGraph(const_tensor.get(), min, max));

            ::opengl::TensorList outputs_cpu;

            // do computation for the graph
            session->Run({});

            ::opengl::TensorNameList output_names({"output"});
            ::opengl::StringList dformats({"ANY"});

            // get cpu outputs from device
            session->GetOutputs(output_names, dformats, &outputs_cpu);

            // check the result
            // check the shape first
            const float* ogl_output_data = outputs_cpu[0]->host<float>();
            const int output_num_elements = outputs_cpu[0]->num_elements();
            const auto output_shape = outputs_cpu[0]->shape();
            for(int i=0;i<output_shape.size();++i){
                EXPECT_EQ(output_shape[i], shape[i]);
            }
            // check the value
            float* cpu_output_data = new float[output_num_elements];
            const float* const_data = const_tensor->host<float>();
            ClipCPU(const_data, min, max, output_num_elements, cpu_output_data);
            for(int i=0; i<output_num_elements; ++i){
                EXPECT_EQ(ogl_output_data[i], cpu_output_data[i])<<"When index: "<<i;
            }

            CleanupTensorList(&outputs_cpu);
        }
    }//namespace
    TEST(ClipTest, SimpleTest){
        float min = 0.0;
        float max = 6.0;
        const IntList shape{1, 2, 3, 5};

        SingleInference(shape, min, max);
    }
}//namespace opengl
