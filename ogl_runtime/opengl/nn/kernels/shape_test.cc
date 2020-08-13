#include <cmath>
#include <memory>
#include <random>
#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/shape.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        int input_width = 3;
        int input_height = 3;
        int input_channels = 3;
        int num_inputs = 1;
        constexpr float precision = 1e-4;

        const ::dlxnet::ModelProto BuildGraph(){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int input_id = AddInputNode(scope_ptr, "input");

            AddShapeNode(scope_ptr, "output", {input_id});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }

        void SingleTest(const std::vector<int>& image_shape){
            // graph session
            auto session = InitSession();
            session->LoadGraph(BuildGraph());

            // std::vector<int> image_shape = {num_inputs, input_height, input_width, input_channels};
            ::opengl::NamedTensorList inputs(1);
            ::opengl::TensorList outputs_cpu;
            inputs[0].first = "input";
            inputs[0].second = Tensor::Random(Tensor::DT_FLOAT, image_shape, dlxnet::TensorProto::ANY);

            // do computation for the graph
            session->Run(inputs);

            ::opengl::TensorNameList output_names({"output"});
            ::opengl::StringList dformats({"ANY"});

            // get cpu outputs from device
            session->GetOutputs(output_names, dformats, &outputs_cpu);

            // check the result
            const float* ogl_output_data = outputs_cpu[0]->host<float>();
            const int output_num_elements = outputs_cpu[0]->num_elements();
            const int* cpu_output_data = image_shape.data();
            for(int i=0;i<output_num_elements;++i){
                float actual_value = ogl_output_data[i];
                float expect_value = cpu_output_data[i];
                EXPECT_TRUE(std::fabs(actual_value - expect_value)< precision)<<"Error When index: "<< i
                    <<" Actualy Value: "<<actual_value<<" Extect Value: "<<expect_value;
            }
        }
    }// namespace

    TEST(ShapeTest, SimpleTest){
        SingleTest({num_inputs, input_height, input_width, input_channels});
        SingleTest({input_width, input_channels});
    }
}//namespace opengl
