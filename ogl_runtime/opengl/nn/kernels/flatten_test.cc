#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/flatten.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        void FlattenCPU(const float* input_data, const int num_elements,
                const IntList& input_shape, const IntList& output_shape,
                float* output_data){
            for(int i=0;i<num_elements;++i){
                auto output_coord = OffsetToCoord(i, output_shape);
                const int input_index = CoordToOffset(output_coord, input_shape);
                output_data[i] = input_data[input_index];
            }
        }

        const ::dlxnet::ModelProto BuildGraph(const Tensor* cpu_tensor, const int axis){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const_id = AddConstNode(scope_ptr, "const", cpu_tensor);
            AddFlattenNode(scope_ptr, "output", {const_id}, {axis});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }

        void SingleInference(const IntList& input_shape, const int axis){
            auto session = InitSession();
            const auto const_tensor = std::unique_ptr<Tensor>(
                    Tensor::Random(Tensor::DT_FLOAT, input_shape,
                        dlxnet::TensorProto::ANY));

            session->LoadGraph(BuildGraph(const_tensor.get(), axis));

            session->Run({});

            // get cpu outputs from device
            ::opengl::TensorList outputs_cpu;
            ::opengl::TensorNameList output_names({"output"});
            ::opengl::StringList dformats({"ANY"});
            session->GetOutputs(output_names, dformats, &outputs_cpu);

            // FlattenCPU();

            CheckSameValueTensor(outputs_cpu[0], const_tensor.get());
            CleanupTensorList(&outputs_cpu);
        }
    } // namespace

    TEST(FlattenTest, SimpleTest){
        DIFFERENT_SHAPE_LOOP_START;
        const int axis = 1;
        SingleInference(shape, axis);
        DIFFERENT_SHAPE_LOOP_END;
    }
}
