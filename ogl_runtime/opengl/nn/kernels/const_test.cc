#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/const.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        const ::dlxnet::ModelProto BuildGraph(const Tensor* cpu_tensor){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const_id = AddConstNode(scope_ptr, "output", cpu_tensor);

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }

        void SingleInference(const IntList& shape){
            const auto cpu_tensor = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
            auto session = InitSession();
            session->LoadGraph(BuildGraph(cpu_tensor.get()));

            // do computation for the graph
            session->Run({});

            ::opengl::TensorNameList output_names({"output"});
            ::opengl::StringList dformats({"ANY"});

            // get cpu outputs from device
            ::opengl::TensorList outputs_cpu;
            session->GetOutputs(output_names, dformats, &outputs_cpu);
            CheckSameTensor(cpu_tensor.get(), outputs_cpu[0]);
            CleanupTensorList(&outputs_cpu);
        }
    }// namespace

    TEST(ConstTest, AnyTest){
        const int max_dims_size = 4;
        const int max_dims = 10;
        for(int dims_size=1; dims_size<=max_dims_size; ++dims_size){
            IntList shape;
            // generate random shape
            for(int i=1;i<=dims_size;++i){
                shape.emplace_back(random()%max_dims+1);
            }
            SingleInference(shape);
        }
    }

    TEST(ConstTest, NHWCTest){
    }

    TEST(ConstTest, NCHWTest){
    }
}//namespace opengl
