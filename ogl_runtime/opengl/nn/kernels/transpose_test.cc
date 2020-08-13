#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/transpose.h"

using namespace ::opengl::testing;

namespace opengl{
    namespace{
        IntList RandomPerm(const int dim_size){
            IntList perm;
            for(int i=dim_size-1;i>=0;--i){
                perm.emplace_back(i);
            }
            return perm;
        }
        IntList TransposeShape(const IntList& shape, const IntList& perms){
            IntList dst_shape;
            // permute dims in perms
            for(auto perm: perms){
                dst_shape.emplace_back(shape[perm]);
            }
            // append the remain dims
            for(int i=perms.size();i<shape.size();++i){
                dst_shape.emplace_back(shape[i]);
            }
            return dst_shape;
        }

        IntList TransposeOutputCoord(const IntList& coord, const IntList& perms){
            CHECK_EQ(coord.size(), perms.size());
            IntList input_coord(4);
            // permute dims in perms
            for(int i=0;i<perms.size();++i){
                input_coord[perms[i]] = coord[i];
            }
            // append the remain dims
            for(int i=perms.size();i<coord.size();++i){
                input_coord.emplace_back(coord[i]);
            }
            return input_coord;
        }

        void TransposeCPU(const float* src_data, float* dst_data,
                const IntList& input_shape, const IntList& output_shape, const IntList& perm){
            // get total num first
            int output_num_elements = 1;
            for(auto item:output_shape){
                output_num_elements*=item;
            }

            for(int i=0;i<output_num_elements;++i){
                // get its coords in output
                auto coord = OffsetToCoord(i, output_shape);
                const auto input_coord = TransposeOutputCoord(coord, perm);
                const auto input_offset = CoordToOffset(input_coord, input_shape);
                dst_data[i] = src_data[input_offset];
            }
        }


        const ::dlxnet::ModelProto BuildGraph(const Tensor* const_tensor, const IntList& perm){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const_tensor_id = AddConstNode(scope_ptr, "const", const_tensor);
            AddTransposeNode(scope_ptr, "output", {const_tensor_id}, {perm});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }

        void SingleInference(const IntList& shape, const IntList& perm){
            auto session = InitSession();
            const auto const_tensor = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                        dlxnet::TensorProto::ANY));
            session->LoadGraph(BuildGraph(const_tensor.get(), perm));

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
            auto cpu_output_shape = TransposeShape(shape, perm);
            const float* cpu_src_data = const_tensor->host<float>();
            for(int i=0;i<output_shape.size();++i){
                EXPECT_EQ(output_shape[i], cpu_output_shape[i]);
            }

            float* cpu_output_data = new float[output_num_elements];
            TransposeCPU(cpu_src_data, cpu_output_data, shape, cpu_output_shape, perm);

            for(int i=0;i<output_num_elements;++i){
                EXPECT_EQ(ogl_output_data[i], cpu_output_data[i])<<"When index: "<<i;
            }

            CleanupTensorList(&outputs_cpu);
        }
    }// namespace
    TEST(TransposeTest, SimpleTest){
        const IntList shape{2, 3, 5};
        const IntList perm{0, 2, 1};
        SingleInference(shape, perm);
    }

    TEST(TransposeTest, DimAndShapeTest){
        // nhwc
        const int max_dims = 10;
        const int max_dims_size = 4;
        for(int dims_size=1; dims_size<=max_dims_size;++dims_size){
            IntList shape;
            // generate random shape
            for(int i=1;i<=dims_size;++i){
                shape.emplace_back(random()%max_dims+1);
            }
            // generate random perms
            IntList perm = RandomPerm(dims_size);
            SingleInference(shape, perm);
        }
    }
}//namespace opengl
