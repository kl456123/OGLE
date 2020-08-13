#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/flatten.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        void GemmCPU(const Tensor* tensor_a, const Tensor* tensor_b, const Tensor* tensor_c,
                Tensor* tensor_y, const float alpha,
                const float beta, const int transb){
            // assume all tensors are in host memory
            const float* a = tensor_a->host<float>();
            const float* b = tensor_b->host<float>();
            const float* c = tensor_c->host<float>();
            float* y = tensor_y->host<float>();
            // CHECK_EQ(transb, 0);

            auto y_shape = tensor_y->shape();
            const int num_elements = tensor_y->num_elements();
            // (M, K) * (K, N) + (M, N) = (M, N)
            // (M, K) * (N, K) + (M, N) = (M, N)
            const int M = y_shape[0];
            const int N = y_shape[1];
            const int K = tensor_a->shape()[1];
            for(int i=0; i<M; ++i){
                for(int j=0; j<N; ++j){
                    y[i*N+j] = 0;
                    for(int m=0; m<K; ++m){
                        // (i, j) = (i, m) * (m, j)
                        if(transb){
                            y[i*N+j] += alpha * a[i*K+m] * b[j*K+ m];
                        }else{
                            y[i*N+j] += alpha * a[i*K+m] * b[m*N+j];
                        }
                    }
                    y[i*N+j]+=beta * c[i*N+j];
                }
            }
        }

        const ::dlxnet::ModelProto BuildGraph(const Tensor* cpu_tensor1,
                const Tensor* cpu_tensor2, const Tensor* cpu_tensor3,
                const float alpha, const float beta, const int transb){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const1_id = AddConstNode(scope_ptr, "const1", cpu_tensor1);
            int const2_id = AddConstNode(scope_ptr, "const2", cpu_tensor2);
            int const3_id = AddConstNode(scope_ptr, "const3", cpu_tensor3);
            AddGemmNode(scope_ptr, "output", {const1_id, const2_id, const3_id}, {alpha, beta, transb});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }

        IntList ComputeShape(const IntList& a_shape,
                const IntList& b_shape, const int transb){
            CHECK_EQ(a_shape.size(), 2);
            CHECK_EQ(b_shape.size(), 2);
            if(transb){
                CHECK_EQ(a_shape[1], b_shape[1]);
                return {a_shape[0], b_shape[0]};
            }else{
                CHECK_EQ(a_shape[1], b_shape[0]);
                return {a_shape[0], b_shape[1]};
            }
        }

        void SingleInference(const IntList& input1_shape, const IntList& input2_shape,
                const IntList& input3_shape, const float alpha, const float beta, const int transb){
            auto session = InitSession();
            const auto const_tensor1_ptr = std::unique_ptr<Tensor>(
                    Tensor::Random(Tensor::DT_FLOAT, input1_shape,
                        dlxnet::TensorProto::ANY));

            const auto const_tensor2_ptr = std::unique_ptr<Tensor>(
                    Tensor::Random(Tensor::DT_FLOAT, input2_shape,
                        dlxnet::TensorProto::ANY));

            auto actual_shape = ComputeShape(input1_shape, input2_shape, transb);
            const auto const_tensor3_ptr = std::unique_ptr<Tensor>(
                    Tensor::Random(Tensor::DT_FLOAT, input3_shape,
                        dlxnet::TensorProto::ANY));

            const auto actual_tensor_ptr = std::unique_ptr<Tensor>(
                    Tensor::Zeros(Tensor::DT_FLOAT, actual_shape,
                        dlxnet::TensorProto::ANY));

            Tensor* const_tensor1 = const_tensor1_ptr.get();
            Tensor* const_tensor2 = const_tensor2_ptr.get();
            Tensor* const_tensor3 = const_tensor3_ptr.get();
            Tensor* actual_tensor = actual_tensor_ptr.get();

            session->LoadGraph(BuildGraph(const_tensor1, const_tensor2,
                        const_tensor3, alpha, beta, transb));

            session->Run({});

            // get cpu outputs from device
            ::opengl::TensorList outputs_cpu;
            ::opengl::TensorNameList output_names({"output"});
            ::opengl::StringList dformats({"ANY"});
            session->GetOutputs(output_names, dformats, &outputs_cpu);

            GemmCPU(const_tensor1, const_tensor2, const_tensor3, actual_tensor,
                    alpha, beta, transb);

            CheckSameTensor(actual_tensor, outputs_cpu[0]);
            CleanupTensorList(&outputs_cpu);
        }

    } // namespace

    TEST(GemmTest, SimpleTest){
        IntList shape1{10, 1024};
        IntList shape2{1024, 1000};
        IntList shape3{10, 1000};
        float alpha = 1.0;
        float beta = 1.0;
        int transb = 0;
        SingleInference(shape1, shape2, shape3, alpha, beta, transb);
    }

    TEST(GemmTest, WithTransBTest){
        IntList shape1{1, 2048};
        IntList shape2{1000, 2048};
        IntList shape3{1000};
        float alpha = 1.0;
        float beta = 1.0;
        int transb = 1;
        SingleInference(shape1, shape2, shape3, alpha, beta, transb);
    }
} // namespace opengl
