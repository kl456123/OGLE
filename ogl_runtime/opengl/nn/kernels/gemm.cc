#include "opengl/nn/kernels/gemm.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/util.h"


namespace opengl{
    GemmKernel::GemmKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/gemm.glsl";
        }

    void GemmKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& gemm_params = attr.gemm_attr();
        alpha_ = gemm_params.alpha();
        beta_ = gemm_params.beta();
        transB_ = gemm_params.transb();

        // single output
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);
    }

    void GemmKernel::Compute(TensorList& inputs, TensorList& outputs){
        CHECK_EQ(inputs[0]->dformat(), dlxnet::TensorProto::ANY4);
        CHECK_EQ(inputs[1]->dformat(), dlxnet::TensorProto::ANY4);
        DLOG(INFO)<<"GemmKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();
        auto input_filter = inputs[1]->device<Texture>();
        bool use_bias = inputs.size()>2;
        program_->SetRetVal(outputs);

        auto input_shape = inputs[0]->shape();
        auto output_shape = outputs[0]->shape();

        program_->set_vec2i("input_shape", inputs[0]->shape()[0], inputs[0]->shape()[1]);
        program_->set_vec2i("output_shape", outputs[0]->shape()[0], outputs[0]->shape()[1]);
        program_->set_int("use_bias", int(use_bias));
        program_->set_int("transb", transB_);
        program_->set_float("beta", beta_);
        program_->set_float("alpha", alpha_);
        // input
        {
            program_->set_image2D("A", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // filter
        {
            program_->set_image2D("B", input_filter->id(),  1);
            OPENGL_CHECK_ERROR;
        }
        if(use_bias){
            // bias
            auto input_bias = inputs[2]->device<Texture>();
            program_->set_image2D("C", input_bias->id(),  2);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }


    void GemmKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // Y = XW+B
        // X, W, B, like conv2d
        // X: (N, C_in)
        // W: (C_in, C_out)
        // B: (N, C_out)
        // Y: (N, C_out)
        // TODO(breakpoint) merge gemm to conv1x1
        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes[0].size(), 2);
        CHECK_EQ(input_shapes[1].size(), 2);

        if(transB_){
            // check in_channels
            CHECK_EQ(input_shapes[1][1], input_shapes[0][1]);
            // Y: (N, C_out)
            output_shapes[0]={input_shapes[0][0], input_shapes[1][0]};
        }else{
            CHECK_EQ(input_shapes[1][0], input_shapes[0][1]);
            output_shapes[0]={input_shapes[0][0], input_shapes[1][1]};
        }

        if(input_shapes.size()>2){
            if(input_shapes[2].size()==2){
                //no need to broadcast
                CHECK_EQ(output_shapes[0][0], input_shapes[2][0]);
                CHECK_EQ(output_shapes[0][1], input_shapes[2][1]);
            }else{
                CHECK_EQ(input_shapes[2].size(), 1);
                // C can be broadcast to (M, N)
                CHECK_EQ(output_shapes[0][1], input_shapes[2][0]);
            }
        }

    }

    GemmKernel::~GemmKernel(){}

    REGISTER_KERNEL_WITH_NAME(GemmKernel, "Gemm");
}//namespace opengl
