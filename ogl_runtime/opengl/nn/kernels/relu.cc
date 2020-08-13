#include "opengl/nn/kernels/relu.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    ReluKernel::ReluKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/relu.glsl";
        }
    ReluKernel::~ReluKernel(){}

    void ReluKernel::SetupAttr(const dlxnet::Attribute& attr){
        // single output
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);
    }
    void ReluKernel::Compute(TensorList& inputs, TensorList& outputs){
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();
        program_->SetRetVal(outputs);

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        program_->Run();
    }

    void ReluKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes.size(), 1);
        output_shapes[0] = input_shapes[0];
    }

    REGISTER_KERNEL_WITH_NAME(ReluKernel, "Relu");
}

